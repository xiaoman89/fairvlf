"""FairVLF model: a vision-language backbone + LoRA + a calibrated fake score.

The fake score for a prompt q is

    s_raw(x, q) = log p(fake | x, q) - log p(real | x, q)

computed from the next-token logits at the answer position, restricted to the
'real' and 'fake' token ids. A learnable affine calibration

    s(x, q) = tau * s_raw + beta

absorbs the instruct model's intrinsic A/B token bias.

NOTE: the exact mechanics of obtaining answer-position logits depend on the
backbone's chat template and processor. The forward pass below is written for
the Qwen2.5-VL family; adapt build_inputs() if you swap backbones. The smoke
test exercises the full code path on a few images.
"""

import torch
import torch.nn as nn


class FairVLFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = None       # set in load()
        self.processor = None      # set in load()
        self.real_token_id = None
        self.fake_token_id = None

        # Learnable score calibration (tiny: 2 scalars).
        if cfg["model"].get("calibrate_score", True):
            self.tau = nn.Parameter(torch.tensor(1.0))
            self.beta = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("tau", torch.tensor(1.0))
            self.register_buffer("beta", torch.tensor(0.0))

    # ------------------------------------------------------------------ load
    def load(self):
        """Load backbone + processor and attach LoRA. Imported lazily so the
        rest of the package can be imported without heavy deps installed."""
        from transformers import AutoProcessor
        from peft import LoraConfig, get_peft_model

        name = self.cfg["model"]["backbone"]
        self.processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        model_cls = _resolve_vlm_class()
        self.backbone = model_cls.from_pretrained(
            name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )

        lcfg = self.cfg["model"]["lora"]
        peft_cfg = LoraConfig(
            r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
            target_modules=lcfg["target_modules"], bias="none",
            task_type="CAUSAL_LM",
        )
        self.backbone = get_peft_model(self.backbone, peft_cfg)

        # Resolve the single-token ids for the answer labels.
        tok = self.processor.tokenizer
        self.real_token_id = _single_token_id(tok, self.cfg["model"]["label_tokens"]["real"])
        self.fake_token_id = _single_token_id(tok, self.cfg["model"]["label_tokens"]["fake"])
        return self

    # --------------------------------------------------------------- scoring
    def score_for_prompt(self, images, prompt_text):
        """Calibrated fake score for a batch of images under one prompt.

        Returns a (B,) tensor. Implementation detail: we build chat inputs that
        end right before the answer token, run a forward pass, and read the
        logits for 'real' vs 'fake' at the final position.
        """
        inputs = self.build_inputs(images, prompt_text)
        out = self.backbone(**inputs)
        # logits: (B, T, V); take the last position's distribution.
        last_logits = out.logits[:, -1, :]
        log_probs = torch.log_softmax(last_logits, dim=-1)
        s_raw = log_probs[:, self.fake_token_id] - log_probs[:, self.real_token_id]
        return self.tau * s_raw + self.beta

    def build_inputs(self, images, prompt_text):
        """Construct processor inputs ending at the answer position.

        Uses the backbone's chat template. Returns a dict of tensors on the
        model's device.
        """
        messages = [
            [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text + " Answer with a single word: real or fake."},
            ]}]
            for _ in images
        ]
        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        inputs = self.processor(
            text=texts, images=[[img] for img in images],
            return_tensors="pt", padding=True,
        )
        device = next(self.backbone.parameters()).device
        return {k: v.to(device) for k, v in inputs.items()}

    # --------------------------------------------------------------- forward
    def forward(self, batch):
        """Compute all scores needed by the combined loss.

        Returns dict with 'score_fused', 'score_neutral', 'score_variants'.
        """
        images = batch["images"]
        p = self.cfg["prompts"]

        # Forensic Prompt Ensemble: average score over forensic prompts.
        forensic_scores = [self.score_for_prompt(images, q) for q in p["forensic"]]
        score_fused = torch.stack(forensic_scores, dim=0).mean(dim=0)

        # DPC variants: neutral vs {fair, attr}.
        score_neutral = self.score_for_prompt(images, p["neutral"])
        score_fair = self.score_for_prompt(images, p["fair"])
        attr_prompts_filled = [p["attr"].format(demographic=d) for d in batch["demographic"]]
        # attr is per-sample; score each individually then stack.
        score_attr = torch.stack([
            self.score_for_prompt([img], ap)[0]
            for img, ap in zip(images, attr_prompts_filled)
        ])
        score_variants = torch.stack([score_fair, score_attr], dim=1)  # (B, 2)

        return {
            "score_fused": score_fused,
            "score_neutral": score_neutral,
            "score_variants": score_variants,
        }


def _resolve_vlm_class():
    """Return a usable vision-language model class across transformers versions."""
    import transformers
    candidates = [
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ]
    for name in candidates:
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise ImportError("No compatible VLM class found. Tried: " + ", ".join(candidates))


def _single_token_id(tokenizer, word: str) -> int:
    """Return the token id for a word, asserting it is a single token.

    real/fake are chosen precisely because they tokenize to one token in most
    vocabularies; if a backbone splits them, raise so the user picks others.
    """
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) != 1:
        # Try with a leading space (many BPE tokenizers prefix a space).
        ids_sp = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids_sp) == 1:
            return ids_sp[0]
        raise ValueError(
            f"Label word {word!r} is not a single token "
            f"(got {ids}). Pick a single-token label in the config."
        )
    return ids[0]
