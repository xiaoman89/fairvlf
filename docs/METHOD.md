# FairVLF method

FairVLF fine-tunes a vision-language backbone (Qwen2.5-VL-7B-Instruct) with LoRA
so that its real/fake decision rests on forensic evidence rather than
demographic appearance or demographic wording.

## Fake score

For an image `x` and prompt `q`, the model scores the answer tokens `real` and
`fake` (single tokens). The raw fake score is

```
s_raw(x, q) = log p(fake | x, q) - log p(real | x, q)
```

A learnable affine calibration absorbs the instruct model's intrinsic token
bias:

```
s(x, q) = tau * s_raw(x, q) + beta
```

The task name and data scope remain "AI-generated face detection / full-face
synthesis"; only the output token labels follow the deepfake-forensics
convention of real/fake (both are single tokens, which keeps the score
definition clean and calibration stable).

## Objective

```
L = L_det + lambda_DPC * L_DPC + lambda_DSA * L_DSA
```

### L_det — detection
Binary cross-entropy on the calibrated, forensic-prompt-ensembled fake score.

### L_DPC — Demographic Prompt Consistency (language-side fairness)
Margin-tolerant penalty on the score shift caused by demographic wording:

```
L_DPC = E_x  sum_{q in {q_fair, q_attr}}  [ max(0, |s(x,q) - s(x,q0)| - eps) ]^2
```

`q0` is neutral, `q_fair` instructs the model to ignore demographics, `q_attr`
injects the true demographic phrase. `q_attr` is used only in training and
diagnosis — never at inference, and never with a deliberately wrong phrase.

### L_DSA — Distributional Score Alignment (visual-side fairness)
MMD alignment of the score distribution across demographic groups within each
matched cell. Fake faces are aligned within (generator, group, label); real
faces within (group, label) only. MMD is used (rather than mean+variance
matching) because VLM score distributions are often bimodal/long-tailed.

## Prompts

- **Forensic Prompt Ensemble (FPE):** five demographic-free forensic prompts;
  the detection score is their average. This is a backbone component, not a
  fairness claim.
- **Demographic variants:** q0 / q_fair / q_attr drive DPC (and the diagnostic
  measurement of language-side bias).

## Controlled ablation

The single most important experiment: with the **same backbone, same data, same
compute**, toggle the loss terms.

| Config           | L_det | L_DPC | L_DSA |
|------------------|:-----:|:-----:|:-----:|
| Naive VLM-LoRA   |   y   |       |       |
| + DPC only       |   y   |   y   |       |
| + DSA only       |   y   |       |   y   |
| FairVLF (full)   |   y   |   y   |   y   |

Set `ablation.use_dpc` / `ablation.use_dsa` in the config. This shows fairness
gains do not come merely from using a large VLM.

## Inference

VLM + LoRA + a fixed forensic prompt (or the FPE average). No demographic label
is needed at inference, and no demographic phrase is injected.
