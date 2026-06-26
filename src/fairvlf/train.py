"""FairVLF training entry point.

Usage:
    python -m fairvlf.train --config configs/smoke_test.yaml

Everything is driven by the YAML config. The run writes a timestamped directory
under runs/ with the resolved config, provenance (git commit, seed), a full log,
and a per-step metrics CSV.
"""

import argparse

import yaml
import torch
from torch.utils.data import DataLoader

from fairvlf.utils.repro import set_seed, make_run_dir, save_provenance
from fairvlf.utils.logging_utils import get_logger, MetricsWriter
from fairvlf.data.dataset import FairFrontierDataset
from fairvlf.data.collate import collate
from fairvlf.losses.combined import combined_loss


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    run_dir = make_run_dir(cfg["run_name"])
    logger = get_logger(run_dir)
    metrics = MetricsWriter(run_dir)
    save_provenance(run_dir, cfg)

    set_seed(cfg["train"]["seed"])
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Config: {args.config}")

    # --- Data ---
    ds = FairFrontierDataset(cfg["data"]["manifest_path"], cfg["data"]["image_root"])
    loader = DataLoader(
        ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        collate_fn=collate, num_workers=2, drop_last=True,
    )
    logger.info(f"Dataset size: {len(ds)}")

    # --- Model (lazy heavy import) ---
    from fairvlf.models.fairvlf_model import FairVLFModel
    model = FairVLFModel(cfg).load()
    device = cfg["train"]["device"]
    model.to(device)
    model.train()

    # Only LoRA + calibration params require grad; optimise those.
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])

    logger.info(f"Trainable params: {sum(p.numel() for p in params):,}")

    # --- Train loop ---
    step = 0
    accum = cfg["train"]["grad_accum"]
    max_steps = cfg["train"]["max_steps"]
    opt.zero_grad()

    while step < max_steps:
        for batch in loader:
            outputs = model(batch)
            loss, comps = combined_loss(outputs, batch, cfg)
            (loss / accum).backward()

            if (step + 1) % accum == 0:
                opt.step()
                opt.zero_grad()

            if step % cfg["train"]["log_every"] == 0:
                logger.info(
                    f"step {step:5d} | total {comps['loss_total']:.4f} "
                    f"| det {comps['loss_det']:.4f} "
                    f"| dpc {comps['loss_dpc']:.4f} "
                    f"| dsa {comps['loss_dsa']:.4f}"
                )
                metrics.log(step, comps)

            save_every = cfg["train"]["save_every"]
            if save_every and step > 0 and step % save_every == 0:
                _save_checkpoint(model, run_dir, step, logger)

            step += 1
            if step >= max_steps:
                break

    # Final checkpoint (full run only; smoke test sets save_every=0).
    if cfg["train"]["save_every"]:
        _save_checkpoint(model, run_dir, step, logger)

    metrics.close()
    logger.info("Training complete.")


def _save_checkpoint(model, run_dir, step, logger):
    import os
    ckpt_dir = os.path.join(run_dir, "checkpoints", f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    # Save only the LoRA adapter + calibration scalars (small).
    model.backbone.save_pretrained(ckpt_dir)
    torch.save({"tau": model.tau, "beta": model.beta},
               os.path.join(ckpt_dir, "calibration.pt"))
    logger.info(f"Saved checkpoint: {ckpt_dir}")


if __name__ == "__main__":
    main()
