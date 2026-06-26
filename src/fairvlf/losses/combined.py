"""Detection loss and the combined FairVLF objective.

    L = L_det + lambda_DPC * L_DPC + lambda_DSA * L_DSA

L_det is binary cross-entropy on the (calibrated) fake score passed through a
sigmoid. The ablation switches let us zero out DPC and/or DSA while keeping the
same backbone and data — this is the controlled ablation that shows fairness
gains do not come merely from using a large VLM.
"""

import torch
import torch.nn.functional as F

from .dpc import dpc_loss
from .dsa import dsa_loss


def detection_loss(fake_score: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """BCE-with-logits on the fake score. labels: 1=fake, 0=real."""
    return F.binary_cross_entropy_with_logits(fake_score, labels.float())


def combined_loss(outputs: dict, batch: dict, cfg) -> tuple:
    """Assemble the full objective from precomputed scores.

    Args:
        outputs: dict with
            'score_fused'   (B,)   fused fake score (forensic prompt ensemble)
            'score_neutral' (B,)   score under q0
            'score_variants'(B,V)  scores under demographic variants
        batch: dict with 'labels', 'groups', 'generators'
        cfg: the resolved config object/dict.

    Returns:
        (total_loss, components_dict) for logging.
    """
    labels = batch["labels"]
    groups = batch["groups"]
    generators = batch["generators"]

    l_det = detection_loss(outputs["score_fused"], labels)

    use_dpc = cfg["ablation"]["use_dpc"] if "ablation" in cfg else True
    use_dsa = cfg["ablation"]["use_dsa"] if "ablation" in cfg else True

    l_dpc = (dpc_loss(outputs["score_neutral"],
                      outputs["score_variants"],
                      epsilon=cfg["loss"]["dpc_epsilon"])
             if use_dpc else l_det.new_tensor(0.0))

    l_dsa = (dsa_loss(outputs["score_fused"], groups, labels, generators,
                      bandwidth=cfg["loss"]["mmd_bandwidth"])
             if use_dsa else l_det.new_tensor(0.0))

    total = (l_det
             + cfg["loss"]["lambda_dpc"] * l_dpc
             + cfg["loss"]["lambda_dsa"] * l_dsa)

    return total, {
        "loss_total": float(total.detach()),
        "loss_det": float(l_det.detach()),
        "loss_dpc": float(l_dpc.detach()),
        "loss_dsa": float(l_dsa.detach()),
    }
