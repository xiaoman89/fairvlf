"""Distributional Score Alignment (DSA) loss — the visual-side fairness term.

Aligns the score distribution across demographic groups within the same
(generator, label) cell using Maximum Mean Discrepancy (MMD). Unlike matching
only mean+variance, MMD is sensitive to distribution shape, so bimodal/long-tail
score distributions (common for VLM log-prob differences) are aligned correctly.

For fake faces we align within each (generator, group, label) cell; for real
faces (which have no generator) we align within (group, label=0) only.
"""

import torch


def _gaussian_kernel(x: torch.Tensor, y: torch.Tensor, bandwidth: float) -> torch.Tensor:
    """Gaussian (RBF) kernel matrix between sample sets x (N,) and y (M,)."""
    x = x.unsqueeze(1)          # (N, 1)
    y = y.unsqueeze(0)          # (1, M)
    dist_sq = (x - y) ** 2      # (N, M)
    return torch.exp(-dist_sq / (2.0 * bandwidth ** 2 + 1e-12))


def _median_bandwidth(a: torch.Tensor, b: torch.Tensor) -> float:
    """Median heuristic for the RBF bandwidth, computed on pooled samples."""
    pooled = torch.cat([a, b])
    if pooled.numel() < 2:
        return 1.0
    diffs = (pooled.unsqueeze(1) - pooled.unsqueeze(0)).abs()
    med = torch.median(diffs[diffs > 0]) if (diffs > 0).any() else torch.tensor(1.0)
    return float(med.clamp(min=1e-3))


def mmd_squared(a: torch.Tensor, b: torch.Tensor, bandwidth="median") -> torch.Tensor:
    """Unbiased-ish squared MMD between two 1-D sample sets a and b.

    Returns a scalar tensor. If either set has <2 samples, returns 0 (cannot
    estimate a distribution from a single point).
    """
    if a.numel() < 2 or b.numel() < 2:
        return a.new_tensor(0.0)
    bw = _median_bandwidth(a, b) if bandwidth == "median" else float(bandwidth)
    k_aa = _gaussian_kernel(a, a, bw).mean()
    k_bb = _gaussian_kernel(b, b, bw).mean()
    k_ab = _gaussian_kernel(a, b, bw).mean()
    return k_aa + k_bb - 2.0 * k_ab


def dsa_loss(scores: torch.Tensor,
             groups: torch.Tensor,
             labels: torch.Tensor,
             generators: torch.Tensor,
             real_generator_id: int = -1,
             bandwidth="median") -> torch.Tensor:
    """Average pairwise MMD across demographic groups within matched cells.

    Args:
        scores:     (B,) calibrated fake scores.
        groups:     (B,) demographic group id per sample.
        labels:     (B,) 0 = real, 1 = fake.
        generators: (B,) generator id per sample; real samples use
                    real_generator_id.
        real_generator_id: the id used to mark real faces.
        bandwidth:  'median' or a float.

    Returns:
        Scalar loss = mean over all valid cell/group-pair MMDs.
    """
    device = scores.device
    terms = []

    # --- Fake faces: align groups within each (generator, label=1) cell ---
    fake_mask = labels == 1
    if fake_mask.any():
        gen_ids = torch.unique(generators[fake_mask])
        for g in gen_ids:
            cell = fake_mask & (generators == g)
            terms += _pairwise_group_mmd(scores, groups, cell, bandwidth, device)

    # --- Real faces: align groups within (label=0), ignoring generator ---
    real_mask = labels == 0
    if real_mask.any():
        terms += _pairwise_group_mmd(scores, groups, real_mask, bandwidth, device)

    if not terms:
        return scores.new_tensor(0.0)
    return torch.stack(terms).mean()


def _pairwise_group_mmd(scores, groups, cell_mask, bandwidth, device):
    """All pairwise group MMDs inside one cell (a boolean mask)."""
    out = []
    present = torch.unique(groups[cell_mask])
    present = present.tolist()
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            a = scores[cell_mask & (groups == present[i])]
            b = scores[cell_mask & (groups == present[j])]
            out.append(mmd_squared(a, b, bandwidth))
    return out
