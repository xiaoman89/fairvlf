"""Self-contained sanity checks for the FairVLF losses.

Runs on CPU with synthetic tensors — no GPU, no model download required. This
is the fastest way to confirm the loss math is wired correctly before renting a
cloud GPU.

Run:
    python scripts/test_losses.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from fairvlf.losses.dpc import dpc_loss
from fairvlf.losses.dsa import dsa_loss, mmd_squared
from fairvlf.losses.combined import detection_loss


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        raise AssertionError(name)


def test_dpc():
    # If variants equal the neutral score, loss should be 0.
    neutral = torch.tensor([1.0, -2.0, 0.5])
    variants = neutral.unsqueeze(1).repeat(1, 2)
    check("DPC zero when prompts agree", torch.isclose(dpc_loss(neutral, variants), torch.tensor(0.0)))

    # Small deviation within epsilon -> still 0.
    variants_small = variants + 0.05
    check("DPC ignores within-tolerance shift",
          torch.isclose(dpc_loss(neutral, variants_small, epsilon=0.1), torch.tensor(0.0)))

    # Large deviation beyond epsilon -> positive.
    variants_big = variants + 1.0
    check("DPC penalises large shift", dpc_loss(neutral, variants_big, epsilon=0.1) > 0)


def test_mmd():
    torch.manual_seed(0)
    a = torch.randn(200)
    b = torch.randn(200)
    c = torch.randn(200) + 5.0  # shifted distribution

    same = mmd_squared(a, b)
    diff = mmd_squared(a, c)
    check("MMD smaller for same distribution than shifted", same < diff)
    check("MMD non-negative-ish for identical-ish", same >= -1e-4)


def test_dsa():
    torch.manual_seed(0)
    B = 120
    scores = torch.randn(B)
    groups = torch.randint(0, 3, (B,))
    labels = torch.randint(0, 2, (B,))
    generators = torch.where(labels == 0, torch.full((B,), -1), torch.randint(0, 2, (B,)))

    loss = dsa_loss(scores, groups, labels, generators)
    check("DSA returns a finite scalar", torch.isfinite(loss))

    # If all groups have identical scores, DSA should be ~0.
    scores_same = torch.zeros(B)
    loss_same = dsa_loss(scores_same, groups, labels, generators)
    check("DSA ~0 when scores identical across groups", loss_same < 1e-3)


def test_detection():
    score = torch.tensor([5.0, -5.0])      # confident fake, confident real
    labels = torch.tensor([1, 0])
    loss = detection_loss(score, labels)
    check("Detection loss low for correct confident preds", loss < 0.05)


if __name__ == "__main__":
    print("Running FairVLF loss sanity checks (CPU, no model)...\n")
    test_dpc()
    test_mmd()
    test_dsa()
    test_detection()
    print("\nAll loss checks passed.")
