"""Collate raw dataset items into a batch the model consumes.

Images are kept as a list of PIL images (the processor handles them); labels,
groups, and generators become tensors.
"""

import torch


def collate(batch_items):
    return {
        "images": [b["image"] for b in batch_items],
        "labels": torch.tensor([b["label"] for b in batch_items], dtype=torch.long),
        "groups": torch.tensor([b["group"] for b in batch_items], dtype=torch.long),
        "generators": torch.tensor([b["generator"] for b in batch_items], dtype=torch.long),
        "demographic": [b["demographic"] for b in batch_items],
    }
