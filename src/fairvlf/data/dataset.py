"""FairFrontier dataset.

Reads a manifest CSV with one row per image. Expected columns (see docs/DATA.md):

    image_path   relative path under image_root
    label        'real' or 'fake'  (or 1/0)
    group        demographic group id (0..num_groups-1) or a group name
    gender       'male' / 'female'
    race         FairFace race string (used to build the demographic phrase)
    generator    generator name for fakes; 'real' for real faces

The dataset returns the raw PIL image plus integer label/group/generator ids.
Prompt text construction lives in build_demographic_phrase().
"""

import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


# Stable id assignment for generators so ids are reproducible across runs.
GENERATOR_IDS = {
    "real": -1,
    "sd15": 0, "sdxl": 1, "flux_schnell": 2,
    "flux_dev": 3, "sd35": 4, "qwen_image": 5,
}


def build_demographic_phrase(race: str, gender: str) -> str:
    """Turn FairFace labels into a natural phrase, e.g. 'East Asian female'.

    Used only to fill the q_attr training/diagnostic prompt. Always uses the
    true label (never a deliberately wrong one).
    """
    race = (race or "").strip().replace("_", " ")
    gender = (gender or "").strip().lower()
    race = race if race else "person"
    return f"{race} {gender}".strip()


def _to_label_int(v) -> int:
    if isinstance(v, str):
        return 1 if v.strip().lower() == "fake" else 0
    return int(v)


def _to_generator_id(v) -> int:
    if isinstance(v, str):
        return GENERATOR_IDS.get(v.strip().lower(), -1)
    return int(v)


class FairFrontierDataset(Dataset):
    def __init__(self, manifest_path: str, image_root: str):
        self.df = pd.read_csv(manifest_path)
        self.image_root = image_root
        required = {"image_path", "label", "group"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row["image_path"])
        image = Image.open(img_path).convert("RGB")

        race = row.get("race", "")
        gender = row.get("gender", "")
        return {
            "image": image,
            "label": _to_label_int(row["label"]),
            "group": int(row["group"]),
            "generator": _to_generator_id(row.get("generator", "real")),
            "demographic": build_demographic_phrase(race, gender),
        }
