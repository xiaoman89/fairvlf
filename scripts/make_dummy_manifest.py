"""Generate a tiny synthetic dataset (blank images + manifest) for smoke testing.

This lets you verify the data loader and (optionally) the full training path
without the real FairFrontier data. Creates:

    data/images/<...>.png   (small solid-color placeholder images)
    data/manifest_smoke.csv

Run:
    python scripts/make_dummy_manifest.py --n 56
"""

import os
import csv
import argparse
import random

from PIL import Image

GENERATORS = ["sd15", "sdxl", "flux_schnell", "flux_dev", "sd35", "qwen_image"]
RACES = ["East Asian", "White", "Black", "Indian",
         "Latino Hispanic", "Middle Eastern", "Southeast Asian"]
GENDERS = ["male", "female"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=56, help="total images (half real)")
    ap.add_argument("--root", default="data")
    args = ap.parse_args()

    img_dir = os.path.join(args.root, "images")
    os.makedirs(img_dir, exist_ok=True)
    rows = []

    for i in range(args.n):
        is_fake = i % 2 == 0
        race = random.choice(RACES)
        gender = random.choice(GENDERS)
        group = RACES.index(race) * 2 + GENDERS.index(gender)  # 0..13
        gen = random.choice(GENERATORS) if is_fake else "real"

        sub = f"fake/{gen}" if is_fake else "real"
        os.makedirs(os.path.join(img_dir, sub), exist_ok=True)
        rel = f"{sub}/img_{i:04d}.png"
        # Small solid-color placeholder (color varies so they aren't identical).
        color = (50 + i % 200, 80, 120)
        Image.new("RGB", (224, 224), color).save(os.path.join(img_dir, rel))

        rows.append({
            "image_path": rel,
            "label": "fake" if is_fake else "real",
            "group": group,
            "gender": gender,
            "race": race,
            "generator": gen,
        })

    manifest = os.path.join(args.root, "manifest_smoke.csv")
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {manifest}")
    print(f"Wrote {len(rows)} placeholder images under {img_dir}")


if __name__ == "__main__":
    main()
