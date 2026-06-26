"""Build a small, balanced smoke-test subset from local FairFrontier data.

Reads the real FairFace faces and the generated fake faces, picks a few images
per (demographic group x source) cell, copies them into a single flat folder,
and writes a manifest CSV in the format FairVLF's data loader expects
(image_path / label / group / gender / race / generator).

This is for the *smoke test* only: a couple of hundred images to verify the
training code runs on the GPU. The full dataset is used later for real training.

Usage (run on your laptop, where the data lives):

    python scripts/prepare_smoke_data.py \
        --fairface_root "F:/fairface_data/images/train" \
        --fakes_root    "F:/fairfrontier/fakes" \
        --out_dir       "F:/fairvlf_smoke" \
        --per_cell 3

Then upload the entire out_dir to AutoDL.
"""

import os
import re
import csv
import shutil
import argparse
import random
from pathlib import Path

# Canonical 14 demographic groups (7 races x 2 genders).
RACES = ["Black", "East Asian", "Indian", "Latino Hispanic",
         "Middle Eastern", "Southeast Asian", "White"]
GENDERS = ["Female", "Male"]

# Map a generator *folder name* on disk to the canonical id string used in code.
GENERATOR_FOLDER_TO_NAME = {
    "SD1.5": "sd15",
    "SDXL": "sdxl",
    "FLUX.1-schnell": "flux_schnell",
    "FLUX.1-dev": "flux_dev",
    "SD3.5": "sd35",
    "Qwen-Image": "qwen_image",
}


def group_id(gender: str, race: str) -> int:
    """Stable 0..13 id for a (gender, race) cell."""
    return GENDERS.index(gender) * len(RACES) + RACES.index(race)


def norm(s: str) -> str:
    """Normalize labels: underscores -> spaces, trimmed."""
    return s.replace("_", " ").strip()


def collect_real(fairface_root: str, per_cell: int, rng):
    """Pick per_cell real faces for each (gender, race) group.

    Real filenames look like '352_Male.png' inside a race folder.
    """
    rows = []
    root = Path(fairface_root)
    for race_folder in sorted(os.listdir(root)):
        race_dir = root / race_folder
        if not race_dir.is_dir():
            continue
        race = norm(race_folder)
        if race not in RACES:
            continue
        # Bucket files by gender parsed from the filename.
        buckets = {"Male": [], "Female": []}
        for fname in os.listdir(race_dir):
            m = re.match(r"(\d+)_(Male|Female)\.png", fname, re.IGNORECASE)
            if not m:
                continue
            gender = m.group(2).capitalize()
            buckets[gender].append(fname)
        for gender, files in buckets.items():
            if not files:
                continue
            picked = rng.sample(files, min(per_cell, len(files)))
            for fname in picked:
                rows.append({
                    "src_abs": str(race_dir / fname),
                    "label": "real",
                    "group": group_id(gender, race),
                    "gender": gender.lower(),
                    "race": race,
                    "generator": "real",
                })
    return rows


def collect_fakes(fakes_root: str, per_cell: int, rng):
    """Pick per_cell fake faces for each (generator, gender, race) cell.

    Fake folders are <gen>/<Gender>-<Race>/, files like '10055_Black_Female.png'.
    We rely on the folder name for the group label (authoritative on disk).
    """
    rows = []
    root = Path(fakes_root)
    for gen_folder in sorted(os.listdir(root)):
        gen_dir = root / gen_folder
        if not gen_dir.is_dir():
            continue
        gen_name = GENERATOR_FOLDER_TO_NAME.get(gen_folder)
        if gen_name is None:
            print(f"  [skip] unknown generator folder: {gen_folder}")
            continue
        for cell_folder in sorted(os.listdir(gen_dir)):
            cell_dir = gen_dir / cell_folder
            if not cell_dir.is_dir():
                continue
            # cell_folder like 'Female-Black' -> gender, race
            try:
                gender_raw, race_raw = cell_folder.split("-", 1)
            except ValueError:
                continue
            gender = gender_raw.capitalize()
            race = norm(race_raw)
            if gender not in GENDERS or race not in RACES:
                continue
            files = [f for f in os.listdir(cell_dir) if f.lower().endswith(".png")]
            if not files:
                continue
            picked = rng.sample(files, min(per_cell, len(files)))
            for fname in picked:
                rows.append({
                    "src_abs": str(cell_dir / fname),
                    "label": "fake",
                    "group": group_id(gender, race),
                    "gender": gender.lower(),
                    "race": race,
                    "generator": gen_name,
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fairface_root", required=True,
                    help="e.g. F:/fairface_data/images/train")
    ap.add_argument("--fakes_root", required=True,
                    help="e.g. F:/fairfrontier/fakes")
    ap.add_argument("--out_dir", required=True,
                    help="output folder to create, e.g. F:/fairvlf_smoke")
    ap.add_argument("--per_cell", type=int, default=3,
                    help="images per (group) cell for real, and per "
                         "(generator,group) cell for fake")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    img_out = out / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    print("Collecting real faces...")
    real_rows = collect_real(args.fairface_root, args.per_cell, rng)
    print(f"  picked {len(real_rows)} real faces")

    print("Collecting fake faces...")
    fake_rows = collect_fakes(args.fakes_root, args.per_cell, rng)
    print(f"  picked {len(fake_rows)} fake faces")

    all_rows = real_rows + fake_rows
    if not all_rows:
        raise SystemExit("No images found. Check the --fairface_root / --fakes_root paths.")

    # Copy images into a flat folder with unique names, build manifest rows.
    manifest = []
    for i, r in enumerate(all_rows):
        src = Path(r["src_abs"])
        # Unique destination name keeps provenance readable.
        dest_name = f"{r['label']}_{r['generator']}_{r['group']:02d}_{i:05d}.png"
        rel = f"images/{dest_name}"
        shutil.copy2(src, img_out / dest_name)
        manifest.append({
            "image_path": rel,
            "label": r["label"],
            "group": r["group"],
            "gender": r["gender"],
            "race": r["race"],
            "generator": r["generator"],
        })

    manifest_path = out / "manifest_smoke.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "label", "group",
                                          "gender", "race", "generator"])
        w.writeheader()
        w.writerows(manifest)

    # Quick summary so you can eyeball the balance.
    print("\n=== Summary ===")
    print(f"Total images: {len(manifest)}")
    n_real = sum(1 for m in manifest if m["label"] == "real")
    print(f"  real: {n_real}, fake: {len(manifest) - n_real}")
    by_gen = {}
    for m in manifest:
        by_gen[m["generator"]] = by_gen.get(m["generator"], 0) + 1
    for g, c in sorted(by_gen.items()):
        print(f"  generator {g}: {c}")
    print(f"\nManifest: {manifest_path}")
    print(f"Images copied to: {img_out}")
    print(f"\nUpload the whole folder '{out}' to AutoDL.")


if __name__ == "__main__":
    main()
