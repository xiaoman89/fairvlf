# Data layout and manifest format

FairVLF reads images through a **manifest CSV**. The images themselves live
under a root directory you configure (`data.image_root` in the YAML); the
manifest stores paths relative to that root plus the labels.

## Directory layout (suggested)

```
data/
├── images/
│   ├── real/                      # FairFace real faces
│   │   └── <race>/<file>.jpg
│   └── fake/
│       ├── sd15/<file>.png
│       ├── sdxl/<file>.png
│       ├── flux_schnell/<file>.png
│       ├── flux_dev/<file>.png
│       ├── sd35/<file>.png
│       └── qwen_image/<file>.png
├── manifest_train.csv
├── manifest_smoke.csv
└── ...
```

## Manifest columns

| Column       | Required | Values                                                        |
|--------------|----------|---------------------------------------------------------------|
| `image_path` | yes      | path relative to `image_root`, e.g. `fake/flux_dev/0001.png`  |
| `label`      | yes      | `real` / `fake` (or `0` / `1`)                                |
| `group`      | yes      | integer demographic group id, `0 .. num_groups-1`             |
| `gender`     | no       | `male` / `female` (used to build the q_attr phrase)           |
| `race`       | no       | FairFace race string (used to build the q_attr phrase)        |
| `generator`  | no       | one of the generator names above; `real` for real faces       |

Real faces have no generator, so set `generator = real`; the loader maps that
to id `-1`, and the DSA loss treats real faces separately (aligned by group and
label only, not by generator).

## Group ids

FairFace defines 7 race categories; combined with 2 genders this gives the
14 demographic cells FairFrontier balances. Assign a stable integer to each
cell and keep that mapping fixed across all runs (record it in the manifest or
a small JSON next to it).

## Generating a tiny manifest for the smoke test

`scripts/make_dummy_manifest.py` writes a small synthetic manifest + blank
images so you can confirm the data loader works without touching the real
dataset. See that script's header for usage.
