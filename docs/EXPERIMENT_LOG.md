# Experiment log

A running record of experiments. Append a dated entry per run. Keep this in the
repo so results stay traceable (the matching `runs/<...>/provenance.json` has
the exact config + git commit).

> Template — copy the block below for each run.

---

## YYYY-MM-DD — <short description>

- **Config:** `configs/<name>.yaml`
- **Run dir:** `runs/<run_name>_<timestamp>/`
- **Git commit:** `<hash>` (from provenance.json)
- **Backbone:** Qwen2.5-VL-7B-Instruct + LoRA
- **Data:** <split / subset / size>
- **Ablation:** DPC=<on/off>, DSA=<on/off>
- **Result (utility):** AUC=__, ACC=__
- **Result (fairness):** worst-group AUC=__, FPR gap=__
- **Notes:** <what you observed, anything surprising, next step>

---

## (example) 2026-XX-XX — first smoke test

- **Config:** `configs/smoke_test.yaml`
- **Run dir:** `runs/smoke_test_2026XXXX_XXXXXX/`
- **Purpose:** verify pipeline runs end-to-end (not a real result).
- **Outcome:** loss computed and decreasing over 20 steps; no crashes.
- **Next step:** scale to a real subset, then full Split A run.
