# Development log

## Day 1

- Initialized repository and folder structure.
- Added environment and dataset conventions.
- Created baseline project files and setup scripts.
- Verified the local dataset is attached under `data/raw`.
- Confirmed the category is `transistor` with 213 training images and 100 test images.
- Generated a dataset contact sheet with 5 normal and 5 defective examples and masks overlayed.

Notes:
- Data will live under `data/raw/` unless a different root is passed via `ANOMALY_DATA_DIR`.
- The transistor split uses defect folders: `bent_lead`, `cut_lead`, `damaged_case`, and `misplaced`.
- Next step: implement PatchCore feature extraction, memory bank, scoring, and evaluation.
