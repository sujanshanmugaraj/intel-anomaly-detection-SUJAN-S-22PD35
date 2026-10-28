# Chip/Package Anomaly Detection with PatchCore

This project implements a PatchCore-style anomaly detection pipeline for chip/package images, with a strong verified baseline and a synthetic defect robustness evaluation. It was built as a practical hackathon-ready pipeline for real-world defect detection and localization on semiconductor-related imagery.

## Why this project

Modern manufacturing and semiconductor inspection rely on detecting small, subtle, and often rare defects in images of chips, packages, or board components. Traditional rule-based pipeline inspection can be brittle, while deep feature-based anomaly detection is much more flexible and scalable.

This project targets that gap by using a pretrained feature extractor and a PatchCore memory bank of normal patterns, then scoring anomalies against this learned normal representation.

## What was implemented

The project includes the full workflow for:

- validating a local dataset,
- checking the folder structure and counts,
- generating a contact sheet for image QA,
- confirming the PatchCore feature extraction path,
- evaluating anomaly detection on real defect data,
- generating synthetic anomalies to test robustness,
- and documenting the workflow end-to-end.

## Project architecture

- `src/config.py`: project and dataset root configuration
- `src/dataset_loader.py`: dataset discovery and metadata helpers
- `src/patchcore.py`: PatchCore implementation and scoring logic
- `src/synthetic_defects.py`: synthetic defect generator for robustness testing
- `scripts/check_dataset.py`: dataset validation and counts
- `scripts/contact_sheet.py`: preview visual QA of normal and defective samples
- `scripts/run_patchcore_baseline.py`: full image-level and pixel-level evaluation
- `scripts/generate_synthetic_defects.py`: synthetic anomalies and evaluation loop
- `tests/test_synthetic_defects.py`: lightweight validation tests

## Verified results

The pipeline was validated in a clean Python environment with these results on the transistor category:

- Image AUROC: 0.9979
- Pixel AUROC: 0.9314
- Synthetic clean-vs-defect AUROC: 0.9859
- Test status: 2 tests passed

These metrics indicate a strong anomaly detection baseline on real defects and a good robustness signal under synthetic defect generation.

## Folder structure

```text
.
├── data/
│   └── raw/
│       └── mvtec_ad/
│           └── transistor/
├── docs/
│   └── END_TO_END_DOCUMENTATION.md
├── experiments/
│   └── synthetic_defects/
├── results/
│   ├── benchmark_summary.md
│   └── contact_sheet_transistor.png
├── scripts/
│   ├── check_dataset.py
│   ├── contact_sheet.py
│   ├── generate_synthetic_defects.py
│   └── run_patchcore_baseline.py
├── src/
│   ├── config.py
│   ├── dataset_loader.py
│   ├── patchcore.py
│   └── synthetic_defects.py
├── tests/
│   └── test_synthetic_defects.py
├── README.md
├── requirements.txt
├── LOG.md
└── .venv/
```

## Dataset attachment

Place the local dataset in a MVTec-style structure under the project root or set it using an environment variable.

### Preferred structure

```text
data/
  raw/
    mvtec_ad/
      transistor/
        train/
          good/
        test/
          good/
          defect_type_1/
          defect_type_2/
        ground_truth/
          defect_type_1/
          defect_type_2/
```

### Environment variable override

```powershell
$env:ANOMALY_DATA_DIR = "C:\path\to\your\dataset"
```

## Environment setup

The project was run in a clean environment because the default machine install of PyTorch caused a Windows DLL initialization issue. The stable fix was to create an isolated venv and install compatible versions.

### Commands

```powershell
cd "C:\Users\Ritanya\PROJECTS\Anomaly detection"
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch==2.4.1 torchvision==0.19.1 scikit-learn==1.4.2 numpy==1.26.4 pillow==10.3.0 faiss-cpu==1.10.0 opencv-python==4.10.0.84 matplotlib==3.8.4 tqdm==4.9.0 pandas==2.2.2
```

### Verification

```powershell
.\.venv\Scripts\python -c "import torch, torchvision; print('torch', torch.__version__); print('torchvision', torchvision.__version__)"
```

Expected output:

```text
torch 2.4.1+cpu
torchvision 0.19.1+cpu
```

## Quick start

### 1. Activate the environment

```powershell
cd "C:\Users\Ritanya\PROJECTS\Anomaly detection"
.\.venv\Scripts\Activate.ps1
```

### 2. Validate dataset structure

```powershell
.\.venv\Scripts\python scripts/check_dataset.py
```

### 3. Generate a contact sheet

```powershell
.\.venv\Scripts\python scripts/contact_sheet.py
```

### 4. Run the PatchCore baseline

```powershell
.\.venv\Scripts\python scripts/run_patchcore_baseline.py
```

### 5. Run synthetic defect validation

```powershell
.\.venv\Scripts\python scripts/generate_synthetic_defects.py
```

### 6. Run the regression tests

```powershell
.\.venv\Scripts\python -m unittest discover -s tests
```

## End-to-end workflow summary

The project followed a practical end-to-end pipeline:

1. dataset discovery and validation,
2. contact sheet generation for visual sanity check,
3. PatchCore feature extraction sanity check,
4. real anomaly evaluation across the dataset,
5. synthetic defect generation to stress-test robustness,
6. benchmark summary and documentation.

This makes the repo useful not just as a model implementation but as a reproducible project pipeline.

## Implementation notes

### PatchCore logic

The model uses a pretrained WideResNet50 backbone and extracts patch embeddings from intermediate feature maps. These embeddings are aggregated into a memory bank of normal representative features. For a test image, the nearest-neighbor distance to the memory bank is measured and the maximum patch response is used as the image anomaly score.

### Synthetic defects

Synthetic defects are generated using simple but effective patterns such as:

- blobs,
- scratches,
- line defects,
- stains,
- and speckle noise.

These are overlaid onto normal chip/package images to check whether the model is still able to separate clean and anomalous inputs.

## Hackathon readiness

This repository is ready for a hackathon submission and demo because it includes:

- a real dataset workflow,
- a baseline anomaly model,
- code for synthetic robustness testing,
- metrics and visual outputs,
- working validation steps,
- and a clear technical narrative.

## Detailed documentation

For the complete walkthrough, commands, outputs, and end-to-end notes, see [docs/END_TO_END_DOCUMENTATION.md](docs/END_TO_END_DOCUMENTATION.md).

## Notes

- Keep seeds fixed across experiments for reproducibility.
- Prefer the clean `.venv` environment for stable Windows PyTorch execution.
- The generated synthetic defect outputs are saved under `experiments/synthetic_defects/`.
- The benchmark summary is documented in [results/benchmark_summary.md](results/benchmark_summary.md).
