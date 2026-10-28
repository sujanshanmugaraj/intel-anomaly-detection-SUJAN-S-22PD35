# End-to-End Project Documentation

## 1. Project objective

This project was built as a hackathon-ready anomaly detection pipeline for chip/package images. The goal was to:

- validate a local dataset,
- reproduce a strong baseline anomaly detector,
- generate synthetic defect samples,
- measure the model’s performance,
- and package the workflow so it can be run end-to-end on a fresh machine.

The project uses a PatchCore-style memory bank approach with a pretrained WideResNet50 backbone and local patch feature matching.

---

## 2. Problem setup and project flow

The workflow followed this sequence:

1. Create a functional Python environment.
2. Attach the local dataset and validate its structure.
3. Inspect dataset counts and sample images.
4. Generate a contact sheet for quick visual QA.
5. Build and test PatchCore feature extraction.
6. Run the actual anomaly scoring pipeline on the full transistor dataset.
7. Generate synthetic defects and evaluate robustness.
8. Package the results and documentation for the hackathon.

---

## 3. Dataset attachment and folder structure

The dataset must be placed in a MVTec-style structure under the project root. The code uses the default dataset root from the project config file:

- [src/config.py](../src/config.py)

Recommended structure:

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

If the dataset is elsewhere on the machine, set the environment variable before running the scripts:

```powershell
$env:ANOMALY_DATA_DIR = "C:\path\to\your\dataset"
```

---

## 4. Environment setup

The project was built and stabilized in a clean virtual environment because the initial machine-level PyTorch installation caused a Windows DLL initialization failure.

### Initial issue

The initial error was:

```text
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed.
Error loading ... torch\lib\c10.dll
```

This was resolved by creating a fresh isolated virtual environment and reinstalling a stable package set.

### Commands used

```powershell
cd "C:\Users\Ritanya\PROJECTS\Anomaly detection"
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch==2.4.1 torchvision==0.19.1 scikit-learn==1.4.2 numpy==1.26.4 pillow==10.3.0 faiss-cpu==1.10.0 opencv-python==4.10.0.84 matplotlib==3.8.4 tqdm==4.9.0 pandas==2.2.2
```

### Verification command

```powershell
cd "C:\Users\Ritanya\PROJECTS\Anomaly detection"
.\.venv\Scripts\python -c "import torch, torchvision; print('torch', torch.__version__); print('torchvision', torchvision.__version__)"
```

### Verified output

```text
torch 2.4.1+cpu
torchvision 0.19.1+cpu
```

---

## 5. Dataset validation

The dataset validator was used to ensure the annotation structure and counts were consistent with MVTec-style data.

### Relevant files

- [scripts/check_dataset.py](../scripts/check_dataset.py)
- [src/dataset_loader.py](../src/dataset_loader.py)

### Commands used

```powershell
cd "C:\Users\Ritanya\PROJECTS\Anomaly detection"
.\.venv\Scripts\python scripts/check_dataset.py
```

### Verified result

The transistor category was validated successfully with:

- train images: 213
- test images: 100
- defect types detected: multiple MVTec-style defect categories

This confirmed that the setup was correct for the next modeling step.

---

## 6. Contact-sheet generation for visual QA

A contact sheet was created to visually inspect normal and defective examples and their masks.

### Relevant file

- [scripts/contact_sheet.py](../scripts/contact_sheet.py)

### Command used

```powershell
cd "C:\Users\Ritanya\PROJECTS\Anomaly detection"
.\.venv\Scripts\python scripts/contact_sheet.py
```

### Result

The generated output was saved to:

- [results/contact_sheet_transistor.png](../results/contact_sheet_transistor.png)

This served as a quick human inspection pass to ensure that the dataset and masks looked valid before running the actual model evaluation.

---

## 7. PatchCore feature extraction and sanity check

The core logic lives in:

- [src/patchcore.py](../src/patchcore.py)

The sanity-check step validates the backbone output and patch extraction. This is where the model was tested before running the full dataset metrics.

### Command used

```powershell
cd "C:\Users\Ritanya\PROJECTS\Anomaly detection"
.\.venv\Scripts\python -c "from PIL import Image; from src.config import DATA_ROOT; from src.patchcore import PatchCoreModel, set_seed; set_seed(42); model = PatchCoreModel(resize=256, coreset_size=256); train = [Image.open(p).convert('RGB') for p in sorted((DATA_ROOT / 'transistor' / 'train' / 'good').glob('*.png'))[:10]]; mem = model.build_memory_bank(train); print('mem_shape', tuple(mem.shape)); print('sample_score', model.score_image(train[0], mem)[1])"
```

### Verified output

```text
mem_shape (256, 1536)
sample_score 16.839820861816406
```

This confirmed that:

- patch features were extracted correctly,
- the memory bank shape was valid,
- and the anomaly score computation path was functioning.

---

## 8. Full PatchCore baseline evaluation

This was the critical benchmark step. The entire model pipeline was run on the transistor dataset to compute image-level and pixel-level AUROC.

### Script used

- [scripts/run_patchcore_baseline.py](../scripts/run_patchcore_baseline.py)

### Command used

```powershell
cd "C:\Users\Ritanya\PROJECTS\Anomaly detection"
.\.venv\Scripts\python scripts/run_patchcore_baseline.py
```

### Verified output

```text
Loading pretrained WideResNet50 backbone...
Train images: 213
Stage2 shape: (512, 32, 32)
Stage3 shape: (1024, 16, 16)
Patch vector count: 1024
Memory bank shape: (512, 1536)
Evaluation items: 100
Image AUROC: 0.9979
Pixel AUROC: 0.9314
PatchCore baseline evaluation complete.
```

### Interpretation

This is a very strong result for a baseline model:

- image-level detection is highly accurate,
- localization is also solid,
- and the project is ready to be used as a credible hackathon anomaly detection solution.

---

## 9. Synthetic defect generation and robustness check

To make the project more robust and interesting for the hackathon, synthetic defect generation was added. This pipeline creates artificial chip-like defect patterns such as blobs, scratches, lines, stains, and speckles, then tests whether the model can separate them from clean samples.

### Relevant files

- [src/synthetic_defects.py](../src/synthetic_defects.py)
- [scripts/generate_synthetic_defects.py](../scripts/generate_synthetic_defects.py)

### Commands used

```powershell
cd "C:\Users\Ritanya\PROJECTS\Anomaly detection"
.\.venv\Scripts\python -m unittest discover -s tests
.\.venv\Scripts\python scripts/generate_synthetic_defects.py
```

### Verified output from synthetic pipeline

```text
Generated synthetic files: 40
Synthetic defect types: ('blob', 'scratch', 'line', 'stain', 'speckle')
Image AUROC on clean-vs-synthetic: 0.9859
Saved synthetic set to: C:\Users\Ritanya\PROJECTS\Anomaly detection\experiments\synthetic_defects\transistor
```

### Interpretation

This shows the model is not only strong on real defects but also remains effective on synthetic anomalies designed to resemble realistic defect patterns. That is important for a hackathon narrative around robustness and extendability.

---

## 10. Regression tests

The tests were kept intentionally lightweight but useful for checking the new synthetic defect logic.

### Test file

- [tests/test_synthetic_defects.py](../tests/test_synthetic_defects.py)

### Command used

```powershell
cd "C:\Users\Ritanya\PROJECTS\Anomaly detection"
.\.venv\Scripts\python -m unittest discover -s tests
```

### Verified output

```text
..
----------------------------------------------------------------------
Ran 2 tests in 0.009s

OK
```

---

## 11. Final project status and hackathon readiness

By the end of the work, the project had the following verified state:

- dataset was attached and validated,
- the model was trained or rather memory-bank built on normal images,
- the baseline anomaly detector produced strong AUROC scores,
- synthetic defects were generated and scored,
- documentation and reproducible commands were prepared,
- and the project was stabilized in a clean environment.

### Summary metrics

| Metric | Value |
| --- | ---: |
| Real image AUROC | 0.9979 |
| Real pixel AUROC | 0.9314 |
| Synthetic defect AUROC | 0.9859 |
| Test status | 2 tests passed |

This is a strong result for a hackathon and qualifies as a complete project story from setup to evaluation.

---

## 12. Recommended next step for presentation

If this project is being presented in a hackathon or review session, use the following flow:

1. Show the dataset structure and validation counts.
2. Show the contact sheet of normal vs defective examples.
3. Show the PatchCore baseline score.
4. Show the synthetic defect set and robustness result.
5. Explain that the solution is easy to extend to more categories or robust training regimes.

---

## 13. Useful commands summary

This is the compact command block to re-run the workflow from scratch:

```powershell
cd "C:\Users\Ritanya\PROJECTS\Anomaly detection"
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch==2.4.1 torchvision==0.19.1 scikit-learn==1.4.2 numpy==1.26.4 pillow==10.3.0 faiss-cpu==1.10.0 opencv-python==4.10.0.84 matplotlib==3.8.4 tqdm==4.9.0 pandas==2.2.2

.\.venv\Scripts\python scripts/check_dataset.py
.\.venv\Scripts\python scripts/contact_sheet.py
.\.venv\Scripts\python scripts/run_patchcore_baseline.py
.\.venv\Scripts\python -m unittest discover -s tests
.\.venv\Scripts\python scripts/generate_synthetic_defects.py
```

---

## 14. Key files overview

- [src/config.py](../src/config.py) — dataset root config
- [src/dataset_loader.py](../src/dataset_loader.py) — dataset enumeration and validation utilities
- [src/patchcore.py](../src/patchcore.py) — core model implementation
- [src/synthetic_defects.py](../src/synthetic_defects.py) — synthetic anomaly generator
- [scripts/check_dataset.py](../scripts/check_dataset.py) — dataset sanity checks
- [scripts/contact_sheet.py](../scripts/contact_sheet.py) — visual dataset QA
- [scripts/run_patchcore_baseline.py](../scripts/run_patchcore_baseline.py) — full model evaluation
- [scripts/generate_synthetic_defects.py](../scripts/generate_synthetic_defects.py) — synthetic defect robustness loop
- [tests/test_synthetic_defects.py](../tests/test_synthetic_defects.py) — regression tests
- [results/contact_sheet_transistor.png](../results/contact_sheet_transistor.png) — contact sheet result
- [results/benchmark_summary.md](../results/benchmark_summary.md) — benchmark summary

---

## 15. Closing note

This repository is now structured like a reproducible anomaly detection project that follows a real end-to-end workflow: dataset validation, model baseline, synthetic augmentation testing, and documentation. It is suitable for a strong hackathon submission and for future extensions such as more defect families, patch-level localization, or ablation studies.
