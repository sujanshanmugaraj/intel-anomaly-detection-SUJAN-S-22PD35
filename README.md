# Visual Anomaly Detection for Chip / Package Inspection (PatchCore)

This project detects visual anomalies — foreign objects such as a strand of hair, cracks, dust / sand particles and excess adhesive — on images of electronic components, **using only images of good parts for training**.

It is a simplified implementation of **PatchCore** ([Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, CVPR 2022](https://arxiv.org/abs/2106.08265)), evaluated on the public **MVTec AD `transistor`** dataset. On top of the paper, I added a **synthetic defect generator** that creates the specific defect types from the problem statement (hair, crack, dust, adhesive blob), so the detector could be tested on those scenarios even though real factory images could not be shared.

📄 Full step-by-step write-up: **[docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)**

---

## 1. Results at a glance

| Evaluation | Images | Metric | Result |
|---|---|---|---:|
| Real defects (MVTec AD transistor) — *is the image defective?* | 60 good + 40 defective | Image AUROC | **0.9979** |
| Real defects — *where is the defect?* | 40 defective, pixel masks | Pixel AUROC | **0.9314** |
| Synthetic hair / crack / dust / adhesive vs clean | 96 synthetic + 16 good | Image AUROC | **1.0000** |
| Unit tests | 4 tests | pass / fail | **4 / 4 passed** |

All numbers above were produced by the scripts in this repo on CPU; the raw console output of every run is saved in [`docs/run_logs/`](docs/run_logs/).

Screenshots of the detection output are in [section 4](#4-output-screenshots).

---

## Research paper explored

**PatchCore — *Towards Total Recall in Industrial Anomaly Detection***
Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, Peter Gehler — CVPR 2022 — [arXiv:2106.08265](https://arxiv.org/abs/2106.08265)

What I took from the paper:
- the idea of building a **memory bank of normal patch features** from a frozen, ImageNet-pretrained WideResNet50, so no defect images are needed for training;
- using **mid-level features (layer2 + layer3)** with **3×3 local neighbourhood aggregation**;
- **greedy coreset subsampling** to shrink the memory bank;
- scoring by **nearest-neighbour distance**, giving both an image score and a localisation heatmap.

I implemented a simplified version of the method; the exact differences are listed in [section 3](#differences-from-the-original-paper-simplifications).

## My contribution and ideas added on top of the paper

The paper only evaluates on the defect types that come with the MVTec dataset. The problem statement asks about **hair, cracks, dust and excess adhesive**, which are not in that dataset. My additions bridge that gap:

1. **Problem-specific synthetic defect generator** ([`src/synthetic_defects.py`](src/synthetic_defects.py)) — generates the defects named in the problem statement on top of good images: `hair` (foreign object), `crack`, `speckle` (dust / sand) and `blob` (excess adhesive), each with a saved pixel mask.
2. **Difficulty levels** — every defect is generated at `easy`, `medium` and `hard` (size × 0.75 / 1.0 / 1.35 and increasing intensity), in both bright and dark variants, to test how robust the detector is rather than reporting one number.
3. **Robustness evaluation** ([`scripts/generate_synthetic_defects.py`](scripts/generate_synthetic_defects.py)) — clean-vs-synthetic AUROC, broken down by defect type and difficulty.
4. **Reliability fixes in the PatchCore code** — the backbone now refuses to run with random weights if the pretrained weights fail to load; the greedy coreset keeps a running minimum distance so it only computes distances to the newly added point each step (much faster on 218k vectors); the pixel AUROC helper was corrected to use mask pixels as labels.
5. **Unit tests** checking the defect generator, the coreset on a hand-worked example and AUROC against a hand-computed value.
6. **Mapping to the real use case** — the table in section 2 links each production anomaly to the defect type used to test it.

**Ideas to add on next** (not yet implemented):
- **Tiled high-resolution scoring** — score the full 1024×1024 image in overlapping tiles so a thin hair or a single dust grain is not lost when downscaling to 256×256.
- **Realistic, low-contrast defects** — curved, blurred hair; cracks that follow surface texture; adhesive modelled as a slightly wider existing glue bead — then plot detection rate against defect contrast to find the detection limit.
- **Excess-adhesive as a shape check** — for "too much gum", compare the glue region's area/outline with the expected one, since a patch-feature method may miss a small volume increase.
- **Operating threshold** — pick a threshold on validation data and report miss rate and false alarms per 1000 parts, which is what a production line needs.
- **Complete the paper** — Gaussian heatmap smoothing, 1 % coreset with random projection, centre crop and re-weighted image score, to improve pixel AUROC.
- **Speed** — FAISS nearest-neighbour search and GPU inference for line-rate inspection.

---

## 2. The problem and how this project maps to it

The task: high-resolution photos are taken between manufacturing steps; detect when something is wrong with the package surface. Defective examples are rare and varied, so the method must work **without labelled defect images**.

| Anomaly in the problem statement | How it is covered here |
|---|---|
| Foreign object — **strand of human hair** | Synthetic `hair` defect (thin straight line, 3–5 px wide on a 1024 px image) |
| **Crack** on the surface | Synthetic `crack` defect (branching polyline) + real `damaged_case` images |
| **Dust / sand particles** | Synthetic `speckle` defect (5–30 small spots) |
| **Too much adhesive / gum** | Synthetic `blob` defect (soft elliptical stain) — an approximation |
| Other real-world defects | Real MVTec classes: `bent_lead`, `cut_lead`, `damaged_case`, `misplaced` |

**Why the transistor dataset?** The real factory images are confidential, so I used [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), the standard public benchmark for industrial anomaly detection. Its `transistor` category is the closest to a chip package: a black plastic body with metal leads on a board, photographed from above at 1024×1024.

---

## 3. Approach (PatchCore in short)

```
 Training (good images only)                     Testing (any image)
 ───────────────────────────                     ───────────────────
 good image ─► WideResNet50 (frozen,             test image ─► same feature extractor
               ImageNet weights)                                    │
               │                                                    ▼
               ▼                                        1024 patch vectors
   layer2 (32×32) + layer3 (16×16 → upsampled)                     │
               │  3×3 local average (neighbourhood)                 ▼
               ▼                                     distance of every patch to its
   1024 patch vectors × 1536 dims per image          nearest neighbour in the memory bank
               │                                                    │
               ▼                                     ┌──────────────┴──────────────┐
   218k vectors ─► greedy coreset ─► memory bank     ▼                             ▼
                                    (512 × 1536)  max patch distance        32×32 distance map
                                                  = image anomaly score     upsampled = heatmap
```

1. **Feature extraction.** A pretrained WideResNet50 is used as a frozen feature extractor — no training. Features from the mid-level blocks (layer2 and layer3) are taken because they capture local texture and shape without being too specific to ImageNet classes.
2. **Neighbourhood aggregation.** Each feature is averaged with its 3×3 neighbours so every patch vector carries some spatial context.
3. **Memory bank.** All patch vectors from the 213 good training images (~218 000 vectors) are reduced to 512 representative vectors with **greedy coreset subsampling** (repeatedly pick the vector farthest from everything picked so far).
4. **Scoring.** For a new image, each patch is compared to its nearest neighbour in the memory bank. A patch that looks unlike anything seen in good images gets a high distance. The **largest patch distance is the image score**; the full distance map, upsampled to the image size, is the **heatmap**.

### What I added beyond the paper
- **Synthetic defect generator** ([`src/synthetic_defects.py`](src/synthetic_defects.py)): `hair`, `crack`, `speckle` (dust) and `blob` (adhesive) at three difficulty levels (easy / medium / hard), bright or dark, with a saved pixel mask for every image.
- **Robustness evaluation** ([`scripts/generate_synthetic_defects.py`](scripts/generate_synthetic_defects.py)): clean-vs-synthetic AUROC, broken down per defect type and difficulty.
- **Heatmap visualisation** ([`scripts/visualize_heatmaps.py`](scripts/visualize_heatmaps.py)) with one shared colour scale across all images.

### Differences from the original paper (simplifications)

| PatchCore paper | This implementation |
|---|---|
| Resize to 256, centre-crop to 224 | Resize to 256, no crop |
| Coreset = 1–10 % of all patches | Fixed 512 patches (≈ 0.2 %) |
| Random projection to speed up coreset search | Exact distances |
| Image score re-weighted by neighbourhood (Eq. 7) | Plain maximum patch distance |
| Heatmap smoothed with Gaussian σ = 4 | No smoothing |

---

## 4. Output screenshots

### 4.1 Detection heatmaps (real and synthetic defects)

Left: input image and its anomaly score. Middle: anomaly heatmap over the image (blue = normal, red = anomalous). Right: ground-truth mask for real defects, or the raw score map for synthetic ones. **All rows share the same colour scale**, so the clean image correctly stays blue.

![Anomaly heatmaps](docs/images/heatmaps_transistor.png)

Scores for the images in the figure ([`docs/run_logs/heatmaps.txt`](docs/run_logs/heatmaps.txt)):

| Image | Type | Anomaly score | Detected where? |
|---|---|---:|---|
| good | real, normal | **17.77** | nothing highlighted ✅ |
| bent_lead | real defect | 22.72 | bent lead, bottom-left ✅ |
| cut_lead | real defect | 20.80 | cut lead, bottom ✅ |
| damaged_case | real defect | 21.40 | chipped corner of case ✅ |
| misplaced | real defect | 20.21 | diffuse over whole part (expected — the whole object is out of place) |
| hair | synthetic | 21.27 | along the hair strand ✅ (weakest signal — thin) |
| crack | synthetic | 27.43 | whole crack path ✅ |
| blob (adhesive) | synthetic | 24.74 | blob, bottom-left ✅ |
| speckle (dust) | synthetic | 22.07 | individual particles ✅ |

### 4.2 Synthetic defects generated for the problem statement

Rows: hair, crack, blob (adhesive), speckle (dust). Columns: easy, medium, hard.

![Synthetic defects](docs/images/synthetic_defects_grid.png)

### 4.3 Dataset contact sheet (visual sanity check)

Top: normal transistors. Bottom: real defects with the ground-truth mask overlaid in red.

![Contact sheet](docs/images/contact_sheet_transistor.png)

---

## 5. All run outputs

<details open>
<summary><b>Dataset check</b> — <code>scripts/check_dataset.py</code></summary>

```text
Dataset root: ...\Anomaly detection\data\raw
Detected categories: ['transistor']

Category: transistor
  train_count=213
  test_count=100
  defect_types=['bent_lead', 'cut_lead', 'damaged_case', 'misplaced']
```
Test split: 60 good, 10 each of bent_lead / cut_lead / damaged_case / misplaced.
</details>

<details open>
<summary><b>PatchCore baseline on real defects</b> — <code>scripts/run_patchcore_baseline.py</code> (10 min 24 s on CPU)</summary>

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
</details>

<details open>
<summary><b>Synthetic defect evaluation</b> — <code>scripts/generate_synthetic_defects.py</code> (3 min 11 s on CPU)</summary>

```text
Generated synthetic files: 96
Synthetic defect types: ('blob', 'hair', 'crack', 'speckle')
Image AUROC on clean-vs-synthetic: 1.0000
```
Memory bank: 80 training images, coreset 256. Negatives: 16 good test images.
</details>

<details open>
<summary><b>Synthetic AUROC by defect type and difficulty</b></summary>

| Defect | Easy | Medium | Hard | Images |
|---|---:|---:|---:|---:|
| hair (foreign object) | 1.000 | 1.000 | 1.000 | 24 |
| crack | 1.000 | 1.000 | 1.000 | 24 |
| speckle (dust / sand) | 1.000 | 1.000 | 1.000 | 24 |
| blob (excess adhesive) | 1.000 | 1.000 | 1.000 | 24 |
| **All** | | | | **96 → 1.0000** |

Source: [`docs/run_logs/synthetic_clean.txt`](docs/run_logs/synthetic_clean.txt)
</details>

<details open>
<summary><b>Unit tests</b> — <code>python -m unittest discover -s tests</code></summary>

```text
....
----------------------------------------------------------------------
Ran 4 tests in 0.028s

OK
```
Tests cover: defect mask shape, defect only changes masked pixels, greedy coreset on a hand-made 1-D example, AUROC against a hand-computed value.
</details>

---

## 6. How to run

**Environment used for the results above:** Windows 11, Python 3.12.3, CPU only (no CUDA). Exact package versions are printed by `scripts/print_env.py`.

```powershell
# 1. create environment
python -m venv venv
.\venv\Scripts\python -m pip install --upgrade pip
.\venv\Scripts\python -m pip install -r requirements.txt

# 2. put MVTec AD "transistor" at data\raw\transistor  (or set $env:ANOMALY_DATA_DIR)

# 3. run
.\venv\Scripts\python scripts\check_dataset.py               # dataset counts
.\venv\Scripts\python scripts\contact_sheet.py               # visual QA image
.\venv\Scripts\python scripts\run_patchcore_baseline.py      # real-defect AUROC (~10 min CPU)
.\venv\Scripts\python scripts\generate_synthetic_defects.py  # synthetic AUROC (~3 min CPU)
.\venv\Scripts\python scripts\visualize_heatmaps.py          # heatmap screenshot (~10 min CPU)
.\venv\Scripts\python -m unittest discover -s tests          # tests
```

Expected dataset layout:

```text
data/raw/transistor/
├── train/good/                 213 images
├── test/{good, bent_lead, cut_lead, damaged_case, misplaced}/
└── ground_truth/{bent_lead, cut_lead, damaged_case, misplaced}/
```

The pretrained WideResNet50 weights are downloaded automatically by torchvision on first run. The code refuses to fall back to random weights if the download fails.

---

## 7. Project structure

```text
.
├── src/
│   ├── config.py                 dataset root (data/raw or $ANOMALY_DATA_DIR)
│   ├── dataset_loader.py         MVTec folder discovery, masks, counts
│   ├── patchcore.py              PatchCore: features, coreset, memory bank, scoring, heatmaps
│   └── synthetic_defects.py      hair / crack / speckle / blob generator
├── scripts/
│   ├── check_dataset.py          dataset validation
│   ├── contact_sheet.py          visual QA image
│   ├── run_patchcore_baseline.py real-defect image + pixel AUROC
│   ├── generate_synthetic_defects.py  synthetic defects + AUROC
│   ├── visualize_heatmaps.py     heatmap screenshot
│   └── print_env.py              environment versions
├── tests/test_synthetic_defects.py
├── experiments/synthetic_defects/transistor/{hair,crack,blob,speckle}/   generated images + masks (not committed — recreate with generate_synthetic_defects.py)
├── docs/
│   ├── DOCUMENTATION.md
│   ├── images/                   screenshots used in this README
│   └── run_logs/                 raw output of every run
├── data/raw/transistor/          dataset (not committed)
└── requirements.txt
```

---



## 8. References

1. K. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, P. Gehler. **Towards Total Recall in Industrial Anomaly Detection.** CVPR 2022. [arXiv:2106.08265](https://arxiv.org/abs/2106.08265)
2. P. Bergmann, M. Fauser, D. Sattlegger, C. Steger. **MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.** CVPR 2019. Dataset licence: CC BY-NC-SA 4.0 (see `data/raw/transistor/license.txt`).
3. S. Zagoruyko, N. Komodakis. **Wide Residual Networks.** BMVC 2016 (WideResNet50 backbone, ImageNet weights from torchvision).
