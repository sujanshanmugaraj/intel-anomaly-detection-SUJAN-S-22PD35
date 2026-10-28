from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_ROOT, PROJECT_ROOT as ROOT
from src.patchcore import PatchCoreModel, set_seed
from src.synthetic_defects import generate_synthetic_dataset


def main() -> None:
    set_seed(42)
    category = "transistor"
    train_dir = DATA_ROOT / category / "train" / "good"
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    output_dir = ROOT / "experiments" / "synthetic_defects" / category
    output_dir.mkdir(parents=True, exist_ok=True)

    defect_types = ("blob", "scratch", "line", "stain", "speckle")
    created = generate_synthetic_dataset(train_dir, output_dir, defect_types=defect_types, images_per_type=8, seed=42)

    model = PatchCoreModel(resize=256, coreset_size=256)
    train_images = [Image.open(p).convert("RGB") for p in sorted(train_dir.glob("*.png"))[:80]]
    memory_bank = model.build_memory_bank(train_images)

    labels: list[int] = []
    scores: list[float] = []
    for defect_dir in sorted(output_dir.iterdir()):
        if not defect_dir.is_dir():
            continue
        for image_path in sorted(defect_dir.glob("*.png"))[:8]:
            image = Image.open(image_path).convert("RGB")
            _, score = model.score_image(image, memory_bank)
            labels.append(1)
            scores.append(score)

    for image_path in sorted(train_dir.glob("*.png"))[:16]:
        image = Image.open(image_path).convert("RGB")
        _, score = model.score_image(image, memory_bank)
        labels.append(0)
        scores.append(score)

    print(f"Generated synthetic files: {len(created)}")
    print(f"Synthetic defect types: {defect_types}")
    print(f"Image AUROC on clean-vs-synthetic: {roc_auc_score(labels, scores):.4f}")
    print(f"Saved synthetic set to: {output_dir}")


if __name__ == "__main__":
    main()
