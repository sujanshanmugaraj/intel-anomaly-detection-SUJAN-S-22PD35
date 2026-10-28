from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_ROOT
from src.patchcore import PatchCoreModel, collect_test_items, set_seed


def main() -> None:
    set_seed(42)
    print("Loading pretrained WideResNet50 backbone...")
    model = PatchCoreModel(resize=256, coreset_size=512)

    train_images = [Image.open(p).convert("RGB") for p in sorted((DATA_ROOT / "transistor" / "train" / "good").glob("*.png"))]
    print(f"Train images: {len(train_images)}")

    sample = train_images[0]
    stage2, stage3, patches = model.extract_patch_vectors(sample)
    print(f"Stage2 shape: {tuple(stage2.shape)}")
    print(f"Stage3 shape: {tuple(stage3.shape)}")
    print(f"Patch vector count: {patches.shape[0]}")

    memory_bank = model.build_memory_bank(train_images)
    print(f"Memory bank shape: {tuple(memory_bank.shape)}")

    test_items, _ = collect_test_items("transistor")
    print(f"Evaluation items: {len(test_items)}")

    image_labels: list[int] = []
    image_scores: list[float] = []
    pixel_labels: list[int] = []
    pixel_scores: list[float] = []

    for image, label, mask_path in test_items:
        patch_scores, image_score = model.score_image(image, memory_bank)
        image_labels.append(label)
        image_scores.append(image_score)

        if mask_path is not None:
            mask = Image.open(mask_path).convert("L").resize((model.resize, model.resize))
            mask_arr = np.array(mask) > 0
            heatmap = model.heatmap_from_scores(patch_scores, 32, 32, (model.resize, model.resize))
            heatmap_arr = heatmap.cpu().numpy()
            pixel_labels.extend(mask_arr.flatten().astype(int).tolist())
            pixel_scores.extend(heatmap_arr.flatten().tolist())
        else:
            heatmap = model.heatmap_from_scores(patch_scores, 32, 32, (model.resize, model.resize))
            heatmap_arr = heatmap.cpu().numpy()
            pixel_labels.extend(np.zeros_like(heatmap_arr, dtype=np.uint8).flatten().astype(int).tolist())
            pixel_scores.extend(heatmap_arr.flatten().tolist())

    image_auc = roc_auc_score(image_labels, image_scores)
    pixel_auc = roc_auc_score(pixel_labels, pixel_scores)

    print(f"Image AUROC: {image_auc:.4f}")
    print(f"Pixel AUROC: {pixel_auc:.4f}")
    print("PatchCore baseline evaluation complete.")


if __name__ == "__main__":
    main()
