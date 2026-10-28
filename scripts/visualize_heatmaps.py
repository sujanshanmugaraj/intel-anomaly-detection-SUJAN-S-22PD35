from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_ROOT
from src.patchcore import PatchCoreModel, _find_mask_path, set_seed


RESULTS_DIR = PROJECT_ROOT / "results"
SYNTHETIC_DIR = PROJECT_ROOT / "experiments" / "synthetic_defects" / "transistor"


def pick_examples(category_dir: Path) -> list[tuple[str, Path, Path | None]]:
    examples: list[tuple[str, Path, Path | None]] = []
    good = sorted((category_dir / "test" / "good").glob("*.png"))
    if good:
        examples.append(("good (real)", good[0], None))
    for defect_dir in sorted((category_dir / "test").iterdir()):
        if defect_dir.name == "good" or not defect_dir.is_dir():
            continue
        images = sorted(defect_dir.glob("*.png"))
        if images:
            examples.append((f"{defect_dir.name} (real)", images[0], _find_mask_path(images[0], category_dir)))
    for synthetic_type in ("hair", "crack", "blob", "speckle"):
        images = [p for p in sorted((SYNTHETIC_DIR / synthetic_type).glob("*_medium_*.png")) if not p.stem.endswith("_mask")]
        if images:
            examples.append((f"{synthetic_type} (synthetic)", images[0], None))
    return examples


def main() -> None:
    set_seed(42)
    category_dir = DATA_ROOT / "transistor"
    model = PatchCoreModel(resize=256, coreset_size=512)
    train_images = [Image.open(p).convert("RGB") for p in sorted((category_dir / "train" / "good").glob("*.png"))]
    memory_bank = model.build_memory_bank(train_images)

    examples = pick_examples(category_dir)
    scored = []
    for title, image_path, mask_path in examples:
        image = Image.open(image_path).convert("RGB")
        patch_scores, image_score = model.score_image(image, memory_bank)
        heatmap = model.heatmap_from_scores(patch_scores, 32, 32, (model.resize, model.resize)).cpu().numpy()
        scored.append((title, image, mask_path, image_score, heatmap))

    # One shared colour scale for every row, so a clean image does not look "hot" just because
    # each map was stretched to its own min/max.
    vmin = float(np.percentile(np.concatenate([h.ravel() for *_, h in scored]), 1))
    vmax = float(max(h.max() for *_, h in scored))

    fig, axes = plt.subplots(len(scored), 3, figsize=(9, 3 * len(scored)))
    for row, (title, image, mask_path, image_score, heatmap) in enumerate(scored):
        shown = np.array(image.resize((model.resize, model.resize)))

        axes[row, 0].imshow(shown)
        axes[row, 0].set_title(f"{title}\nscore={image_score:.2f}", fontsize=9)
        axes[row, 1].imshow(shown)
        axes[row, 1].imshow(heatmap, cmap="jet", alpha=0.5, vmin=vmin, vmax=vmax)
        axes[row, 1].set_title("anomaly heatmap", fontsize=9)
        if mask_path is not None:
            mask = np.array(Image.open(mask_path).convert("L").resize((model.resize, model.resize)))
            axes[row, 2].imshow(mask, cmap="gray")
            axes[row, 2].set_title("ground-truth mask", fontsize=9)
        else:
            axes[row, 2].imshow(heatmap, cmap="jet", vmin=vmin, vmax=vmax)
            axes[row, 2].set_title("raw score map", fontsize=9)
        for ax in axes[row]:
            ax.axis("off")
        print(f"{title:28s} score={image_score:.3f}")

    fig.tight_layout()
    out_path = RESULTS_DIR / "heatmaps_transistor.png"
    fig.savefig(out_path, dpi=110)
    print(f"Saved heatmap figure to: {out_path}")


if __name__ == "__main__":
    main()
