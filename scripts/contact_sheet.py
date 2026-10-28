from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_ROOT
from src.dataset_loader import describe_dataset, list_defect_types, list_mvttec_training_images, _find_mask_for_image


RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def overlay_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    image_arr = np.array(image).astype(np.float32)
    mask_arr = np.array(mask).astype(np.float32) / 255.0
    overlay = image_arr.copy()
    red = np.array([255, 0, 0], dtype=np.float32)
    overlay[mask_arr > 0.5] = (0.7 * overlay[mask_arr > 0.5] + 0.3 * red).astype(np.uint8)
    return Image.fromarray(overlay.astype(np.uint8))


def build_panel(image_path: Path, mask_path: Path | None = None, title: str = "") -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if mask_path is not None:
        mask = Image.open(mask_path).convert("L")
        image = overlay_mask(image, mask)

    canvas = Image.new("RGB", (256, 256), color=(30, 30, 30))
    img_resized = image.resize((220, 220))
    canvas.paste(img_resized, (18, 18))

    if title:
        draw = ImageDraw.Draw(canvas)
        draw.text((18, 5), title, fill=(255, 255, 255))
    return canvas


def main() -> None:
    category = "transistor"
    category_dir = DATA_ROOT / category
    if not category_dir.exists():
        raise FileNotFoundError(f"Category not found: {category_dir}")

    summary = describe_dataset(category, DATA_ROOT)
    print(summary)

    normal_paths = list_mvttec_training_images(category_dir)[:5]
    defect_types = list_defect_types(category_dir)
    defect_samples = []

    for defect_type in defect_types:
        defect_dir = category_dir / "test" / defect_type
        for image_path in sorted(defect_dir.glob("*.png"))[:1]:
            defect_samples.append((image_path, _find_mask_for_image(image_path, category_dir), defect_type))
            if len(defect_samples) >= 5:
                break
        if len(defect_samples) >= 5:
            break

    total_panels = []
    for idx, image_path in enumerate(normal_paths, start=1):
        total_panels.append(build_panel(image_path, title=f"Normal {idx}"))

    for idx, (image_path, mask_path, defect_type) in enumerate(defect_samples, start=1):
        total_panels.append(build_panel(image_path, mask_path, f"{defect_type} {idx}"))

    panel_w, panel_h = total_panels[0].size
    cols = 5
    rows = 2
    sheet = Image.new("RGB", (cols * panel_w, rows * panel_h), color=(20, 20, 20))

    for idx, panel in enumerate(total_panels):
        row = idx // cols
        col = idx % cols
        x = col * panel_w
        y = row * panel_h
        sheet.paste(panel, (x, y))

    out_path = RESULTS_DIR / "contact_sheet_transistor.png"
    sheet.save(out_path)
    print(f"Saved contact sheet to {out_path}")


if __name__ == "__main__":
    main()
