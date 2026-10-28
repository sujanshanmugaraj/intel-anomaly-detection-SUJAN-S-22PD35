from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


def generate_defect_mask(shape: tuple[int, int], defect_type: str = "blob", rng: np.random.Generator | None = None) -> np.ndarray:
    """Create a normalized defect mask for a synthetic anomaly patch."""
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError(f"shape must be positive, got {shape!r}")

    rng = np.random.default_rng() if rng is None else rng
    mask = np.zeros((height, width), dtype=np.float32)

    defect_type = defect_type.lower()

    if defect_type in {"blob", "stain"}:
        cx = rng.integers(0, width)
        cy = rng.integers(0, height)
        rx = rng.integers(max(12, width // 12), max(24, width // 4))
        ry = rng.integers(max(12, height // 12), max(24, height // 4))
        y, x = np.ogrid[:height, :width]
        dist = ((x - cx) / max(rx, 1)) ** 2 + ((y - cy) / max(ry, 1)) ** 2
        mask = np.exp(-dist * 2.5)
        if defect_type == "stain":
            mask *= rng.uniform(0.7, 1.0)

    elif defect_type == "scratch":
        x0 = rng.integers(0, max(1, width // 3))
        x1 = rng.integers(max(x0 + 10, width // 2), width)
        y0 = rng.integers(0, height)
        y1 = rng.integers(0, height)
        if x0 == x1:
            x1 = min(width - 1, x0 + 2)
        points = np.linspace((x0, y0), (x1, y1), num=max(height, width), endpoint=True)
        for x, y in np.rint(points).astype(int):
            x = np.clip(x, 0, width - 1)
            y = np.clip(y, 0, height - 1)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    xx = x + dx
                    yy = y + dy
                    if 0 <= xx < width and 0 <= yy < height:
                        mask[yy, xx] = max(mask[yy, xx], 1.0)

    elif defect_type == "line":
        x0 = rng.integers(0, width)
        y0 = rng.integers(0, height)
        x1 = rng.integers(0, width)
        y1 = rng.integers(0, height)
        points = np.linspace((x0, y0), (x1, y1), num=max(height, width), endpoint=True)
        for x, y in np.rint(points).astype(int):
            x = np.clip(x, 0, width - 1)
            y = np.clip(y, 0, height - 1)
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    xx = x + dx
                    yy = y + dy
                    if 0 <= xx < width and 0 <= yy < height:
                        mask[yy, xx] = max(mask[yy, xx], 1.0)

    elif defect_type == "speckle":
        num_spots = max(8, width * height // 200)
        for _ in range(num_spots):
            cx = rng.integers(0, width)
            cy = rng.integers(0, height)
            radius = rng.integers(2, max(5, width // 20))
            y, x = np.ogrid[:height, :width]
            spot = ((x - cx) ** 2 + (y - cy) ** 2) <= radius**2
            mask[spot] = np.maximum(mask[spot], rng.uniform(0.4, 1.0))

    else:
        raise ValueError(f"Unsupported defect type: {defect_type!r}")

    return np.clip(mask, 0.0, 1.0)


def apply_defect(image: Image.Image, defect_type: str = "blob", intensity: float = 0.8, rng: np.random.Generator | None = None) -> Image.Image:
    """Apply a synthetic defect to an RGB image and return the modified image."""
    rng = np.random.default_rng() if rng is None else rng
    image_array = np.asarray(image.convert("RGB")).astype(np.float32)
    mask = generate_defect_mask(image_array.shape[:2], defect_type=defect_type, rng=rng)
    strength = np.clip(float(intensity), 0.05, 1.0)

    if defect_type in {"scratch", "line"}:
        image_array = image_array - mask[..., None] * (140.0 * strength)
    else:
        image_array = image_array - mask[..., None] * (110.0 * strength)

    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    return Image.fromarray(image_array, mode="RGB")


def generate_synthetic_dataset(
    source_dir: str | Path,
    save_dir: str | Path,
    defect_types: Sequence[str] = ("blob", "scratch", "line", "stain", "speckle"),
    images_per_type: int = 10,
    seed: int = 42,
) -> list[Path]:
    """Create a synthetic defect dataset by overlaying generated anomalies on clean images."""
    source_dir = Path(source_dir)
    save_dir = Path(save_dir)
    rng = np.random.default_rng(seed)

    created: list[Path] = []
    image_paths = sorted(source_dir.glob("*.png")) + sorted(source_dir.glob("*.jpg")) + sorted(source_dir.glob("*.jpeg"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {source_dir}")

    for defect_type in defect_types:
        defect_dir = save_dir / defect_type
        defect_dir.mkdir(parents=True, exist_ok=True)
        for idx, path in enumerate(image_paths[:images_per_type]):
            image = Image.open(path).convert("RGB")
            synthetic = apply_defect(image, defect_type=defect_type, intensity=0.7 + 0.3 * (idx % 3) / 3, rng=rng)
            output_path = defect_dir / f"{path.stem}_{defect_type}_{idx:02d}{path.suffix}"
            synthetic.save(output_path)
            created.append(output_path)

    return created
