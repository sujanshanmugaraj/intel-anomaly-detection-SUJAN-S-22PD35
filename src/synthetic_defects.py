from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


def _resolve_defect_type(defect_type: str) -> str:
    normalized = defect_type.lower().strip()
    aliases = {
        "blob": "blob",
        "stain": "blob",
        "spot": "blob",
        "patch": "blob",
        "hair": "hair",
        "line": "hair",
        "scratch": "crack",
        "crack": "crack",
        "speckle": "speckle",
        "dust": "speckle",
        "particle": "speckle",
    }
    return aliases.get(normalized, normalized)


def _difficulty_scale(difficulty: str, width: int, height: int) -> tuple[float, float]:
    normalized = difficulty.lower().strip() if difficulty else "medium"
    factor = {"easy": 0.75, "medium": 1.0, "hard": 1.35}.get(normalized, 1.0)
    base_length = max(1.0, min(width, height) * 0.15 * factor)
    return float(factor), float(base_length)


def generate_defect_mask(
    shape: tuple[int, int],
    defect_type: str = "blob",
    rng: np.random.Generator | None = None,
    difficulty: str = "medium",
) -> np.ndarray:
    """Create a normalized defect mask for a synthetic anomaly patch."""
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError(f"shape must be positive, got {shape!r}")

    rng = np.random.default_rng() if rng is None else rng
    mask = np.zeros((height, width), dtype=np.float32)
    defect_type = _resolve_defect_type(defect_type)
    difficulty_scale, base_length = _difficulty_scale(difficulty, width, height)

    if defect_type in {"blob", "stain"}:
        cx = rng.integers(0, width)
        cy = rng.integers(0, height)
        rx = rng.uniform(width * 0.04, width * 0.18) * difficulty_scale
        ry = rng.uniform(height * 0.04, height * 0.18) * difficulty_scale
        y, x = np.ogrid[:height, :width]
        dist = ((x - cx) / max(rx, 1.0)) ** 2 + ((y - cy) / max(ry, 1.0)) ** 2
        mask = np.exp(-dist * 1.8)
        mask *= rng.uniform(0.7, 1.0)

    elif defect_type in {"hair", "line"}:
        length = int(rng.uniform(0.2 * width, 0.7 * width) * (0.8 + difficulty_scale * 0.5))
        length = max(80, min(length, max(width, height) * 2))
        x0 = rng.integers(0, width)
        y0 = rng.integers(0, height)
        angle = rng.uniform(0, 2 * np.pi)
        x1 = int(np.clip(x0 + np.cos(angle) * length, 0, width - 1))
        y1 = int(np.clip(y0 + np.sin(angle) * length, 0, height - 1))
        thickness = rng.integers(1, 3)
        points = np.linspace((x0, y0), (x1, y1), num=max(length, 2), endpoint=True)
        for x, y in np.rint(points).astype(int):
            x = int(np.clip(x, 0, width - 1))
            y = int(np.clip(y, 0, height - 1))
            for dx in range(-thickness, thickness + 1):
                for dy in range(-thickness, thickness + 1):
                    xx = x + dx
                    yy = y + dy
                    if 0 <= xx < width and 0 <= yy < height:
                        mask[yy, xx] = max(mask[yy, xx], 1.0)

    elif defect_type == "crack":
        x0 = rng.integers(0, width)
        y0 = rng.integers(0, height)
        thickness = rng.integers(1, 3)
        for _ in range(rng.integers(3, 7)):
            x1 = rng.integers(0, width)
            y1 = rng.integers(0, height)
            points = np.linspace((x0, y0), (x1, y1), num=max(12, max(width, height) // 8), endpoint=True)
            for x, y in np.rint(points).astype(int):
                x = int(np.clip(x, 0, width - 1))
                y = int(np.clip(y, 0, height - 1))
                for dx in range(-thickness, thickness + 1):
                    for dy in range(-thickness, thickness + 1):
                        xx = x + dx
                        yy = y + dy
                        if 0 <= xx < width and 0 <= yy < height:
                            mask[yy, xx] = max(mask[yy, xx], 1.0)
            x0, y0 = x1, y1

    elif defect_type == "speckle":
        num_spots = rng.integers(5, 30)
        for _ in range(num_spots):
            cx = rng.integers(0, width)
            cy = rng.integers(0, height)
            radius = rng.integers(2, max(4, int(base_length * 0.25)))
            y, x = np.ogrid[:height, :width]
            spot = ((x - cx) ** 2 + (y - cy) ** 2) <= radius**2
            mask[spot] = np.maximum(mask[spot], rng.uniform(0.5, 1.0))

    else:
        raise ValueError(f"Unsupported defect type: {defect_type!r}")

    return np.clip(mask, 0.0, 1.0)


def apply_defect(
    image: Image.Image,
    defect_type: str = "blob",
    intensity: float = 0.8,
    rng: np.random.Generator | None = None,
    difficulty: str = "medium",
    variant: str = "dark",
) -> tuple[Image.Image, np.ndarray]:
    """Apply a synthetic defect to an RGB image and return the modified image and mask."""
    rng = np.random.default_rng() if rng is None else rng
    image_array = np.asarray(image.convert("RGB")).astype(np.float32)
    mask = generate_defect_mask(image_array.shape[:2], defect_type=defect_type, rng=rng, difficulty=difficulty)
    strength = np.clip(float(intensity), 0.05, 1.0)
    defect_strength = 50.0 + 140.0 * strength

    if variant.lower() in {"bright", "light", "positive"}:
        image_array = image_array + mask[..., None] * defect_strength
    else:
        image_array = image_array - mask[..., None] * defect_strength

    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    return Image.fromarray(image_array, mode="RGB"), mask


def generate_synthetic_dataset(
    source_dir: str | Path,
    save_dir: str | Path,
    defect_types: Sequence[str] = ("blob", "hair", "crack", "speckle"),
    images_per_type: int = 10,
    seed: int = 42,
    difficulty_levels: Sequence[str] = ("easy", "medium", "hard"),
) -> list[Path]:
    """Create a synthetic defect dataset by overlaying generated anomalies on clean images."""
    source_dir = Path(source_dir)
    save_dir = Path(save_dir)
    rng = np.random.default_rng(seed)

    image_paths = sorted(source_dir.glob("*.png")) + sorted(source_dir.glob("*.jpg")) + sorted(source_dir.glob("*.jpeg"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {source_dir}")

    created: list[Path] = []
    for defect_type in defect_types:
        defect_dir = save_dir / defect_type
        defect_dir.mkdir(parents=True, exist_ok=True)
        for difficulty in difficulty_levels:
            for idx, path in enumerate(image_paths[:images_per_type]):
                image = Image.open(path).convert("RGB")
                variant = "bright" if rng.random() < 0.5 else "dark"
                intensity = 0.3 + 0.5 * ((idx % 3) / 2.0) + (0.2 if difficulty == "hard" else 0.1)
                synthetic, mask = apply_defect(
                    image,
                    defect_type=defect_type,
                    intensity=intensity,
                    rng=rng,
                    difficulty=difficulty,
                    variant=variant,
                )
                output_stem = f"{path.stem}_{defect_type}_{difficulty}_{idx:02d}"
                image_path = defect_dir / f"{output_stem}.png"
                mask_path = defect_dir / f"{output_stem}_mask.png"
                synthetic.save(image_path)
                Image.fromarray((mask * 255.0).astype(np.uint8), mode="L").save(mask_path)
                created.append(image_path)

    return created
