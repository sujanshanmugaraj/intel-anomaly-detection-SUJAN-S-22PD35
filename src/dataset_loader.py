from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

from PIL import Image

from src.config import DATA_ROOT


@dataclass
class Sample:
    image_path: Path
    label: str
    defect_type: str | None = None
    mask_path: Path | None = None


def _sorted_image_files(dir_path: Path) -> list[Path]:
    if not dir_path.exists():
        return []
    return sorted(
        [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}],
        key=lambda p: p.name,
    )


def list_category_dirs(root: Path | str | None = None) -> List[Path]:
    root_path = Path(root) if root is not None else DATA_ROOT
    if not root_path.exists():
        return []
    return sorted(path for path in root_path.iterdir() if path.is_dir())


def list_mvttec_training_images(category_dir: Path) -> list[Path]:
    train_dir = category_dir / "train" / "good"
    return _sorted_image_files(train_dir)


def list_mvttec_test_images(category_dir: Path) -> list[Path]:
    test_dir = category_dir / "test"
    if not test_dir.exists():
        return []
    images: list[Path] = []
    for defect_dir in sorted(test_dir.iterdir()):
        if defect_dir.is_dir() and defect_dir.name != "good":
            images.extend(_sorted_image_files(defect_dir))
    return images


def list_mvttec_normal_test_images(category_dir: Path) -> list[Path]:
    normal_dir = category_dir / "test" / "good"
    return _sorted_image_files(normal_dir)


def list_defect_types(category_dir: Path) -> list[str]:
    test_dir = category_dir / "test"
    if not test_dir.exists():
        return []
    return sorted(p.name for p in test_dir.iterdir() if p.is_dir() and p.name != "good")


def _find_mask_for_image(image_path: Path, category_dir: Path) -> Path | None:
    defect_type = image_path.parent.name
    mask_dir = category_dir / "ground_truth" / defect_type
    if not mask_dir.exists():
        return None

    stem = image_path.stem
    candidates = [
        mask_dir / f"{stem}_mask.png",
        mask_dir / f"{stem}_mask.jpg",
        mask_dir / f"{stem}.png",
        mask_dir / f"{stem}.jpg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # fallback for MVTec-style names where the mask lives under a different filename pattern
    for mask_path in _sorted_image_files(mask_dir):
        if mask_path.stem.startswith(stem):
            return mask_path
    return None


def iter_mvttec_samples(category_dir: Path) -> Iterator[Sample]:
    for image_path in list_mvttec_training_images(category_dir):
        yield Sample(image_path=image_path, label="normal")

    good_dir = category_dir / "test" / "good"
    for image_path in _sorted_image_files(good_dir):
        yield Sample(image_path=image_path, label="normal", defect_type="good")

    for defect_type in list_defect_types(category_dir):
        defect_dir = category_dir / "test" / defect_type
        for image_path in _sorted_image_files(defect_dir):
            mask_path = _find_mask_for_image(image_path, category_dir)
            yield Sample(
                image_path=image_path,
                label="defective",
                defect_type=defect_type,
                mask_path=mask_path,
            )


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def describe_dataset(category_name: str, root: Path | str | None = None) -> dict:
    root_path = Path(root) if root is not None else DATA_ROOT
    category_dir = root_path / category_name
    train_images = list_mvttec_training_images(category_dir)
    normal_test_images = list_mvttec_normal_test_images(category_dir)
    defect_types = list_defect_types(category_dir)

    defect_counts = {}
    for defect_type in defect_types:
        defect_dir = category_dir / "test" / defect_type
        defect_counts[defect_type] = len(_sorted_image_files(defect_dir))

    total_test_count = len(normal_test_images) + sum(defect_counts.values())
    return {
        "category": category_name,
        "category_dir": str(category_dir),
        "train_count": len(train_images),
        "normal_test_count": len(normal_test_images),
        "defective_test_count": sum(defect_counts.values()),
        "test_count": total_test_count,
        "defect_types": defect_types,
        "defect_counts": defect_counts,
    }
