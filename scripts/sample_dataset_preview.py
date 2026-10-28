from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from src.config import DATA_ROOT


def main() -> None:
    root = DATA_ROOT
    print(f"Previewing dataset from: {root}")
    if not root.exists():
        print("Dataset root does not exist yet.")
        return

    for category in sorted(p for p in root.iterdir() if p.is_dir()):
        train_dir = category / "train" / "good"
        if train_dir.exists():
            samples = sorted(train_dir.glob("*.png"))[:3]
            print(f"\n{category.name} training samples: {len(samples)}")
            for sample in samples:
                img = Image.open(sample)
                print(f"  {sample.name} -> {img.size}")


if __name__ == "__main__":
    main()
