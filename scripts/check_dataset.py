from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_ROOT
from src.dataset_loader import describe_dataset, list_category_dirs


def main() -> None:
    print(f"Dataset root: {DATA_ROOT}")
    dirs = list_category_dirs(DATA_ROOT)

    if not dirs:
        print("No dataset found at the current path.")
        print("Attach your dataset under: data/raw/")
        print("or set ANOMALY_DATA_DIR to the absolute path of your dataset root.")
        return

    print(f"Detected categories: {[d.name for d in dirs]}")
    for category_dir in dirs:
        info = describe_dataset(category_dir.name, DATA_ROOT)
        print(f"\nCategory: {info['category']}")
        print(f"  train_count={info['train_count']}")
        print(f"  test_count={info['test_count']}")
        print(f"  defect_types={info['defect_types']}")


if __name__ == "__main__":
    main()
