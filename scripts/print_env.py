from __future__ import annotations

import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    print("=== Environment ===")
    print(f"OS: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")

    try:
        import torch
        print(f"Torch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
    except Exception as exc:  # pragma: no cover
        print(f"Torch not available: {exc}")

    try:
        import torchvision
        print(f"TorchVision: {torchvision.__version__}")
    except Exception as exc:  # pragma: no cover
        print(f"TorchVision not available: {exc}")

    try:
        import sklearn
        print(f"Scikit-learn: {sklearn.__version__}")
    except Exception as exc:  # pragma: no cover
        print(f"Scikit-learn not available: {exc}")

    try:
        import numpy
        print(f"NumPy: {numpy.__version__}")
    except Exception as exc:  # pragma: no cover
        print(f"NumPy not available: {exc}")

    try:
        import PIL
        print(f"Pillow: {PIL.__version__}")
    except Exception as exc:  # pragma: no cover
        print(f"Pillow not available: {exc}")


if __name__ == "__main__":
    main()
