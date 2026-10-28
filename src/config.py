from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw"
DATA_ROOT = Path(os.environ.get("ANOMALY_DATA_DIR", DEFAULT_DATA_ROOT)).expanduser().resolve()


def ensure_dataset_root() -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return DATA_ROOT
