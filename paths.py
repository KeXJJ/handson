"""Machine-independent locations shared by download, training and validation."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('HANDSON_WORK_DIR', str(REPO_ROOT/'workdir'))).expanduser().resolve()
DATASET_DIR = Path(os.environ.get('HANDSON_DATASET_DIR', str(ROOT/'dataset'))).expanduser().resolve()
UPSTREAM_ROOT = REPO_ROOT/'third_party/handson/openpi'
DATASET_READER_ROOT = REPO_ROOT/'third_party/dataset_reader'
