"""This file is for project path."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = Path(os.environ.get("DATA_DIR", PROJECT_ROOT))
