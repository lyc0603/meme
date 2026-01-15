"""This file is for project path."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Directory Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = Path(os.environ.get("DATA_DIR", PROJECT_ROOT))

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
