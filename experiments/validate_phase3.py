"""Phase 3 validation alias pointing to preprocessing/validate_phase3.py."""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from preprocessing.validate_phase3 import run_hardened_validation

if __name__ == "__main__":
    success = run_hardened_validation()
    sys.exit(0 if success else 1)
