"""
Pytest integration runner for Phase 8A MongoDB Quarantine Store.
"""

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from experiments.validate_phase8a import Phase8AValidator


def test_phase8a_quarantine_validation_suite():
    """Runs all 18 compliance and regression checks for Phase 8A."""
    validator = Phase8AValidator()
    validator.run_all()
    failed_tests = [r for r in validator.results if not r["passed"]]
    assert len(failed_tests) == 0, f"Failed Phase 8A checks: {failed_tests}"
