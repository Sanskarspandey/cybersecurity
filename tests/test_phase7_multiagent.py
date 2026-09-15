"""
Pytest integration runner for Phase 7 Multi-Agent AI Orchestration Layer.
"""

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from experiments.validate_phase7 import Phase7Validator


def test_phase7_multiagent_validation_suite():
    """Runs all 16 compliance checks for Phase 7."""
    validator = Phase7Validator()
    validator.run_all()
    failed_tests = [r for r in validator.results if not r["passed"]]
    assert len(failed_tests) == 0, f"Failed Phase 7 checks: {failed_tests}"
