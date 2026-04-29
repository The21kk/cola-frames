#!/usr/bin/env python
"""Run phase 1 tests and show output."""

import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_phase1_infrastructure.py", "-v", "--tb=short"],
    cwd="c:\\Users\\Operador\\Desktop\\Nico\\cola-frames",
    capture_output=False,
    text=True
)

sys.exit(result.returncode)
