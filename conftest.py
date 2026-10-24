"""
Pytest configuration for IoMT Statistics Dashboard tests.
Ensures repository root is in python search path.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
