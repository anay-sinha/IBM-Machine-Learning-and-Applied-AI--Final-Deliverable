"""
conftest.py
============
Shared pytest fixtures for the fraud detection test suite.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so imports work from any CWD
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
