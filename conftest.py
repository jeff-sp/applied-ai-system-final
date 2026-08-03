"""
Makes the repo root importable so `from src.recommender import ...` works
whether tests are run as `pytest` or `python3 -m pytest`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
