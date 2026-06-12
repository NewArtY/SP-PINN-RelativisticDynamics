"""Resolve repository paths and make the ``relsim`` package importable."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DATA, exist_ok=True)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
