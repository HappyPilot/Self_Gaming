import os
import sys


ROOT = os.path.dirname(os.path.dirname(__file__))
AGENTS_DIR = os.path.join(ROOT, "agents")
if AGENTS_DIR not in sys.path:
    sys.path.insert(0, AGENTS_DIR)
