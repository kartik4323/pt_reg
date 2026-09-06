"""Seed and invoke the official CLI without modifying its source or defaults."""
import os
from pathlib import Path
import runpy
import sys

import pytorch_lightning as pl


if __name__ == '__main__':
    pl.seed_everything(int(os.environ.get('SOTA_SEED', '42')), workers=True)
    entry = Path.cwd() / 'puzzle_diff' / 'train_3d.py'
    sys.path.insert(0, str(entry.parent))
    sys.argv[0] = str(entry)
    runpy.run_path(str(entry), run_name='__main__')
