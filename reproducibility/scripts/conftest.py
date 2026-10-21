"""Repository-root pytest configuration.

Two jobs, both additive:

1. Put ``src/`` and ``exploration/dirac/`` on ``sys.path`` so that test
   modules (and an interactive session started from the repository root)
   can ``import dirac``, ``import basis``, ``import feast`` and the
   Paper A helper modules exactly as the drivers do via their own
   ``sys.path.insert``.  Nothing is installed and no module is shadowed.

2. Exclude the three *script-style* files whose names begin with
   ``test_`` but which are pre-registered numerical demonstrations, not
   pytest modules: they contain no ``def test_*`` functions and they run
   their full computation at import time (minutes of Gauss-Legendre work),
   which would otherwise execute during collection.  They remain runnable
   with ``python exploration/dirac/test_gamma_basis.py`` and are left
   byte-for-byte unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

for _sub in ("src", "exploration/dirac"):
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

collect_ignore = [
    "exploration/dirac/test_gamma_basis.py",
    "exploration/dirac/test_cusp_quadrature.py",
    "exploration/dirac/test_shifted_pencil.py",
]
