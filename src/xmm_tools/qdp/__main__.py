"""Allow running the QDP GUI via ``python -m xmm_tools.qdp``."""

import sys

from xmm_tools.qdp.gui import main

if __name__ == "__main__":
    sys.exit(main())
