"""Allow running the source-list merger via ``python -m xmm_tools.srclist``."""

import sys

from xmm_tools.srclist.gui import main

if __name__ == "__main__":
    sys.exit(main())
