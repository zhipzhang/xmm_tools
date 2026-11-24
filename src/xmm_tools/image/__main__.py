"""Allow running the FITS image viewer via ``python -m xmm_tools.image``."""

import sys

from xmm_tools.image.gui import main

if __name__ == "__main__":
    sys.exit(main())
