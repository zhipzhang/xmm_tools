Interactive GUI Tools (``xmm-qdp`` & ``xmm-img``)
=================================================

``xmm_tools`` provides two Dear ImGui / ImPlot desktop viewers (installed via
``pip install -e ".[gui]"``).

``xmm-qdp`` — XMM-Newton QDP Viewer
-----------------------------------

View ``espfilt`` flare histograms (``*-hist.qdp``) and ``mosback``/``pnback``
background diagnostics (``*-bkgspec.qdp``) with full support for XMM PLT directives
(``viewport``, ``win``, ``yplot``, ``col``, ``la ...``).

.. code-block:: bash

   xmm-qdp path/to/mos1S001-bkgspec.qdp

``xmm-img`` — FITS Flux Image & DS9 Region Tool
-----------------------------------------------

Inspect FITS flux images (such as ``quick_image`` outputs) and interactively build
DS9 ``fk5`` sky regions for ``xmm-esas spectrum``:

.. code-block:: bash

   xmm-img path/to/soft_flux.fits

- **Stretch & Colormap**: linear / log / sqrt / asinh stretch, Gaussian smoothing, interactive colorbar.
- **Strip Grid Generator**: draggable origin & angle, continuous equal-width boxes with ``text={label}``
  prefixes (ready for ``xmm-esas spectrum`` ``mode = "split"``), single-edge height stretching.
- **Manual Shapes & Live Editor**: add Circle, Ellipse, Box, or Polygon; drag center handles on the canvas;
  edit parameters live; load/save standard DS9 ``.reg`` files via ``astropy-regions``.
