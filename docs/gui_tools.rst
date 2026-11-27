Interactive GUI Tools (``xmm-qdp`` & ``xmm-img``)
=================================================

``xmm_tools`` provides two Dear ImGui / ImPlot desktop viewers (installed via
``pip install -e ".[gui]"``).

``xmm-qdp`` — XMM-Newton QDP Viewer
-----------------------------------

View ``espfilt`` flare histograms (``*-hist.qdp``) and ``mosback``/``pnback``
background diagnostics (``*-bkgspec.qdp``, ``*-augspec.qdp``, ``*-ratehard.qdp``)
with full support for XMM PLT directives (``viewport``, ``win``, ``yplot``,
``col``, ``la ...``).

.. code-block:: bash

   # Inspect flare filtering histogram
   xmm-qdp esas_work/soft_proton_flare/mos1_evt-hist.qdp

   # Inspect particle-background model vs. corner spectrum
   xmm-qdp esas_work/spectrum/background/mos1S001-bkgspec.qdp

``xmm-img`` — FITS Flux Image & DS9 Region Tool
-----------------------------------------------

Inspect FITS flux images (such as ``qpb_image`` outputs) and interactively
build DS9 ``fk5`` sky regions for ``xmm-ana spectrum``:

.. code-block:: bash

   xmm-img esas_work/qpb_image/soft_500_2000_flux.fits

Key capabilities:

- **Stretch & Colormap**: linear / log / sqrt / asinh stretch, Gaussian
  smoothing, interactive colorbar and contrast controls.
- **Strip Grid Generator**: draggable origin & position angle, continuous
  equal-width boxes with ``text={label}`` prefixes (directly compatible with
  ``xmm-ana spectrum`` ``mode = "split"``), and single-edge height stretching.
- **Manual Shapes & Live Editor**: add Circle, Ellipse, Box, or Polygon; drag
  center handles on the canvas; edit parameters live; load/save standard DS9
  ``.reg`` files via ``astropy-regions``.

Wiring Saved Regions into ``ana.toml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After saving a DS9 ``.reg`` file from ``xmm-img`` (for example,
``regions/east_jet_bins.reg``), add a matching ``[[esas.spectrum.regions]]`` block
to ``ana.toml``:

.. code-block:: toml

   [[esas.spectrum.regions]]
   name = "east_jet"
   file = "regions/east_jet_bins.reg"
   mode = "split"       # Each labeled box becomes spectrum/east_jet/<label>/
   overwrite = false

Then run ``xmm-ana run --config ana.toml`` (or ``xmm-ana spectrum``) to
extract spectra and particle backgrounds for every bin automatically.
