Interactive GUI Tools (``xmm-qdp``, ``xmm-img``, & ``xmm-srclist``)
===================================================================

``xmm_tools`` provides three Dear ImGui / ImPlot desktop viewers (installed via
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

``xmm-srclist`` — Multi-Epoch Source-List Merger & Mask Prep
------------------------------------------------------------

Load multiple epoch ``cheese/`` directories or ``emllist.fits`` source tables,
cross-match detections across epochs, inspect overlays on a background FITS
image, filter or manually toggle individual sources, and export a unified
``merged_emllist.fits`` (and optional DS9 ``.reg`` overlay):

.. code-block:: bash

   # Launch the interactive GUI with multiple epoch cheese directories
   xmm-srclist obs1/cheese obs2/cheese --bg esa_work/qpb_image/soft_500_2000_flux.fits

   # Or merge and export headlessly from the CLI
   xmm-srclist obs1/cheese obs2/cheese \
     --match-radius 5.0 --min-ml 15.0 --min-flux 2e-15 \
     --export-fits merged_emllist.fits --export-reg merged_sources.reg

Feeding ``merged_emllist.fits`` into ``xmm-ana mask`` & ``spectrum``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once ``merged_emllist.fits`` is exported, generate per-instrument detector and
sky exclusion masks for each observation via ``xmm-ana mask``, then point
``xmm-ana spectrum`` (or ``xmm-ana run``) at that custom mask directory with
``--mask-dir`` (or ``[esas.spectrum].mask_dir`` in ``ana.toml``):

.. code-block:: bash

   # 1. Generate custom masks in a dedicated directory (leaves cheese/ untouched)
   xmm-ana mask --config ana.toml \
     --srclist merged_emllist.fits \
     --out-dir mask_merged

   # 2. Extract spectra using the merged-catalog masks
   xmm-ana spectrum --config ana.toml --mask-dir mask_merged
   # Or run flare + spectrum together:
   xmm-ana run --config ana.toml --stages flare,spectrum --mask-dir mask_merged
