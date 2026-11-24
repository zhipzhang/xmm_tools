xmm_tools Documentation
=======================

**xmm_tools** is a modular Python toolkit for XMM-Newton X-ray extended-source analysis,
bridging SAS/ESAS data reduction, interactive DS9 region & QDP inspection, and
transparent OOP Sherpa joint spectral fitting.

Installation
------------

Core ESAS & Sherpa wrappers (also works inside headless ``xmmsas`` Docker containers):

.. code-block:: bash

   pip install -e .

With interactive GUI viewers (``xmm-qdp`` and ``xmm-img``):

.. code-block:: bash

   pip install -e ".[gui]"

With documentation build dependencies:

.. code-block:: bash

   pip install -e ".[docs]"

End-to-End Workflow
-------------------

1. **ESAS Reduction** (``xmm-esas``): initialize ODF/CIF/events, filter soft-proton flares,
   create swiss-cheese point-source masks, generate quick-look flux images, and extract
   per-region spectra with ``protonscale`` metadata.
2. **Interactive Inspection & Regions** (``xmm-img``, ``xmm-qdp``): inspect FITS flux images,
   generate continuous strip grids or custom DS9 regions, and inspect ESAS ``.qdp`` diagnostics.
3. **Joint Spectral Fitting** (``xmm_tools.sherpa``): load ``spectrum/manifest.json``,
   assemble 3-channel responses (sky ``rmf+arf``, soft-proton ``diag_rmf`` without ARF,
   and instrumental lines ``rmf + unit_arf``), fit jointly, export JSON, and plot residuals.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   esas_config
   sherpa_joint_fit
   gui_tools

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
