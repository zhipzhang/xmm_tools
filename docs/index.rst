xmm_tools Documentation
=======================

**xmm_tools** is a modular Python toolkit for XMM-Newton X-ray extended-source
analysis, bridging SAS/ESAS data reduction, interactive DS9 region & QDP
inspection, and transparent OOP Sherpa joint spectral fitting.

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

Quick Start
-----------

.. code-block:: bash

   # 1. One-time ODF initialization (generates esas.toml + standardized *_evt.fits)
   xmm-esas init --odf-dir /path/to/ODF --workdir ./esas_work --ccfpath /ccf/valid
   cd esas_work

   # 2. Run flare filtering, point-source masking, and quick-look flux imaging automatically
   xmm-esas run --config esas.toml

   # 3. Inspect flare histograms & draw DS9 regions on the flux map
   xmm-qdp soft_proton_flare/mos1_evt-hist.qdp
   xmm-img quick_image/soft_500_2000_flux.fits

   # 4. Add [[spectrum.regions]] to esas.toml, then extract spectra (cached stages are skipped)
   xmm-esas run --config esas.toml

   # Or run any specific subset of stages directly:
   xmm-esas run --config esas.toml --stages cheese,spectrum --only background --force

End-to-End Workflow
-------------------

1. **ESAS Reduction** (:doc:`esas_config`): initialize ODF/CIF/events with
   ``xmm-esas init``, then run all or selected config-driven stages via
   ``xmm-esas run`` (or individual subcommands ``flare``, ``cheese``,
   ``quick-image``, ``spectrum``).
2. **Interactive Inspection & Regions** (:doc:`gui_tools`): inspect QDP
   diagnostics with ``xmm-qdp``, view QPB-subtracted FITS flux maps with
   ``xmm-img``, and export DS9 ``.reg`` files ready for ``xmm-esas spectrum``.
3. **Joint Spectral Fitting** (:doc:`sherpa_joint_fit`): load
   ``spectrum/manifest.json`` into ``xmm_tools.sherpa``, assemble 3-channel
   responses (sky ``rmf+arf``, soft-proton diagonal RMF without ARF, and
   instrumental lines ``rmf + unit_arf``), fit jointly, export JSON, and plot
   residuals.

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
