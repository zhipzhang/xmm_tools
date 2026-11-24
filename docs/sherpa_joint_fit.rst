Sherpa Joint Spectral Fitting (``xmm_tools.sherpa``)
======================================================

``xmm_tools.sherpa`` provides a transparent, collision-free OOP wrapper on top of
Sherpa's native objects (``DataPHA``, ``DataSimulFit``, ``SimulFitModel``, ``Fit``),
without relying on global ``sherpa.astro.ui`` IDs.

Design Principles
-----------------

1. **Three-Channel Response Folding**:

   - **Sky Background + Source Emission**: folded with native ``rmf + arf``.
   - **Soft Proton**: folded with **diagonal RMF without ARF** (built-in ``mos1-diag.rsp.gz``,
     ``mos2-diag.rsp.gz``, ``pn-diag.rsp.gz`` are bundled in the package and loaded automatically).
   - **Instrument Fluorescence Lines**: folded with ``rmf + unit_arf`` (or ``sky_arf``).

2. **Physical Scaling**:

   - Celestial sky components scale by ``BACKSCAL`` solid angle :math:`\Omega` (``per_arcmin2=True``).
   - Soft-proton components scale by ``protonscale`` ``Scaled Ave Flux`` (``use_proton_scale=True``
     or ``ds.proton_ratio(ref_ds)``), never by ``BACKSCAL``.

3. **Full Transparency**:

   - You define the exact Sherpa or XSPEC models (e.g., ``powerlaw`` vs ``bknpowerlaw`` for soft proton,
     ``apec``/``vnei``/``vapec`` for source) via ``ModelBuilder`` (``lambda m: ...`` or ``lambda m, ds: ...``).
   - Every sub-model is uniquely named and exposed for parameter inspection and linking.

Complete Joint-Fit Example
--------------------------

.. code-block:: python

   from xmm_tools.sherpa import JointFitSession, SpectrumCollection

   # 1. Load selected regions from spectrum/manifest.json
   #    - Skipped (0-event) instruments are omitted automatically
   #    - PN automatically loads *-fovtootsub.pi
   #    - Built-in diagonal RMFs and proton_scale values are loaded automatically
   col = SpectrumCollection.from_manifest(
       "spectrum/manifest.json",
       regions=["background", "east_jet/e4", "east_jet/e5"],
   )
   col.notice(0.5, 7.0).group_counts(30).subtract()

   session = JointFitSession(col)

   # 2. Cross-calibration: MOS1/MOS2 free, PN frozen to 1.0, shared per instrument
   session.set_calibration(
       lambda m, ds: m.constant(
           "cal",
           factor=1.0,
           bounds={"factor": (0.1, 10.0)},
           frozen=(ds.instrument == "pn"),
       ),
       scope="per_instrument",
   )

   # 3. Sky background: global across all regions, scaled by BACKSCAL solid angle
   session.set_sky_background(
       lambda m: m.tbabs("gal", nH=0.56, frozen=True)
       * (m.apec("gh", kT=0.2) + m.powerlaw("cxb", PhoIndex=1.4, frozen=True)),
       scope="global",
       per_arcmin2=True,
   )

   # 4. Source emission: only on source regions (not background), shared across MOS1/MOS2/PN per region
   session.set_source(
       lambda m: m.tbabs("nh", nH=0.72) * m.apec("jet", kT=0.3),
       scope="region",
       regions=["east_jet/e4", "east_jet/e5"],
   )

   # 5. Soft proton: diagonal RMF without ARF; scaled by ESAS protonscale (Scaled Ave Flux)
   session.set_soft_proton(
       lambda m: m.bknpowerlaw(
           "sp",
           PhoIndx1=0.6,
           BreakE=3.0,
           PhoIndx2=3.2,
           bounds={"BreakE": (1.5, 5.0), "PhoIndx2": (0.0, 6.0)},
       ),
       scope="per_instrument",
       use_proton_scale=True,
   )

   # 6. Instrument fluorescence lines: independent per dataset, unit ARF
   session.set_instrument_lines(
       lambda m, ds: (
           m.gaussian("al", LineE=1.486, Sigma=0.0, frozen=["LineE", "Sigma"])
           + m.gaussian("si", LineE=1.739, Sigma=0.0, frozen=["LineE", "Sigma"])
           if ds.instrument in ("mos1", "mos2")
           else m.gaussian("al", LineE=1.486, Sigma=0.0, frozen=["LineE", "Sigma"])
       ),
       scope="dataset",
       response="unit_arf",
   )

   session.build()

   # Optional: link specific parameters across regions
   session.link("east_jet/e5.source_emission.nh.nH", "east_jet/e4.source_emission.nh.nH")

   # 7. Fit, inspect summary, export JSON, and plot
   report = session.fit(stat="chi2gehrels", method="levmar", estimate_errors=True)
   report.print_summary(include_linked=True)
   report.save_json("fit_results/joint_fit.json")

   # Reload saved JSON and access fitted component parameters explicitly
   # prev = FitReport.from_json("fit_results/joint_fit.json")
   # sky = prev.component("sky_background")   # -> {"gal": {"nH": ...}, "gh": {"kT": ...}, ...}
   # kT_gh = prev.get("sky_background", "gh", "kT")

   session.plot("east_jet/e4", instrument="mos1", save_path="fit_results/e4_mos1.pdf")
   session.plot("east_jet/e4", instrument=["mos1", "mos2", "pn"], save_path="fit_results/e4_all.png")

Scopes Reference
----------------

- ``scope="global"``: 1 shared model instance across all regions and instruments.
- ``scope="region"``: 1 model instance per region, shared across ``mos1``/``mos2``/``pn`` inside that region.
- ``scope="per_instrument"``: 1 model instance per instrument, shared across regions.
- ``scope="dataset"``: 1 independent model instance per ``(region, instrument)`` dataset.
