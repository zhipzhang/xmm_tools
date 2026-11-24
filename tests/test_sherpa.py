"""Tests for xmm_tools.sherpa OOP wrappers (runs against real Sherpa when available)."""

from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from astropy.io import fits

try:
    import sherpa  # noqa: F401

    _HAS_SHERPA = True
except ImportError:
    _HAS_SHERPA = False


def _write_ogip_pha(path: Path, counts: np.ndarray, backscal: float) -> None:
    n_ch = len(counts)
    cols = [
        fits.Column(name="CHANNEL", format="J", array=np.arange(1, n_ch + 1, dtype=np.int32)),
        fits.Column(name="COUNTS", format="J", array=counts.astype(np.int32)),
    ]
    hdu = fits.BinTableHDU.from_columns(cols, name="SPECTRUM")
    hdr = hdu.header
    hdr["TELESCOP"] = "XMM"
    hdr["INSTRUME"] = "EMOS1"
    hdr["EXPOSURE"] = 10000.0
    hdr["BACKSCAL"] = float(backscal)
    hdr["AREASCAL"] = 1.0
    hdr["DETCHANS"] = n_ch
    hdr["HDUCLASS"] = "OGIP"
    hdr["HDUCLAS1"] = "SPECTRUM"
    hdr["HDUVERS"] = "1.2.1"
    hdr["CHANTYPE"] = "PI"
    hdr["POISSERR"] = True
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)


def _write_ogip_rmf(path: Path, e_lo: np.ndarray, e_hi: np.ndarray) -> None:
    n = len(e_lo)
    mat_cols = [
        fits.Column(name="ENERG_LO", format="E", array=e_lo.astype(np.float32)),
        fits.Column(name="ENERG_HI", format="E", array=e_hi.astype(np.float32)),
        fits.Column(name="N_GRP", format="I", array=np.ones(n, dtype=np.int16)),
        fits.Column(name="F_CHAN", format="I", array=np.arange(1, n + 1, dtype=np.int16)),
        fits.Column(name="N_CHAN", format="I", array=np.ones(n, dtype=np.int16)),
        fits.Column(name="MATRIX", format="E", array=np.ones(n, dtype=np.float32)),
    ]
    mat_hdu = fits.BinTableHDU.from_columns(mat_cols, name="MATRIX")
    mat_hdu.header["DETCHANS"] = n
    mat_hdu.header["TLMIN4"] = 1
    mat_hdu.header["HDUCLASS"] = "OGIP"
    mat_hdu.header["HDUCLAS1"] = "RESPONSE"
    mat_hdu.header["HDUCLAS2"] = "RSP_MATRIX"

    eb_cols = [
        fits.Column(name="CHANNEL", format="I", array=np.arange(1, n + 1, dtype=np.int16)),
        fits.Column(name="E_MIN", format="E", array=e_lo.astype(np.float32)),
        fits.Column(name="E_MAX", format="E", array=e_hi.astype(np.float32)),
    ]
    eb_hdu = fits.BinTableHDU.from_columns(eb_cols, name="EBOUNDS")
    eb_hdu.header["DETCHANS"] = n
    eb_hdu.header["HDUCLASS"] = "OGIP"
    eb_hdu.header["HDUCLAS1"] = "RESPONSE"
    eb_hdu.header["HDUCLAS2"] = "EBOUNDS"
    fits.HDUList([fits.PrimaryHDU(), mat_hdu, eb_hdu]).writeto(path, overwrite=True)


def _write_ogip_arf(path: Path, e_lo: np.ndarray, e_hi: np.ndarray) -> None:
    cols = [
        fits.Column(name="ENERG_LO", format="E", array=e_lo.astype(np.float32)),
        fits.Column(name="ENERG_HI", format="E", array=e_hi.astype(np.float32)),
        fits.Column(name="SPECRESP", format="E", array=np.full(len(e_lo), 500.0, dtype=np.float32)),
    ]
    hdu = fits.BinTableHDU.from_columns(cols, name="SPECRESP")
    hdu.header["HDUCLASS"] = "OGIP"
    hdu.header["HDUCLAS1"] = "RESPONSE"
    hdu.header["HDUCLAS2"] = "SPECRESP"
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)


def _make_synthetic_spectrum_files(
    directory: Path,
    prefix: str,
    *,
    is_pn: bool = False,
    backscal: float = 1440000.0,
) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    n_ch = 20
    e_edges = np.linspace(0.4, 2.4, n_ch + 1)
    e_lo, e_hi = e_edges[:-1], e_edges[1:]

    rng = np.random.default_rng(42)
    counts = rng.poisson(30, size=n_ch)
    bkg_counts = rng.poisson(5, size=n_ch)

    pha_name = f"{prefix}-fovtootsub.pi" if is_pn else f"{prefix}-fovt.pi"
    pha_p = directory / pha_name
    bkg_p = directory / f"{prefix}-bkg.pi"
    rmf_p = directory / f"{prefix}.rmf"
    arf_p = directory / f"{prefix}.arf"

    _write_ogip_pha(pha_p, counts, backscal)
    _write_ogip_pha(bkg_p, bkg_counts, backscal)
    _write_ogip_rmf(rmf_p, e_lo, e_hi)
    _write_ogip_arf(arf_p, e_lo, e_hi)
    return {"pha": pha_p, "bkg": bkg_p, "rmf": rmf_p, "arf": arf_p}


@unittest.skipUnless(_HAS_SHERPA, "Sherpa is not installed in this Python environment")
class TestSherpaOOP(unittest.TestCase):
    def test_builtin_diagonal_rmfs_load(self):
        from sherpa.astro.io import read_rmf
        from xmm_tools.sherpa.dataset import default_diagonal_rmf_path

        for inst in ("mos1", "mos2", "pn"):
            p = default_diagonal_rmf_path(inst)
            self.assertIsNotNone(p)
            self.assertTrue(p.exists())
            drmf = read_rmf(str(p))
            self.assertGreater(drmf.detchans, 100)

    def test_manifest_joint_fit_and_json_export(self):
        from xmm_tools.esas.spectrum import (
            InstrumentProducts,
            RegionSpectrumResult,
            SpectrumResult,
        )
        from xmm_tools.sherpa import (
            JointFitSession,
            SpectrumCollection,
        )

        with tempfile.TemporaryDirectory() as tmp:
            sdir = Path(tmp) / "spectrum"
            bkg_files_m1 = _make_synthetic_spectrum_files(
                sdir / "background", "mos1S001", backscal=1440000.0
            )
            bkg_files_pn = _make_synthetic_spectrum_files(
                sdir / "background", "pnS003", is_pn=True, backscal=1440000.0
            )
            src_files_m1 = _make_synthetic_spectrum_files(
                sdir / "east_jet" / "e4", "mos1S001", backscal=720000.0
            )

            spec_res = SpectrumResult(
                spectrum_dir=sdir,
                manifest_path=sdir / "manifest.json",
                regions={
                    "background": RegionSpectrumResult(
                        key="background",
                        group="background",
                        label="background",
                        mode="union",
                        region_dir=sdir / "background",
                        source_file=Path("bkg.reg"),
                        elow=500,
                        ehigh=2000,
                        instruments={
                            "mos1": InstrumentProducts(
                                fovt=bkg_files_m1["pha"],
                                bkg=bkg_files_m1["bkg"],
                                rmf=bkg_files_m1["rmf"],
                                arf=bkg_files_m1["arf"],
                                event_count=100,
                                proton_scale=80.0,
                            ),
                            "pn": InstrumentProducts(
                                fovtootsub=bkg_files_pn["pha"],
                                bkg=bkg_files_pn["bkg"],
                                rmf=bkg_files_pn["rmf"],
                                arf=bkg_files_pn["arf"],
                                event_count=100,
                                proton_scale=70.0,
                            ),
                        },
                    ),
                    "east_jet/e4": RegionSpectrumResult(
                        key="east_jet/e4",
                        group="east_jet",
                        label="e4",
                        mode="split",
                        region_dir=sdir / "east_jet" / "e4",
                        source_file=Path("east.reg"),
                        elow=500,
                        ehigh=2000,
                        instruments={
                            "mos1": InstrumentProducts(
                                fovt=src_files_m1["pha"],
                                bkg=src_files_m1["bkg"],
                                rmf=src_files_m1["rmf"],
                                arf=src_files_m1["arf"],
                                event_count=50,
                                proton_scale=10.0,
                            ),
                            "pn": InstrumentProducts(
                                skipped=True,
                                event_count=0,
                                skip_reason="zero events",
                            ),
                        },
                    ),
                },
            )
            spec_res.save_manifest()

            col = SpectrumCollection.from_manifest(spec_res.manifest_path)
            self.assertEqual(len(col), 3)
            self.assertAlmostEqual(col["background", "mos1"].solid_angle, 1.0, places=4)
            self.assertAlmostEqual(col["east_jet/e4", "mos1"].solid_angle, 0.5, places=4)
            self.assertAlmostEqual(
                col["east_jet/e4", "mos1"].proton_ratio(col["background", "mos1"]),
                0.125,
                places=5,
            )

            col.notice(0.5, 2.0)

            session = (
                JointFitSession(col)
                .set_calibration(
                    lambda m, ds: m.const1d(
                        "cal",
                        c0=1.0,
                        bounds={"c0": (0.1, 10.0)},
                        frozen=(ds.instrument == "pn"),
                    ),
                    scope="per_instrument",
                )
                .set_sky_background(
                    lambda m: m.powlaw1d("sky", gamma=1.4, ampl=1e-4, frozen=["gamma"]),
                    scope="global",
                    per_arcmin2=True,
                )
                .set_source(
                    lambda m: m.powlaw1d("jet", gamma=2.0, ampl=1e-4),
                    scope="region",
                    regions=["east_jet/e4"],
                )
                .set_soft_proton(
                    lambda m: m.powlaw1d("sp", gamma=0.8, ampl=1e-4),
                    scope="dataset",
                    use_proton_scale=True,
                )
                .set_instrument_lines(
                    lambda m, ds: (
                        m.gauss1d("al", pos=1.486, fwhm=0.01, ampl=1e-4, frozen=["pos", "fwhm"])
                        + m.gauss1d("si", pos=1.739, fwhm=0.01, ampl=1e-5, frozen=["pos", "fwhm"])
                        if ds.instrument in ("mos1", "mos2")
                        else m.gauss1d("al", pos=1.486, fwhm=0.01, ampl=1e-4, frozen=["pos", "fwhm"])
                    ),
                    scope="dataset",
                    response="unit_arf",
                )
                .build()
            )
            # Direct parameter linking & access on exposed Sherpa models
            session.link(
                "east_jet/e4.mos1.soft_proton.sp.gamma",
                "background.mos1.soft_proton.sp.gamma",
            )
            audit = session.audit_parameters()
            self.assertTrue(len(audit["free"]) > 0)
            self.assertEqual(len(audit["linked"]), 1)

            report = session.fit(stat="wstat", method="levmar", estimate_errors=False)
            self.assertTrue(report.succeeded)
            summary_txt = report.summary(only_free=True, include_linked=True)
            self.assertIn("Fit Summary: wstat", summary_txt)
            self.assertIn("->", summary_txt)
            json_path = report.save_json(Path(tmp) / "fit_report.json")
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(raw["stat_name"], "wstat")
            self.assertIn("east_jet__e4__mos1", raw["components"])
            self.assertIn("source_emission", raw["components"]["east_jet__e4__mos1"])
            self.assertIn("soft_proton", raw["components"]["east_jet__e4__mos1"])
            self.assertIn("instrument_lines", raw["components"]["east_jet__e4__mos1"])

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plot_file = Path(tmp) / "e4_mos1.png"
            fig, (ax_spec, ax_res) = session.plot(
                "east_jet/e4", instrument="mos1", save_path=plot_file
            )
            self.assertTrue(plot_file.exists() and plot_file.stat().st_size > 0)
            plt.close(fig)
