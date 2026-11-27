"""Tests for XMM-Newton QDP parsing and GUI state helpers."""

import tempfile
import unittest
from pathlib import Path

from xmm_tools.qdp import filter_series_arrays, load_qdp, parse_qdp_text

try:
    from xmm_tools.qdp.gui import QDPViewerState
except ModuleNotFoundError as exc:
    if exc.name != "imgui_bundle" and not (exc.name or "").startswith("imgui_bundle."):
        raise
    QDPViewerState = None  # type: ignore[assignment,misc]


EXAMPLE_HIST = Path(
    "/Users/zhipz/CodeProjects/XMMNewton_Workdir/analysis/"
    "2026_Oct_SS433_analysis/esas_analysis/soft_proton_flare/mos2S002-hist.qdp"
)
EXAMPLE_BKGSPEC = Path(
    "/Users/zhipz/CodeProjects/XMMNewton_Workdir/analysis/"
    "2026_Oct_SS433_analysis/esas_analysis/spectrum_revised_study/background/pnS003-bkgspec.qdp"
)


class TestQDPParser(unittest.TestCase):
    def test_parse_synthetic_qdp_with_no_separator_and_terr(self):
        text = """\
! Comment line
READ SERR 2
READ TERR 3
win 1
yplot 1 2
la t Test Spectrum
la x Energy (keV)
la y Counts
log x on
log y on
r x 0.2 10.0
1.0  10.0 1.0  5.0 0.5 -0.4
2.0  20.0 2.0  8.0 0.8 -0.6
NO NO NO NO NO NO
3.0  30.0 3.0  9.0 0.9 -0.7
"""
        data = parse_qdp_text(text)
        self.assertEqual(len(data.raw_tables), 2)
        self.assertEqual(data.raw_tables[0].shape, (2, 6))
        self.assertEqual(data.raw_tables[1].shape, (1, 6))
        self.assertEqual(len(data.windows), 1)
        win = data.windows[0]
        self.assertEqual(win.title, "Test Spectrum")
        self.assertEqual(win.xlabel, "Energy (keV)")
        self.assertEqual(win.ylabel, "Counts")
        self.assertTrue(win.log_x)
        self.assertTrue(win.log_y)
        self.assertEqual(win.xlim, (0.2, 10.0))

        self.assertEqual(len(data.series), 2)
        s1, s2 = data.series
        self.assertIsNotNone(s1.yerr_low)
        self.assertAlmostEqual(float(s1.yerr_low[0]), 1.0)
        self.assertIsNotNone(s2.yerr_high)
        self.assertIsNotNone(s2.yerr_low)
        self.assertAlmostEqual(float(s2.yerr_high[0]), 0.5)
        self.assertAlmostEqual(float(s2.yerr_low[0]), 0.4)

        tables = data.to_astropy_tables()
        self.assertEqual(len(tables), 2)
        self.assertEqual(len(tables[0]), 2)

    def test_load_example_hist_qdp(self):
        if not EXAMPLE_HIST.exists():
            self.skipTest(f"Example file not found: {EXAMPLE_HIST}")
        data = load_qdp(EXAMPLE_HIST)
        self.assertEqual(len(data.raw_tables), 1)
        self.assertEqual(data.raw_tables[0].shape, (3789, 14))
        self.assertEqual(len(data.windows), 3)
        self.assertEqual(data.windows[0].y_vectors, [1, 2, 7, 8])
        self.assertEqual(data.windows[1].y_vectors, [3, 4])
        self.assertEqual(data.windows[2].y_vectors, [5, 6])
        self.assertEqual(data.series[6].color_index, 2)  # col 2 on 7 (Red)
        self.assertEqual(data.series[7].color_index, 4)  # col 4 on 8 (Blue)
        self.assertFalse(data.windows[0].log_y)
        self.assertTrue(data.windows[1].log_y)
        self.assertTrue(data.series[0].step)  # line step 2 -> vec 2 (y_index 1)
        self.assertGreater(len(data.windows[0].labels), 0)

        # Verify sentinel -1.0 filtering and log filtering
        xs, ys, _, _ = filter_series_arrays(
            data.series[0], hide_minus_one=True, log_x=False, log_y=False
        )
        self.assertEqual(len(xs), 657)
        self.assertEqual(len(ys), 657)

    def test_load_example_bkgspec_qdp(self):
        if not EXAMPLE_BKGSPEC.exists():
            self.skipTest(f"Example file not found: {EXAMPLE_BKGSPEC}")
        data = load_qdp(EXAMPLE_BKGSPEC)
        self.assertEqual(len(data.raw_tables), 1)
        self.assertEqual(data.raw_tables[0].shape, (4096, 4))
        self.assertEqual(len(data.windows), 1)
        win = data.windows[0]
        self.assertEqual(win.title, "0991990101 -- PNS003")
        self.assertEqual(win.xlabel, "Energy (keV)")
        self.assertEqual(win.ylabel, "Counts")
        self.assertTrue(win.log_x)
        self.assertTrue(win.log_y)
        self.assertEqual(win.xlim, (0.2, 15.0))
        self.assertEqual(len(data.series), 2)
        self.assertTrue(data.series[0].visible)
        self.assertFalse(data.series[1].visible)  # line off 3 / col off 3

        if QDPViewerState is not None:
            state = QDPViewerState(
                recent_store_path=Path(tempfile.mkdtemp()) / "recent.json"
            )
            state.open_file(EXAMPLE_BKGSPEC)
            self.assertEqual(state.error_message, "")
            self.assertIsNotNone(state.data)

    @unittest.skipIf(QDPViewerState is None, "imgui_bundle not installed")
    def test_recent_files_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "recent.json"
            q1 = Path(tmp) / "a.qdp"
            q2 = Path(tmp) / "b.qdp"
            q1.write_text("1 2\n", encoding="utf-8")
            q2.write_text("3 4\n", encoding="utf-8")

            s1 = QDPViewerState(recent_store_path=store)
            s1.open_file(q1)
            s1.open_file(q2)
            s1.open_file(q1)  # re-opening moves q1 to front without duplicating
            self.assertEqual(s1.recent_files, [str(q1.resolve()), str(q2.resolve())])

            s2 = QDPViewerState(recent_store_path=store)
            self.assertEqual(s2.recent_files, [str(q1.resolve()), str(q2.resolve())])

            s2.clear_recent()
            self.assertEqual(s2.recent_files, [])


if __name__ == "__main__":
    unittest.main()
