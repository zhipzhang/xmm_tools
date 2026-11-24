"""Sherpa OOP dataset and 3-channel response wrappers for XMM-Newton ESAS spectra."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from astropy.io import fits
import numpy as np

from xmm_tools.esas.spectrum import SpectrumResult

PIXEL_TO_ARCMIN2 = (0.05 / 60.0) ** 2  # 0.05 arcsec physical pixel -> arcmin^2
_BUILTIN_DIAG_RMF = {
    "mos1": "mos1-diag.rsp.gz",
    "mos2": "mos2-diag.rsp.gz",
    "pn": "pn-diag.rsp.gz",
}


def default_diagonal_rmf_path(instrument: str) -> Path | None:
    """Return the built-in ESAS diagonal RMF path for ``mos1``, ``mos2``, or ``pn``."""
    fname = _BUILTIN_DIAG_RMF.get(instrument.lower())
    if not fname:
        return None
    candidate = Path(__file__).resolve().parent / "data" / fname
    return candidate if candidate.exists() else None


def _require_sherpa() -> dict[str, Any]:
    try:
        from sherpa.astro.instrument import RMF1D, Response1D
        from sherpa.astro.io import read_arf, read_pha, read_rmf
        from sherpa.astro.data import DataRMF
    except ImportError as exc:
        raise ImportError(
            "Sherpa is required for xmm_tools.sherpa. "
            "Run inside a CIAO / Sherpa Python environment."
        ) from exc
    return {
        "RMF1D": RMF1D,
        "Response1D": Response1D,
        "read_arf": read_arf,
        "read_pha": read_pha,
        "read_rmf": read_rmf,
        "DataRMF": DataRMF,
    }


def make_diagonal_rmf(rmf: Any) -> Any:
    """Construct an in-memory unit diagonal ``DataRMF`` matching ``rmf`` energy grids."""
    sh = _require_sherpa()
    DataRMF = sh["DataRMF"]
    n_e = len(rmf.energ_lo)
    n_c = len(rmf.e_min)
    n_diag = min(n_e, n_c)

    # Map each energy bin to its diagonal channel (1-based FITS channel index)
    offset = int(getattr(rmf, "offset", 1))
    n_grp = np.zeros(n_e, dtype=np.int32)
    f_chan = np.arange(offset, offset + n_diag, dtype=np.int32)
    n_chan = np.ones(n_diag, dtype=np.int32)
    n_grp[:n_diag] = 1
    matrix = np.ones(n_diag, dtype=np.float64)

    return DataRMF(
        name=f"{getattr(rmf, 'name', 'rmf')}_diag",
        detchans=rmf.detchans,
        energ_lo=rmf.energ_lo,
        energ_hi=rmf.energ_hi,
        n_grp=n_grp,
        f_chan=f_chan,
        n_chan=n_chan,
        matrix=matrix,
        offset=offset,
        e_min=rmf.e_min,
        e_max=rmf.e_max,
    )


def _read_pha_header_meta(pha_path: Path) -> tuple[float, float]:
    """Return ``(solid_angle_arcmin2, exposure_s)`` from a PHA SPECTRUM header."""
    with fits.open(pha_path) as hdul:
        hdr = hdul["SPECTRUM"].header
        backscal = float(hdr.get("BACKSCAL", 1.0))
        exposure = float(hdr.get("EXPOSURE", 1.0))
    return backscal * PIXEL_TO_ARCMIN2, exposure


@dataclass
class SpectrumDataset:
    """OOP wrapper around one Sherpa ``DataPHA`` with 3-channel ESAS response helpers."""

    ds_id: str
    region_key: str
    instrument: str
    pha_path: Path
    pha: Any
    rmf: Any
    arf: Any
    bkg: Any | None = None
    diag_rmf: Any | None = None
    solid_angle: float = 1.0
    exposure: float = 1.0

    @classmethod
    def from_files(
        cls,
        ds_id: str,
        *,
        region_key: str,
        instrument: str,
        pha_path: str | Path,
        rmf_path: str | Path,
        arf_path: str | Path,
        bkg_path: str | Path | None = None,
        diag_rmf_path: str | Path | None = None,
        use_errors: bool = True,
    ) -> SpectrumDataset:
        sh = _require_sherpa()
        pha_p = Path(pha_path)
        pha = sh["read_pha"](str(pha_p), use_errors=use_errors)
        rmf = sh["read_rmf"](str(rmf_path))
        arf = sh["read_arf"](str(arf_path))
        pha.set_rmf(rmf)
        pha.set_arf(arf)
        pha.set_analysis("energy")

        bkg = None
        if bkg_path is not None:
            bkg = sh["read_pha"](str(bkg_path), use_errors=use_errors)
            pha.set_background(bkg)

        resolved_diag = (
            Path(diag_rmf_path)
            if diag_rmf_path is not None
            else default_diagonal_rmf_path(instrument)
        )
        if resolved_diag is not None:
            diag_rmf = sh["read_rmf"](str(resolved_diag))
            diag_rmf.offset = getattr(rmf, "offset", 1)
            if diag_rmf.detchans != rmf.detchans and diag_rmf_path is None:
                diag_rmf = make_diagonal_rmf(rmf)
        else:
            diag_rmf = make_diagonal_rmf(rmf)

        solid_angle, exposure = _read_pha_header_meta(pha_p)
        return cls(
            ds_id=ds_id,
            region_key=region_key,
            instrument=instrument,
            pha_path=pha_p,
            pha=pha,
            rmf=rmf,
            arf=arf,
            bkg=bkg,
            diag_rmf=diag_rmf,
            solid_angle=solid_angle,
            exposure=exposure,
        )

    def notice(self, emin: float | None = None, emax: float | None = None) -> SpectrumDataset:
        self.pha.ignore()
        self.pha.notice(emin, emax)
        return self

    def group_counts(self, num: int) -> SpectrumDataset:
        self.pha.group_counts(int(num))
        return self

    def group_snr(self, snr: float) -> SpectrumDataset:
        self.pha.group_snr(float(snr))
        return self

    def subtract(self) -> SpectrumDataset:
        self.pha.subtract()
        return self

    def unsubtract(self) -> SpectrumDataset:
        self.pha.unsubtract()
        return self

    def fold_sky(self, model: Any) -> Any:
        """Convolve celestial model with ``rmf + arf``."""
        sh = _require_sherpa()
        rsp = sh["Response1D"](self.pha)
        return rsp(model)

    def fold_diagonal(self, model: Any) -> Any:
        """Convolve soft-proton model with ``diag_rmf`` (without ARF)."""
        sh = _require_sherpa()
        if self.diag_rmf is None:
            self.diag_rmf = make_diagonal_rmf(self.rmf)
        return sh["RMF1D"](self.diag_rmf, pha=self.pha)(model)

    def fold_unit_arf(self, model: Any) -> Any:
        """Convolve detector line model with ``rmf`` only (unit ARF)."""
        sh = _require_sherpa()
        return sh["RMF1D"](self.rmf, pha=self.pha)(model)


@dataclass
class SpectrumCollection:
    """Collection of ``SpectrumDataset`` objects indexed by ``ds_id``, region, and instrument."""

    datasets: dict[str, SpectrumDataset] = field(default_factory=dict)

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        regions: Iterable[str] | None = None,
        instruments: Iterable[str] = ("mos1", "mos2", "pn"),
        diag_rmf: dict[str, str | Path] | None = None,
        use_errors: bool = True,
    ) -> SpectrumCollection:
        """Load active spectra from ``spectrum/manifest.json`` (skipping 0-event instruments)."""
        spec_res = SpectrumResult.from_manifest(Path(manifest_path))
        selected_regions = set(regions) if regions is not None else None
        inst_filter = tuple(instruments)
        diag_map = diag_rmf or {}

        loaded: dict[str, SpectrumDataset] = {}
        for reg_key, reg_res in sorted(spec_res.regions.items()):
            if selected_regions is not None and reg_key not in selected_regions:
                continue
            safe_reg = reg_key.replace("/", "__")
            for inst in inst_filter:
                prod = reg_res.instruments.get(inst)
                if prod is None or prod.skipped:
                    continue
                pha_p = prod.fovtootsub if inst == "pn" else prod.fovt
                if pha_p is None or prod.rmf is None or prod.arf is None:
                    continue
                ds_id = f"{safe_reg}__{inst}"
                loaded[ds_id] = SpectrumDataset.from_files(
                    ds_id,
                    region_key=reg_key,
                    instrument=inst,
                    pha_path=pha_p,
                    rmf_path=prod.rmf,
                    arf_path=prod.arf,
                    bkg_path=prod.bkg,
                    diag_rmf_path=diag_map.get(inst),
                    use_errors=use_errors,
                )
        return cls(datasets=loaded)

    def __getitem__(self, key: str | tuple[str, str]) -> SpectrumDataset:
        if isinstance(key, tuple):
            reg_key, inst = key
            for ds in self.datasets.values():
                if ds.region_key == reg_key and ds.instrument == inst:
                    return ds
            raise KeyError(f"No dataset for region={reg_key!r}, instrument={inst!r}")
        return self.datasets[key]

    def __iter__(self):
        return iter(self.datasets.values())

    def __len__(self) -> int:
        return len(self.datasets)

    @property
    def regions(self) -> list[str]:
        return list(dict.fromkeys(ds.region_key for ds in self.datasets.values()))

    @property
    def instruments(self) -> list[str]:
        return list(dict.fromkeys(ds.instrument for ds in self.datasets.values()))

    def by_region(self, region_key: str) -> list[SpectrumDataset]:
        return [ds for ds in self.datasets.values() if ds.region_key == region_key]

    def by_instrument(self, instrument: str) -> list[SpectrumDataset]:
        return [ds for ds in self.datasets.values() if ds.instrument == instrument]

    def notice(self, emin: float | None = None, emax: float | None = None) -> SpectrumCollection:
        for ds in self.datasets.values():
            ds.notice(emin, emax)
        return self

    def group_counts(self, num: int) -> SpectrumCollection:
        for ds in self.datasets.values():
            ds.group_counts(num)
        return self

    def group_snr(self, snr: float) -> SpectrumCollection:
        for ds in self.datasets.values():
            ds.group_snr(snr)
        return self

    def subtract(self) -> SpectrumCollection:
        for ds in self.datasets.values():
            ds.subtract()
        return self
