"""Single-pass TOML configuration loader for ``xmm-ana``."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]

from xmm_tools.config.schema import (
    AnaConfig,
    BandSpec,
    CheeseConfig,
    FlareConfig,
    InstrumentsConfig,
    PathsConfig,
    QPBImageConfig,
    SpectrumConfig,
    SpectrumRegionSpec,
    default_qpb_bands,
)
from xmm_tools.exec.env import SASEnv


def _table(raw: Mapping[str, Any], key: str) -> dict[str, Any]:
    val = raw.get(key, {})
    if not isinstance(val, dict):
        raise TypeError(f"Expected [{key}] table in TOML config")
    return val


def _parse_regions(spec_sec: Mapping[str, Any]) -> list[SpectrumRegionSpec]:
    raw_regions = spec_sec.get("regions", [])
    parsed: list[SpectrumRegionSpec] = []
    for idx, item in enumerate(raw_regions):
        if isinstance(item, str):
            parsed.append(
                SpectrumRegionSpec(
                    name=Path(item).stem or f"reg_{idx + 1}",
                    file=item,
                )
            )
        elif isinstance(item, dict):
            if "file" not in item:
                raise ValueError(
                    f"Missing required 'file' key in [[esas.spectrum.regions]] entry #{idx + 1}"
                )
            file_val = str(item["file"])
            name_val = str(item.get("name") or Path(file_val).stem)
            parsed.append(
                SpectrumRegionSpec(
                    name=name_val,
                    file=file_val,
                    mode=str(item.get("mode", "single")),
                    overwrite=bool(item.get("overwrite", False)),
                    elow=int(item["elow"]) if item.get("elow") is not None else None,
                    ehigh=int(item["ehigh"]) if item.get("ehigh") is not None else None,
                )
            )
        else:
            raise TypeError(
                "Expected string or table entry in [esas.spectrum].regions"
            )
    return parsed


def _parse_qpb_bands(qi_sec: Mapping[str, Any]) -> dict[str, BandSpec]:
    qi_bands_raw = qi_sec.get("bands")
    if not isinstance(qi_bands_raw, dict) or not qi_bands_raw:
        return default_qpb_bands()

    bands: dict[str, BandSpec] = {}
    for bname, bval in qi_bands_raw.items():
        if not isinstance(bval, dict):
            raise TypeError(f"Expected [sas.qpb_image.bands.{bname}] table")
        if "emin" not in bval or "emax" not in bval:
            raise ValueError(
                f"Missing 'emin' or 'emax' in [sas.qpb_image.bands.{bname}]"
            )
        bands[bname] = BandSpec(
            name=bname,
            emin=int(bval["emin"]),
            emax=int(bval["emax"]),
            subbands=[[int(a), int(b)] for a, b in bval.get("subbands", [])],
        )
    return bands


def load_ana_config(path: str | Path) -> AnaConfig:
    """Parse and validate an ``AnaConfig`` from ``path`` in a single TOML pass."""
    toml_path = Path(path)
    raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))

    paths_sec = _table(raw, "paths")
    workdir_raw = paths_sec.get("workdir", ".")
    workdir = Path(workdir_raw)
    if not workdir.is_absolute():
        workdir = (toml_path.parent / workdir).resolve()

    env = SASEnv.from_mapping(_table(raw, "env"), base_dir=workdir)

    sas_sec = _table(raw, "sas")
    qi_sec = _table(sas_sec, "qpb_image")

    paths = PathsConfig(
        workdir=workdir,
        flare_dir=str(paths_sec.get("flare_dir", "soft_proton_flare")),
        cheese_dir=str(paths_sec.get("cheese_dir", "cheese")),
        qpb_image_dir=str(
            paths_sec.get("qpb_image_dir", qi_sec.get("dir", "qpb_image"))
        ),
        spectrum_dir=str(paths_sec.get("spectrum_dir", "spectrum")),
    )

    inst_sec = _table(raw, "instruments")
    instruments = InstrumentsConfig(
        mos1=inst_sec.get("mos1", "mos1_evt.fits"),
        mos2=inst_sec.get("mos2", "mos2_evt.fits"),
        pn=inst_sec.get("pn", "pn_evt.fits"),
        pn_oot=inst_sec.get("pn_oot", "pn_oot_evt.fits"),
    )

    esas_sec = _table(raw, "esas")

    esp_sec = _table(esas_sec, "espfilt")
    flare_sec = _table(esas_sec, "flare")
    merged_flare = {**esp_sec, **flare_sec}
    flare = FlareConfig(
        elow=int(merged_flare.get("elow", 2500)),
        ehigh=int(merged_flare.get("ehigh", 8500)),
        withsmoothing=bool(merged_flare.get("withsmoothing", True)),
        smooth=int(merged_flare.get("smooth", 51)),
        rangescale_mos=float(merged_flare.get("rangescale_mos", 6.0)),
        rangescale_pn=float(merged_flare.get("rangescale_pn", 15.0)),
        allowsigma=float(merged_flare.get("allowsigma", 3.0)),
        method=str(merged_flare.get("method", "histogram")),
        keepinterfiles=bool(merged_flare.get("keepinterfiles", False)),
        apply_flag_filter=bool(merged_flare.get("apply_flag_filter", True)),
        mos_flag_expr=str(
            merged_flare.get("mos_flag_expr", "((FLAG & 0x766aa000)==0)")
        ),
        pn_flag_expr=str(
            merged_flare.get("pn_flag_expr", "((FLAG & 0x762a097c)==0)")
        ),
    )

    ch_sec = _table(esas_sec, "cheese")
    cheese = CheeseConfig(
        elowlist=[int(x) for x in ch_sec.get("elowlist", [350, 2500])],
        ehighlist=[int(x) for x in ch_sec.get("ehighlist", [1100, 8000])],
        scale=float(ch_sec.get("scale", 0.4)),
        mlmin=float(ch_sec.get("mlmin", 15.0)),
        dist=float(ch_sec.get("dist", 50.0)),
        ratesoft=float(ch_sec.get("ratesoft", 0.2)),
        ratehard=float(ch_sec.get("ratehard", 0.2)),
        ratetotal=float(ch_sec.get("ratetotal", 0.2)),
        keepinterfiles=bool(ch_sec.get("keepinterfiles", False)),
        extra_params=dict(ch_sec.get("extra_params", {})),
    )

    spec_sec = _table(esas_sec, "spectrum")
    spectrum = SpectrumConfig(
        max_workers=int(spec_sec.get("max_workers", 2)),
        elow=int(spec_sec.get("elow", 400)),
        ehigh=int(spec_sec.get("ehigh", 12500)),
        pattern_mos=int(spec_sec.get("pattern_mos", 12)),
        pattern_pn=int(spec_sec.get("pattern_pn", 4)),
        withsrcrem=bool(spec_sec.get("withsrcrem", True)),
        keepinterfiles=bool(spec_sec.get("keepinterfiles", True)),
        withplotfiles=bool(spec_sec.get("withplotfiles", True)),
        mos1_ccds=str(spec_sec.get("mos1_ccds", "T T F T T F T")),
        mos2_ccds=str(spec_sec.get("mos2_ccds", "T T T T T T T")),
        pn_quads=str(spec_sec.get("pn_quads", "T T T T")),
        regions=_parse_regions(spec_sec),
        spectra_extra_params=dict(spec_sec.get("spectra_extra_params", {})),
        back_extra_params=dict(spec_sec.get("back_extra_params", {})),
    )

    qi_gti_raw = qi_sec.get("gti_files", {})
    if not isinstance(qi_gti_raw, dict):
        raise TypeError("Expected [sas.qpb_image.gti_files] table in TOML config")
    qi_gti_files = {
        str(k): str(v)
        for k, v in qi_gti_raw.items()
        if v is not None and str(v).strip()
    }

    qpb_image = QPBImageConfig(
        dir=paths.qpb_image_dir,
        bin_size=int(qi_sec.get("bin_size", 50)),
        ecut=float(qi_sec.get("ecut", 1000.0)),
        epn_weight=float(qi_sec.get("epn_weight", 1.0)),
        em1_weight=float(qi_sec.get("em1_weight", 0.4)),
        em2_weight=float(qi_sec.get("em2_weight", 0.4)),
        smoothstyle=str(qi_sec.get("smoothstyle", "simple")),
        convolverstyle=str(qi_sec.get("convolverstyle", "gaussian")),
        width=float(qi_sec.get("width", 5.0)),
        minwidth=float(qi_sec.get("minwidth", 0.0)),
        maxwidth=float(qi_sec.get("maxwidth", 10.0)),
        desiredsnr=float(qi_sec.get("desiredsnr", 10.0)),
        nconvolvers=int(qi_sec.get("nconvolvers", 20)),
        attfile=str(qi_sec.get("attfile", "")),
        gti_files=qi_gti_files,
        default_subband_edges=[
            int(x)
            for x in qi_sec.get(
                "default_subband_edges",
                [350, 500, 750, 1000, 1500, 2000, 3000, 4500, 7000, 10000],
            )
        ],
        bands=_parse_qpb_bands(qi_sec),
        evqpb_extra_params=dict(qi_sec.get("evqpb_extra_params", {})),
        eimageget_extra_params=dict(qi_sec.get("eimageget_extra_params", {})),
        eimagecombine_extra_params=dict(
            qi_sec.get("eimagecombine_extra_params", {})
        ),
    )

    return AnaConfig(
        env=env,
        paths=paths,
        instruments=instruments,
        qpb_image=qpb_image,
        flare=flare,
        cheese=cheese,
        spectrum=spectrum,
    )
