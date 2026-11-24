"""TOML configuration schema and loader for XMM-Newton ESAS workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]

from xmm_tools.sas.env import SASEnv


@dataclass
class PathsConfig:
    """Relative/absolute subdirectories for ESAS analysis outputs."""

    workdir: Path = Path(".")
    flare_dir: str = "soft_proton_flare"
    cheese_dir: str = "cheese"
    spectrum_dir: str = "spectrum"
    image_dir: str = "images"

    def resolve(self, sub: str) -> Path:
        p = Path(sub)
        return p if p.is_absolute() else self.workdir / p


@dataclass
class InstrumentsConfig:
    """Standardized event-file names produced by ``xmm-esas init``."""

    mos1: str | None = "mos1_evt.fits"
    mos2: str | None = "mos2_evt.fits"
    pn: str | None = "pn_evt.fits"
    pn_oot: str | None = "pn_oot_evt.fits"

    def active_mos(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        if self.mos1:
            out.append(("mos1", self.mos1))
        if self.mos2:
            out.append(("mos2", self.mos2))
        return out


@dataclass
class EspfiltConfig:
    """Parameters for ``espfilt`` soft-proton flare filtering."""

    elow: int = 2500
    ehigh: int = 8500
    withsmoothing: bool = True
    smooth: int = 51
    rangescale_mos: float = 6.0
    rangescale_pn: float = 15.0
    allowsigma: float = 3.0
    method: str = "histogram"
    keepinterfiles: bool = False
    apply_flag_filter: bool = True
    mos_flag_expr: str = "((FLAG & 0x766aa000)==0)"
    pn_flag_expr: str = "((FLAG & 0x762a097c)==0)"


@dataclass
class SourceDetectConfig:
    """Parameters for point-source detection and swiss-cheese masking."""

    method: str = "cheese"
    elowlist: list[int] = field(default_factory=lambda: [350, 2500])
    ehighlist: list[int] = field(default_factory=lambda: [1100, 8000])
    scale: float = 0.4
    mlmin: float = 15.0
    dist: float = 50.0
    ratesoft: float = 0.2
    ratehard: float = 0.2
    ratetotal: float = 0.2
    keepinterfiles: bool = False
    extra_regions: list[str] = field(default_factory=list)


@dataclass
class SpectrumConfig:
    """Parameters for ``mosspectra``/``pnspectra`` and ``mosback``/``pnback``."""

    max_workers: int = 2
    elow: int = 400
    ehigh: int = 12500
    regions: list[str] = field(default_factory=list)


@dataclass
class ImagingConfig:
    """Parameters for ESAS multi-band imaging."""

    bands: list[list[int]] = field(
        default_factory=lambda: [[400, 1250], [2000, 7000]]
    )
    binning: int = 2


@dataclass
class ESASConfig:
    """Top-level ESAS pipeline configuration."""

    env: SASEnv = field(default_factory=SASEnv)
    paths: PathsConfig = field(default_factory=PathsConfig)
    instruments: InstrumentsConfig = field(default_factory=InstrumentsConfig)
    espfilt: EspfiltConfig = field(default_factory=EspfiltConfig)
    source_detect: SourceDetectConfig = field(default_factory=SourceDetectConfig)
    spectrum: SpectrumConfig = field(default_factory=SpectrumConfig)
    imaging: ImagingConfig = field(default_factory=ImagingConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> ESASConfig:
        """Load an ``ESASConfig`` from a TOML file."""
        toml_path = Path(path)
        raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        env = SASEnv.from_toml(toml_path)

        paths_sec = _table(raw, "paths")
        workdir_raw = paths_sec.get("workdir", ".")
        workdir = Path(workdir_raw)
        if not workdir.is_absolute():
            workdir = (toml_path.parent / workdir).resolve()

        paths = PathsConfig(
            workdir=workdir,
            flare_dir=str(paths_sec.get("flare_dir", "soft_proton_flare")),
            cheese_dir=str(paths_sec.get("cheese_dir", "cheese")),
            spectrum_dir=str(paths_sec.get("spectrum_dir", "spectrum")),
            image_dir=str(paths_sec.get("image_dir", "images")),
        )

        inst_sec = _table(raw, "instruments")
        instruments = InstrumentsConfig(
            mos1=inst_sec.get("mos1", "mos1_evt.fits"),
            mos2=inst_sec.get("mos2", "mos2_evt.fits"),
            pn=inst_sec.get("pn", "pn_evt.fits"),
            pn_oot=inst_sec.get("pn_oot", "pn_oot_evt.fits"),
        )

        esp_sec = _table(raw, "espfilt")
        espfilt = EspfiltConfig(
            elow=int(esp_sec.get("elow", 2500)),
            ehigh=int(esp_sec.get("ehigh", 8500)),
            withsmoothing=bool(esp_sec.get("withsmoothing", True)),
            smooth=int(esp_sec.get("smooth", 51)),
            rangescale_mos=float(esp_sec.get("rangescale_mos", 6.0)),
            rangescale_pn=float(esp_sec.get("rangescale_pn", 15.0)),
            allowsigma=float(esp_sec.get("allowsigma", 3.0)),
            method=str(esp_sec.get("method", "histogram")),
            keepinterfiles=bool(esp_sec.get("keepinterfiles", False)),
            apply_flag_filter=bool(esp_sec.get("apply_flag_filter", True)),
            mos_flag_expr=str(
                esp_sec.get("mos_flag_expr", "((FLAG & 0x766aa000)==0)")
            ),
            pn_flag_expr=str(
                esp_sec.get("pn_flag_expr", "((FLAG & 0x762a097c)==0)")
            ),
        )

        sd_sec = _table(raw, "source_detect") or _table(raw, "cheese")
        source_detect = SourceDetectConfig(
            method=str(sd_sec.get("method", "cheese")),
            elowlist=[int(x) for x in sd_sec.get("elowlist", [350, 2500])],
            ehighlist=[int(x) for x in sd_sec.get("ehighlist", [1100, 8000])],
            scale=float(sd_sec.get("scale", 0.4)),
            mlmin=float(sd_sec.get("mlmin", 15.0)),
            dist=float(sd_sec.get("dist", 50.0)),
            ratesoft=float(sd_sec.get("ratesoft", 0.2)),
            ratehard=float(sd_sec.get("ratehard", 0.2)),
            ratetotal=float(sd_sec.get("ratetotal", 0.2)),
            keepinterfiles=bool(sd_sec.get("keepinterfiles", False)),
            extra_regions=[str(x) for x in sd_sec.get("extra_regions", [])],
        )

        spec_sec = _table(raw, "spectrum")
        spectrum = SpectrumConfig(
            max_workers=int(spec_sec.get("max_workers", 2)),
            elow=int(spec_sec.get("elow", 400)),
            ehigh=int(spec_sec.get("ehigh", 12500)),
            regions=[str(x) for x in spec_sec.get("regions", [])],
        )

        img_sec = _table(raw, "imaging")
        imaging = ImagingConfig(
            bands=[
                [int(a), int(b)]
                for a, b in img_sec.get("bands", [[400, 1250], [2000, 7000]])
            ],
            binning=int(img_sec.get("binning", 2)),
        )

        return cls(
            env=env,
            paths=paths,
            instruments=instruments,
            espfilt=espfilt,
            source_detect=source_detect,
            spectrum=spectrum,
            imaging=imaging,
        )


def _table(raw: dict[str, Any], key: str) -> dict[str, Any]:
    val = raw.get(key, {})
    if not isinstance(val, dict):
        raise ValueError(f"Expected [{key}] table in TOML config")
    return val


def render_template_toml(
    *,
    workdir: str = ".",
    sas_ccf: str = "ccf.cif",
    sas_ccfpath: str = "/ccf/valid",
    sas_odf: str = "SUM.SAS",
    mos1: str = "mos1_evt.fits",
    mos2: str = "mos2_evt.fits",
    pn: str = "pn_evt.fits",
    pn_oot: str = "pn_oot_evt.fits",
) -> str:
    """Render a commented ``esas.toml`` configuration file."""
    return f"""\
# XMM-Newton ESAS analysis configuration

[env]
sas_ccf = "{sas_ccf}"
sas_ccfpath = "{sas_ccfpath}"
sas_odf = "{sas_odf}"

[paths]
workdir = "{workdir}"
flare_dir = "soft_proton_flare"
cheese_dir = "cheese"
spectrum_dir = "spectrum"
image_dir = "images"

[instruments]
mos1 = "{mos1}"
mos2 = "{mos2}"
pn = "{pn}"
pn_oot = "{pn_oot}"

[espfilt]
elow = 2500
ehigh = 8500
withsmoothing = true
smooth = 51
rangescale_mos = 6.0
rangescale_pn = 15.0
allowsigma = 3.0
method = "histogram"
keepinterfiles = false
apply_flag_filter = true
mos_flag_expr = "((FLAG & 0x766aa000)==0)"
pn_flag_expr = "((FLAG & 0x762a097c)==0)"

[source_detect]
method = "cheese"
elowlist = [350, 2500]
ehighlist = [1100, 8000]
scale = 0.4
mlmin = 15.0
dist = 50.0
ratesoft = 0.2
ratehard = 0.2
ratetotal = 0.2
keepinterfiles = false
extra_regions = []

[spectrum]
max_workers = 2
elow = 400
ehigh = 12500
regions = []

[imaging]
bands = [[400, 1250], [2000, 7000]]
binning = 2
"""
