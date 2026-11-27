"""TOML configuration schema dataclasses and centralized validation."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any

from xmm_tools.exec.env import SASEnv

VALID_REGION_MODES = ("single", "split", "union")


@dataclass
class PathsConfig:
    """Relative or absolute subdirectories for analysis outputs."""

    workdir: Path = Path(".")
    flare_dir: str = "soft_proton_flare"
    cheese_dir: str = "cheese"
    qpb_image_dir: str = "qpb_image"
    spectrum_dir: str = "spectrum"

    def resolve(self, sub: str | Path) -> Path:
        p = Path(sub)
        if p.is_absolute():
            return p
        return (
            (self.workdir / p).resolve()
            if not self.workdir.is_absolute()
            else self.workdir / p
        )


@dataclass
class InstrumentsConfig:
    """Standardized event-file names produced by ``xmm-ana init``."""

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

    def active_keys(self) -> list[str]:
        keys: list[str] = []
        if self.mos1:
            keys.append("mos1")
        if self.mos2:
            keys.append("mos2")
        if self.pn:
            keys.append("pn")
        return keys


@dataclass
class FlareConfig:
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

    def __post_init__(self) -> None:
        if self.elow < 0 or self.elow >= self.ehigh:
            raise ValueError(
                f"Invalid flare energy bounds: elow={self.elow}, ehigh={self.ehigh} "
                "(expected 0 <= elow < ehigh)"
            )
        if self.allowsigma <= 0:
            raise ValueError(
                f"Invalid flare allowsigma={self.allowsigma}; must be > 0"
            )


EspfiltConfig = FlareConfig


@dataclass
class CheeseConfig:
    """Parameters for ``cheese`` point-source detection and swiss-cheese masking."""

    elowlist: list[int] = field(default_factory=lambda: [350, 2500])
    ehighlist: list[int] = field(default_factory=lambda: [1100, 8000])
    scale: float = 0.4
    mlmin: float = 15.0
    dist: float = 50.0
    ratesoft: float = 0.2
    ratehard: float = 0.2
    ratetotal: float = 0.2
    keepinterfiles: bool = False
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpectrumRegionSpec:
    """One DS9 or SAS detector region entry for ``spectrum`` extraction."""

    name: str
    file: str
    mode: str = "single"
    overwrite: bool = False
    elow: int | None = None
    ehigh: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in VALID_REGION_MODES:
            raise ValueError(
                f"Invalid spectrum region mode {self.mode!r}; "
                f"expected one of {VALID_REGION_MODES}"
            )
        if not self.name.strip():
            raise ValueError("Spectrum region 'name' must not be empty")
        if not self.file.strip():
            raise ValueError("Spectrum region 'file' must not be empty")
        if (
            self.elow is not None
            and self.ehigh is not None
            and (self.elow < 0 or self.elow >= self.ehigh)
        ):
            raise ValueError(
                f"Invalid energy bounds for spectrum region {self.name!r}: "
                f"elow={self.elow}, ehigh={self.ehigh}"
            )


@dataclass
class SpectrumConfig:
    """Parameters for ``mosspectra``/``pnspectra`` and ``mosback``/``pnback``."""

    max_workers: int = 2
    elow: int = 400
    ehigh: int = 12500
    pattern_mos: int = 12
    pattern_pn: int = 4
    withsrcrem: bool = True
    keepinterfiles: bool = True
    withplotfiles: bool = True
    mos1_ccds: str = "T T F T T F T"
    mos2_ccds: str = "T T T T T T T"
    pn_quads: str = "T T T T"
    regions: list[SpectrumRegionSpec] = field(default_factory=list)
    spectra_extra_params: dict[str, Any] = field(default_factory=dict)
    back_extra_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError(
                f"Invalid spectrum max_workers={self.max_workers}; must be >= 1"
            )
        if self.elow < 0 or self.elow >= self.ehigh:
            raise ValueError(
                f"Invalid spectrum energy bounds: elow={self.elow}, ehigh={self.ehigh} "
                "(expected 0 <= elow < ehigh)"
            )
        seen_names: set[str] = set()
        for reg in self.regions:
            if reg.name in seen_names:
                raise ValueError(
                    f"Duplicate spectrum region name {reg.name!r} in [esas.spectrum]"
                )
            seen_names.add(reg.name)


@dataclass
class BandSpec:
    """Target energy band for QPB-subtracted flux imaging (energies in eV)."""

    name: str
    emin: int
    emax: int
    subbands: list[list[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.emin < 0 or self.emin >= self.emax:
            raise ValueError(
                f"Invalid band {self.name!r} bounds: emin={self.emin}, emax={self.emax} "
                "(expected 0 <= emin < emax)"
            )
        for pair in self.subbands:
            if len(pair) != 2:
                raise ValueError(
                    f"Invalid subband {pair!r} in band {self.name!r}; expected [lo, hi]"
                )
            lo, hi = pair
            if not (lo < hi and self.emin <= lo and hi <= self.emax):
                raise ValueError(
                    f"Invalid subband [{lo}, {hi}] in band {self.name!r} "
                    f"(must satisfy {self.emin} <= lo < hi <= {self.emax})"
                )


def default_qpb_bands() -> dict[str, BandSpec]:
    return {
        "soft": BandSpec(
            name="soft",
            emin=500,
            emax=2000,
            subbands=[[500, 1000], [1000, 2000]],
        ),
        "hard": BandSpec(
            name="hard",
            emin=2000,
            emax=7000,
            subbands=[[2000, 4500], [4500, 7000]],
        ),
    }


@dataclass
class QPBImageConfig:
    """Parameters for ``evqpb`` + ``eimageget`` + ``eimagecombine`` QPB-subtracted imaging."""

    dir: str = "qpb_image"
    bin_size: int = 50
    ecut: float = 1000.0
    epn_weight: float = 1.0
    em1_weight: float = 0.4
    em2_weight: float = 0.4
    smoothstyle: str = "simple"
    convolverstyle: str = "gaussian"
    width: float = 5.0
    minwidth: float = 0.0
    maxwidth: float = 10.0
    desiredsnr: float = 10.0
    nconvolvers: int = 20
    attfile: str = ""
    gti_files: dict[str, str] = field(default_factory=dict)
    default_subband_edges: list[int] = field(
        default_factory=lambda: [350, 500, 750, 1000, 1500, 2000, 3000, 4500, 7000, 10000]
    )
    bands: dict[str, BandSpec] = field(default_factory=default_qpb_bands)
    evqpb_extra_params: dict[str, Any] = field(default_factory=dict)
    eimageget_extra_params: dict[str, Any] = field(default_factory=dict)
    eimagecombine_extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnaConfig:
    """Top-level ``xmm-ana`` pipeline configuration."""

    env: SASEnv = field(default_factory=SASEnv)
    paths: PathsConfig = field(default_factory=PathsConfig)
    instruments: InstrumentsConfig = field(default_factory=InstrumentsConfig)
    qpb_image: QPBImageConfig = field(default_factory=QPBImageConfig)
    flare: FlareConfig = field(default_factory=FlareConfig)
    cheese: CheeseConfig = field(default_factory=CheeseConfig)
    spectrum: SpectrumConfig = field(default_factory=SpectrumConfig)
    espfilt: InitVar[FlareConfig | None] = None

    def __post_init__(self, espfilt: FlareConfig | None) -> None:
        if espfilt is not None:
            self.flare = espfilt
        if self.paths.qpb_image_dir == "qpb_image" and self.qpb_image.dir != "qpb_image":
            self.paths.qpb_image_dir = self.qpb_image.dir
        elif self.qpb_image.dir == "qpb_image" and self.paths.qpb_image_dir != "qpb_image":
            self.qpb_image.dir = self.paths.qpb_image_dir

    @classmethod
    def from_toml(cls, path: str | Path) -> AnaConfig:
        """Load and validate an ``AnaConfig`` from a TOML file."""
        from xmm_tools.config.loader import load_ana_config

        return load_ana_config(path)


def _get_espfilt(self: AnaConfig) -> FlareConfig:
    return self.flare


def _set_espfilt(self: AnaConfig, value: FlareConfig) -> None:
    self.flare = value


AnaConfig.espfilt = property(  # type: ignore[method-assign]
    _get_espfilt,
    _set_espfilt,
    doc="Backward-compatible alias for :attr:`flare`.",
)
