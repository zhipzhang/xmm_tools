"""TOML configuration schema and loader for ``xmm-ana`` workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]

from xmm_tools.exec.env import SASEnv


@dataclass
class PathsConfig:
    """Relative/absolute subdirectories for analysis outputs."""

    workdir: Path = Path(".")
    flare_dir: str = "soft_proton_flare"
    cheese_dir: str = "cheese"
    qpb_image_dir: str = "qpb_image"
    spectrum_dir: str = "spectrum"

    def resolve(self, sub: str | Path) -> Path:
        p = Path(sub)
        if p.is_absolute():
            return p
        return (self.workdir / p).resolve() if not self.workdir.is_absolute() else self.workdir / p


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


VALID_REGION_MODES = ("single", "split", "union")


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


@dataclass
class BandSpec:
    """Target energy band for QPB-subtracted flux imaging (energies in eV)."""

    name: str
    emin: int
    emax: int
    subbands: list[list[int]] = field(default_factory=list)


def _default_qpb_bands() -> dict[str, BandSpec]:
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
    with_asmooth: bool = False
    attfile: str = ""
    gti_files: dict[str, str] = field(default_factory=dict)
    default_subband_edges: list[int] = field(
        default_factory=lambda: [350, 500, 750, 1000, 1500, 2000, 3000, 4500, 7000, 10000]
    )
    bands: dict[str, BandSpec] = field(default_factory=_default_qpb_bands)
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
    espfilt: EspfiltConfig = field(default_factory=EspfiltConfig)
    cheese: CheeseConfig = field(default_factory=CheeseConfig)
    spectrum: SpectrumConfig = field(default_factory=SpectrumConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> AnaConfig:
        """Load an ``AnaConfig`` from a TOML file."""
        toml_path = Path(path)
        raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))

        paths_sec = _table(raw, "paths")
        workdir_raw = paths_sec.get("workdir", ".")
        workdir = Path(workdir_raw)
        if not workdir.is_absolute():
            workdir = (toml_path.parent / workdir).resolve()

        env = SASEnv.from_toml(toml_path, base_dir=workdir)

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
        raw_regions = spec_sec.get("regions", [])
        parsed_regions: list[SpectrumRegionSpec] = []
        for idx, item in enumerate(raw_regions):
            if isinstance(item, str):
                parsed_regions.append(
                    SpectrumRegionSpec(name=Path(item).stem or f"reg_{idx + 1}", file=item)
                )
            elif isinstance(item, dict):
                if "file" not in item:
                    raise ValueError(
                        f"Missing required 'file' key in [[esas.spectrum.regions]] entry #{idx + 1}"
                    )
                file_val = str(item["file"])
                name_val = str(item.get("name") or Path(file_val).stem)
                parsed_regions.append(
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
                raise ValueError(
                    "Expected string or table entry in [esas.spectrum].regions"
                )

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
            regions=parsed_regions,
            spectra_extra_params=dict(spec_sec.get("spectra_extra_params", {})),
            back_extra_params=dict(spec_sec.get("back_extra_params", {})),
        )

        qi_bands_raw = qi_sec.get("bands")
        if isinstance(qi_bands_raw, dict) and qi_bands_raw:
            qi_bands: dict[str, BandSpec] = {}
            for bname, bval in qi_bands_raw.items():
                if not isinstance(bval, dict):
                    raise ValueError(f"Expected [sas.qpb_image.bands.{bname}] table")
                if "emin" not in bval or "emax" not in bval:
                    raise ValueError(
                        f"Missing 'emin' or 'emax' in [sas.qpb_image.bands.{bname}]"
                    )
                qi_bands[bname] = BandSpec(
                    name=bname,
                    emin=int(bval["emin"]),
                    emax=int(bval["emax"]),
                    subbands=[
                        [int(a), int(b)] for a, b in bval.get("subbands", [])
                    ],
                )
        else:
            qi_bands = _default_qpb_bands()

        qi_gti_raw = qi_sec.get("gti_files", {})
        if not isinstance(qi_gti_raw, dict):
            raise ValueError("Expected [sas.qpb_image.gti_files] table in TOML config")
        qi_gti_files = {
            str(k): str(v)
            for k, v in qi_gti_raw.items()
            if v is not None and str(v).strip()
        }

        qpb_image = QPBImageConfig(
            dir=paths.qpb_image_dir,
            bin_size=int(qi_sec.get("bin_size", 50)),
            with_asmooth=bool(qi_sec.get("with_asmooth", False)),
            attfile=str(qi_sec.get("attfile", "")),
            gti_files=qi_gti_files,
            default_subband_edges=[
                int(x)
                for x in qi_sec.get(
                    "default_subband_edges",
                    [350, 500, 750, 1000, 1500, 2000, 3000, 4500, 7000, 10000],
                )
            ],
            bands=qi_bands,
            evqpb_extra_params=dict(qi_sec.get("evqpb_extra_params", {})),
            eimageget_extra_params=dict(qi_sec.get("eimageget_extra_params", {})),
            eimagecombine_extra_params=dict(
                qi_sec.get("eimagecombine_extra_params", {})
            ),
        )

        return cls(
            env=env,
            paths=paths,
            instruments=instruments,
            qpb_image=qpb_image,
            espfilt=espfilt,
            cheese=cheese,
            spectrum=spectrum,
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
    """Render a commented ``ana.toml`` configuration file."""
    return f"""\
# XMM-Newton analysis configuration (SAS + ESAS)

[env]
sas_ccf = "{sas_ccf}"                 # Calibration index file (generated by cifbuild)
sas_ccfpath = "{sas_ccfpath}"        # Path to XMM-Newton CCF calibration repository
sas_odf = "{sas_odf}"                 # ODF summary file (*SUM.SAS, generated by odfingest)
regions_path = ""                    # Optional fallback path containing 'regions' package

[paths]
workdir = "{workdir}"                 # Root analysis directory (relative to this TOML or absolute)
flare_dir = "soft_proton_flare"      # Subdirectory for espfilt + FLAG filtering
cheese_dir = "cheese"                # Subdirectory for point-source detection and masks
qpb_image_dir = "qpb_image"          # Subdirectory for QPB-subtracted flux images
spectrum_dir = "spectrum"            # Subdirectory for extracted spectra and particle backgrounds

[instruments]
mos1 = "{mos1}"               # Standardized MOS1 event file (leave empty "" to disable)
mos2 = "{mos2}"               # Standardized MOS2 event file
pn = "{pn}"                   # Standardized PN event file
pn_oot = "{pn_oot}"           # Standardized PN Out-of-Time event file

# ==============================================================================
# SAS Standard Tasks
# ==============================================================================

[sas.qpb_image]
bin_size = 50                        # Spatial pixel bin size (50 physical units = 2.5 arcsec/pix)
with_asmooth = false                 # Run eimagecombine asmooth (false = keep unsmoothed sub/exp maps)
attfile = ""                         # Attitude FITS file (empty = reuse atthk.fits or run atthkgen)
default_subband_edges = [350, 500, 750, 1000, 1500, 2000, 3000, 4500, 7000, 10000]
evqpb_extra_params = {{}}              # Extra key=value parameters passed through to evqpb
eimageget_extra_params = {{}}          # Extra key=value parameters passed through to eimageget
eimagecombine_extra_params = {{}}      # Extra key=value parameters passed through to eimagecombine

# Optional per-instrument GTI overrides (relative to workdir or absolute).
# When omitted, GTI files are loaded automatically from soft_proton_flare/manifest.json.
# [sas.qpb_image.gti_files]
# mos1 = "soft_proton_flare/mos1S001-gti.fits"
# mos2 = "soft_proton_flare/mos2S002-gti.fits"
# pn = "soft_proton_flare/pnS003-gti.fits"

[sas.qpb_image.bands.soft]
emin = 500                           # Target soft-band lower bound (eV)
emax = 2000                          # Target soft-band upper bound (eV)
subbands = [[500, 1000], [1000, 2000]]

[sas.qpb_image.bands.hard]
emin = 2000                          # Target hard-band lower bound (eV)
emax = 7000                          # Target hard-band upper bound (eV)
subbands = [[2000, 4500], [4500, 7000]]

# ==============================================================================
# ESAS Standard Tasks
# ==============================================================================

[esas.espfilt]
elow = 2500                          # Lower energy bound for flare light-curve/histogram (eV)
ehigh = 8500                         # Upper energy bound for flare light-curve/histogram (eV)
withsmoothing = true                 # Smooth light curves before histogram fitting
smooth = 51                          # Smoothing boxcar width (seconds)
rangescale_mos = 6.0                 # Histogram range scale factor for MOS1/MOS2
rangescale_pn = 15.0                 # Histogram range scale factor for PN
allowsigma = 3.0                     # Good-time acceptance half-width (in Gaussian sigma)
method = "histogram"                 # espfilt filtering method ("histogram" or "ratio")
keepinterfiles = false               # Keep SAS temporary files inside espfilt
apply_flag_filter = true             # Run ESAS evselect FLAG filter on *-allevc.fits
mos_flag_expr = "((FLAG & 0x766aa000)==0)"
pn_flag_expr = "((FLAG & 0x762a097c)==0)"

[esas.cheese]
elowlist = [350, 2500]               # Soft/hard lower energy bounds (eV)
ehighlist = [1100, 8000]             # Soft/hard upper energy bounds (eV)
scale = 0.4                          # PSF-scaled mask radius factor (larger = bigger holes)
mlmin = 15.0                         # emldetect minimum detection likelihood threshold
dist = 50.0                          # Minimum separation between detected sources (arcsec)
ratesoft = 0.2                       # Soft-band flux/rate threshold for masking
ratehard = 0.2                       # Hard-band flux/rate threshold for masking
ratetotal = 0.2                      # Total-band flux/rate threshold for masking
keepinterfiles = false               # Let SAS clean up its own temporary files
extra_params = {{}}                    # Extra key=value parameters passed through to cheese

[esas.spectrum]
max_workers = 2                      # Concurrent workers for multi-region extraction
elow = 400                           # Default spectrum lower energy bound (eV)
ehigh = 12500                        # Default spectrum upper energy bound (eV)
pattern_mos = 12                     # Event pattern threshold for MOS1/MOS2
pattern_pn = 4                       # Event pattern threshold for EPIC-pn
withsrcrem = true                    # Apply cheese point-source masks
keepinterfiles = true                # Keep intermediate spectra/plot products
withplotfiles = true                 # Generate QDP diagnostic plots in mosback/pnback
mos1_ccds = "T T F T T F T"          # Active MOS1 CCDs (CCD3 & CCD6 disabled)
mos2_ccds = "T T T T T T T"          # Active MOS2 CCDs
pn_quads = "T T T T"                 # Active EPIC-pn quadrants
spectra_extra_params = {{}}          # Extra key=value parameters for mosspectra/pnspectra
back_extra_params = {{}}             # Extra key=value parameters for mosback/pnback

# Add one [[esas.spectrum.regions]] block per target region or DS9 batch file:
# [[esas.spectrum.regions]]
# name = "east_jet"
# file = "regions/east_jet_bins.reg"
# mode = "split"                     # "single" | "split" | "union"
# overwrite = false                  # true to re-run and update spectrum/manifest.json
"""
