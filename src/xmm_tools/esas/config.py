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
    """Target energy band for quick-look flux imaging (energies in eV)."""

    name: str
    emin: int
    emax: int
    subbands: list[list[int]] = field(default_factory=list)


def _default_quick_bands() -> dict[str, BandSpec]:
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
class QuickImageConfig:
    """Parameters for ``evqpb`` + ``eimageget`` + ``eimagecombine`` quick imaging."""

    dir: str = "quick_image"
    bin_size: int = 40
    with_vignetting: bool = True
    use_cheese_mask: bool = False
    combine_instruments: bool = True
    attfile: str = ""
    default_subband_edges: list[int] = field(
        default_factory=lambda: [350, 500, 750, 1000, 1500, 2000, 3000, 4500, 7000, 10000]
    )
    bands: dict[str, BandSpec] = field(default_factory=_default_quick_bands)
    evqpb_extra_params: dict[str, Any] = field(default_factory=dict)
    eimageget_extra_params: dict[str, Any] = field(default_factory=dict)
    eimagecombine_extra_params: dict[str, Any] = field(default_factory=dict)


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
    quick_image: QuickImageConfig = field(default_factory=QuickImageConfig)
    imaging: ImagingConfig = field(default_factory=ImagingConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> ESASConfig:
        """Load an ``ESASConfig`` from a TOML file."""
        toml_path = Path(path)
        raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))

        paths_sec = _table(raw, "paths")
        workdir_raw = paths_sec.get("workdir", ".")
        workdir = Path(workdir_raw)
        if not workdir.is_absolute():
            workdir = (toml_path.parent / workdir).resolve()

        env = SASEnv.from_toml(toml_path, base_dir=workdir)

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
            extra_params=dict(sd_sec.get("extra_params", {})),
        )

        spec_sec = _table(raw, "spectrum")
        raw_regions = spec_sec.get("regions", [])
        parsed_regions: list[SpectrumRegionSpec] = []
        for idx, item in enumerate(raw_regions):
            if isinstance(item, str):
                parsed_regions.append(
                    SpectrumRegionSpec(name=Path(item).stem or f"reg_{idx + 1}", file=item)
                )
            elif isinstance(item, dict):
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
                    "Expected string or table entry in [spectrum].regions"
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

        qi_sec = _table(raw, "quick_image")
        qi_bands_raw = qi_sec.get("bands")
        if isinstance(qi_bands_raw, dict) and qi_bands_raw:
            qi_bands: dict[str, BandSpec] = {}
            for bname, bval in qi_bands_raw.items():
                if not isinstance(bval, dict):
                    raise ValueError(f"Expected [quick_image.bands.{bname}] table")
                qi_bands[bname] = BandSpec(
                    name=bname,
                    emin=int(bval["emin"]),
                    emax=int(bval["emax"]),
                    subbands=[
                        [int(a), int(b)] for a, b in bval.get("subbands", [])
                    ],
                )
        else:
            qi_bands = _default_quick_bands()

        quick_image = QuickImageConfig(
            dir=str(qi_sec.get("dir", "quick_image")),
            bin_size=int(qi_sec.get("bin_size", 40)),
            with_vignetting=bool(qi_sec.get("with_vignetting", True)),
            use_cheese_mask=bool(qi_sec.get("use_cheese_mask", False)),
            combine_instruments=bool(qi_sec.get("combine_instruments", True)),
            attfile=str(qi_sec.get("attfile", "")),
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
            quick_image=quick_image,
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
sas_ccf = "{sas_ccf}"                 # Calibration index file (generated by cifbuild)
sas_ccfpath = "{sas_ccfpath}"        # Path to XMM-Newton CCF calibration repository
sas_odf = "{sas_odf}"                 # ODF summary file (*SUM.SAS, generated by odfingest)
regions_path = ""                    # Optional fallback path containing 'regions' package (e.g. inside Docker)

[paths]
workdir = "{workdir}"                 # Root analysis directory (relative to this TOML or absolute)
flare_dir = "soft_proton_flare"      # Subdirectory for espfilt + FLAG filtering
cheese_dir = "cheese"                # Subdirectory for point-source detection and masks
spectrum_dir = "spectrum"            # Subdirectory for extracted spectra and particle backgrounds
image_dir = "images"                 # Subdirectory for multi-band imaging products

[instruments]
mos1 = "{mos1}"               # Standardized MOS1 event file (leave empty "" to disable)
mos2 = "{mos2}"               # Standardized MOS2 event file
pn = "{pn}"                   # Standardized PN event file
pn_oot = "{pn_oot}"           # Standardized PN Out-of-Time event file

[espfilt]
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

[source_detect]
method = "cheese"                    # Detector backend ("cheese" by default)
elowlist = [350, 2500]               # Soft/hard lower energy bounds (eV)
ehighlist = [1100, 8000]             # Soft/hard upper energy bounds (eV)
scale = 0.4                          # PSF-scaled mask radius factor (larger = bigger holes)
mlmin = 15.0                         # emldetect minimum detection likelihood threshold
dist = 50.0                          # Minimum separation between detected sources (arcsec)
ratesoft = 0.2                       # Soft-band flux/rate threshold for masking
ratehard = 0.2                       # Hard-band flux/rate threshold for masking
ratetotal = 0.2                      # Total-band flux/rate threshold for masking
keepinterfiles = false               # Let SAS clean up its own temporary files
extra_regions = []                   # Optional DS9 .reg files to merge into exclusion masks
extra_params = {{}}                    # Extra key=value parameters passed through to cheese

[spectrum]
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

# Add one [[spectrum.regions]] block per target region or DS9 batch file:
# [[spectrum.regions]]
# name = "east_jet"
# file = "regions/east_jet_bins.reg"
# mode = "split"                     # "single" | "split" | "union"
# overwrite = false                  # true to re-run and update spectrum/manifest.json

[quick_image]
dir = "quick_image"                  # Output directory for quick-look QPB-subtracted flux images
bin_size = 40                        # Spatial pixel bin size (40 physical units ≈ 2 arcsec/pix)
with_vignetting = true               # Apply telescope vignetting correction in eimageget
use_cheese_mask = false              # Apply cheese point-source masks if available
combine_instruments = true           # Merge MOS1+MOS2+PN with eimagecombine per target band
attfile = ""                         # Attitude FITS file (empty = reuse atthk.fits or run atthkgen)
default_subband_edges = [350, 500, 750, 1000, 1500, 2000, 3000, 4500, 7000, 10000]
evqpb_extra_params = {{}}              # Extra key=value parameters passed through to evqpb
eimageget_extra_params = {{}}          # Extra key=value parameters passed through to eimageget
eimagecombine_extra_params = {{}}      # Extra key=value parameters passed through to eimagecombine

[quick_image.bands.soft]
emin = 500                           # Target soft-band lower bound (eV)
emax = 2000                          # Target soft-band upper bound (eV)
subbands = [[500, 1000], [1000, 2000]]  # Narrow sub-bands for accurate exposure weighting

[quick_image.bands.hard]
emin = 2000                          # Target hard-band lower bound (eV)
emax = 7000                          # Target hard-band upper bound (eV)
subbands = [[2000, 4500], [4500, 7000]]

[imaging]
bands = [[400, 1250], [2000, 7000]]  # Energy bands for imaging [elow, ehigh] (eV)
binning = 2                          # Spatial image binning factor
"""
