"""Astronomical region models, continuous strip-grid generator, and DS9 I/O."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal, Sequence

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
import regions as astropy_regions

from xmm_tools.image.fits_io import FitsImageData, sky_to_pixel


@dataclass
class CircleRegion:
    """Circular region in FK5 sky coordinates (radius in arcsec)."""

    name: str
    ra_deg: float
    dec_deg: float
    radius_arcsec: float
    include: bool = True


@dataclass
class BoxRegion:
    """Rotated rectangular region in FK5 sky coordinates (sizes in arcsec)."""

    name: str
    ra_deg: float
    dec_deg: float
    width_arcsec: float
    height_arcsec: float
    angle_deg: float = 0.0  # DS9 box angle (deg counter-clockwise from North/East axis)
    include: bool = True


@dataclass
class EllipseRegion:
    """Elliptical region in FK5 sky coordinates (semi-axes in arcsec)."""

    name: str
    ra_deg: float
    dec_deg: float
    semi_major_arcsec: float
    semi_minor_arcsec: float
    angle_deg: float = 0.0
    include: bool = True


@dataclass
class PolygonRegion:
    """Polygonal region in FK5 sky coordinates ``[(ra_deg, dec_deg), ...]``."""

    name: str
    vertices_deg: list[tuple[float, float]]
    include: bool = True


RegionItem = CircleRegion | BoxRegion | EllipseRegion | PolygonRegion


@dataclass
class StripGridParams:
    """Parameters for generating contiguous, equal-width rectangular strips."""

    start_ra_deg: float
    start_dec_deg: float
    pa_deg: float  # Position angle East of North (deg) along strip progression
    width_arcsec: float  # Width of each strip along the progression axis
    height_arcsec: float  # Length of each strip perpendicular to the progression axis
    count: int
    gap_arcsec: float = 0.0
    prefix: str = "bin"


def generate_strip_grid(params: StripGridParams) -> list[BoxRegion]:
    """Generate ``params.count`` contiguous equal-width ``BoxRegion`` strips."""
    if params.count <= 0:
        return []

    origin = SkyCoord(
        ra=params.start_ra_deg * u.deg,
        dec=params.start_dec_deg * u.deg,
        frame="fk5",
    )
    step_arcsec = params.width_arcsec + params.gap_arcsec
    pa = params.pa_deg * u.deg
    # DS9 box(ra, dec, width, height, angle): angle=0 aligns width with East-West;
    # rotating by (90 - pa) aligns `width` along `pa` (East of North).
    ds9_angle = (90.0 - params.pa_deg) % 360.0

    boxes: list[BoxRegion] = []
    for idx in range(params.count):
        offset = (idx + 0.5) * step_arcsec * u.arcsec
        center = origin.directional_offset_by(pa, offset)
        boxes.append(
            BoxRegion(
                name=f"{params.prefix}{idx + 1}",
                ra_deg=float(center.ra.deg),
                dec_deg=float(center.dec.deg),
                width_arcsec=params.width_arcsec,
                height_arcsec=params.height_arcsec,
                angle_deg=ds9_angle,
                include=True,
            )
        )
    return boxes


def strip_params_from_sky_points(
    start_ra_deg: float,
    start_dec_deg: float,
    end_ra_deg: float,
    end_dec_deg: float,
    *,
    width_arcsec: float,
    height_arcsec: float,
    gap_arcsec: float = 0.0,
    prefix: str = "bin",
) -> StripGridParams:
    """Compute ``StripGridParams`` from user-picked start and end sky points."""
    p1 = SkyCoord(ra=start_ra_deg * u.deg, dec=start_dec_deg * u.deg, frame="fk5")
    p2 = SkyCoord(ra=end_ra_deg * u.deg, dec=end_dec_deg * u.deg, frame="fk5")
    sep_arcsec = float(p1.separation(p2).arcsec)
    pa_deg = float(p1.position_angle(p2).deg)
    step = max(width_arcsec + gap_arcsec, 1e-3)
    count = max(1, int(round(sep_arcsec / step)))
    return StripGridParams(
        start_ra_deg=start_ra_deg,
        start_dec_deg=start_dec_deg,
        pa_deg=pa_deg,
        width_arcsec=width_arcsec,
        height_arcsec=height_arcsec,
        count=count,
        gap_arcsec=gap_arcsec,
        prefix=prefix,
    )


def box_corners_pixels(
    box: BoxRegion, img: FitsImageData
) -> list[tuple[float, float]] | None:
    """Compute the 4 corner coordinates in 0-based pixels for ImPlot overlay."""
    center_px = sky_to_pixel(img, box.ra_deg, box.dec_deg)
    if center_px is None:
        return None
    cx, cy = center_px
    scale = max(img.pixel_scale_arcsec, 1e-6)
    hw = 0.5 * box.width_arcsec / scale
    hh = 0.5 * box.height_arcsec / scale

    # In standard north-up image, East is -x (left) and North is +y (up),
    # so angle_deg counter-clockwise from +x corresponds to -angle_deg or +angle_deg
    # depending on parity; compute sky corners directly via WCS for exactness.
    if img.wcs is not None:
        center = SkyCoord(ra=box.ra_deg * u.deg, dec=box.dec_deg * u.deg, frame="fk5")
        pa_width = (90.0 - box.angle_deg) * u.deg
        pa_height = (180.0 - box.angle_deg) * u.deg
        hw_arc = 0.5 * box.width_arcsec * u.arcsec
        hh_arc = 0.5 * box.height_arcsec * u.arcsec
        corners: list[tuple[float, float]] = []
        for s_w, s_h in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
            pt = center.directional_offset_by(pa_width, s_w * hw_arc)
            pt = pt.directional_offset_by(pa_height, s_h * hh_arc)
            px = sky_to_pixel(img, float(pt.ra.deg), float(pt.dec.deg))
            if px is None:
                return None
            corners.append(px)
        return corners

    rad = math.radians(box.angle_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    return [
        (cx + sx * hw * ca - sy * hh * sa, cy + sx * hw * sa + sy * hh * ca)
        for sx, sy in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    ]


def resize_box_edge(
    box: BoxRegion,
    drag_ra_deg: float,
    drag_dec_deg: float,
    *,
    edge: Literal["top", "bottom"],
    min_height_arcsec: float = 2.0,
) -> BoxRegion:
    """Single-edge height stretch: keep the opposite edge fixed and update center & height."""
    center = SkyCoord(ra=box.ra_deg * u.deg, dec=box.dec_deg * u.deg, frame="fk5")
    pa_top_deg = (180.0 - box.angle_deg) % 360.0
    if edge == "top":
        anchor_dir = (pa_top_deg + 180.0) % 360.0
        grow_dir = pa_top_deg
    else:
        anchor_dir = pa_top_deg
        grow_dir = (pa_top_deg + 180.0) % 360.0

    anchor = center.directional_offset_by(
        anchor_dir * u.deg, 0.5 * box.height_arcsec * u.arcsec
    )
    drag_pt = SkyCoord(ra=drag_ra_deg * u.deg, dec=drag_dec_deg * u.deg, frame="fk5")
    sep_arc = float(anchor.separation(drag_pt).arcsec)
    pa_drag = float(anchor.position_angle(drag_pt).deg)
    proj_arc = sep_arc * math.cos(math.radians(pa_drag - grow_dir))
    new_height = max(proj_arc, min_height_arcsec)

    new_center = anchor.directional_offset_by(
        grow_dir * u.deg, 0.5 * new_height * u.arcsec
    )
    box.ra_deg = float(new_center.ra.deg)
    box.dec_deg = float(new_center.dec.deg)
    box.height_arcsec = float(new_height)
    return box


def ellipse_outline_pixels(
    ellipse: EllipseRegion, img: FitsImageData, *, n_pts: int = 48
) -> list[tuple[float, float]] | None:
    """Sample an ``EllipseRegion`` boundary in 0-based pixel coordinates."""
    if img.wcs is None:
        return None
    center = SkyCoord(
        ra=ellipse.ra_deg * u.deg, dec=ellipse.dec_deg * u.deg, frame="fk5"
    )
    pa_major = (90.0 - ellipse.angle_deg) * u.deg
    pa_minor = (180.0 - ellipse.angle_deg) * u.deg
    pts: list[tuple[float, float]] = []
    for i in range(n_pts):
        theta = 2.0 * math.pi * i / n_pts
        pt = center.directional_offset_by(
            pa_major, ellipse.semi_major_arcsec * math.cos(theta) * u.arcsec
        )
        pt = pt.directional_offset_by(
            pa_minor, ellipse.semi_minor_arcsec * math.sin(theta) * u.arcsec
        )
        px = sky_to_pixel(img, float(pt.ra.deg), float(pt.dec.deg))
        if px is None:
            return None
        pts.append(px)
    return pts


def polygon_pixels(
    poly: PolygonRegion, img: FitsImageData
) -> list[tuple[float, float]] | None:
    """Convert a ``PolygonRegion``'s sky vertices to 0-based pixel coordinates."""
    if img.wcs is None:
        return None
    pts: list[tuple[float, float]] = []
    for ra, dec in poly.vertices_deg:
        px = sky_to_pixel(img, ra, dec)
        if px is None:
            return None
        pts.append(px)
    return pts


def export_ds9_regions(
    regions: Sequence[RegionItem],
    path: str | Path,
) -> Path:
    """Export regions to a standard DS9 FK5 ``.reg`` file."""
    out_path = Path(path)
    lines = [
        "# Region file format: DS9 version 4.1",
        'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" '
        "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1",
        "fk5",
    ]
    for reg in regions:
        prefix = "" if reg.include else "-"
        if isinstance(reg, CircleRegion):
            lines.append(
                f'{prefix}circle({reg.ra_deg:.7f},{reg.dec_deg:.7f},{reg.radius_arcsec:.3f}") '
                f"# text={{{reg.name}}}"
            )
        elif isinstance(reg, BoxRegion):
            lines.append(
                f'{prefix}box({reg.ra_deg:.7f},{reg.dec_deg:.7f},'
                f'{reg.width_arcsec:.3f}",{reg.height_arcsec:.3f}",{reg.angle_deg:.3f}) '
                f"# text={{{reg.name}}}"
            )
        elif isinstance(reg, EllipseRegion):
            lines.append(
                f'{prefix}ellipse({reg.ra_deg:.7f},{reg.dec_deg:.7f},'
                f'{reg.semi_major_arcsec:.3f}",{reg.semi_minor_arcsec:.3f}",{reg.angle_deg:.3f}) '
                f"# text={{{reg.name}}}"
            )
        elif isinstance(reg, PolygonRegion):
            coords = ",".join(
                f"{ra:.7f},{dec:.7f}" for ra, dec in reg.vertices_deg
            )
            lines.append(f"{prefix}polygon({coords}) # text={{{reg.name}}}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def parse_ds9_regions(
    path: str | Path, *, wcs: WCS | None = None
) -> list[RegionItem]:
    """Parse circle, box, ellipse, and polygon regions from a DS9 ``.reg`` file."""
    parsed = astropy_regions.Regions.read(Path(path), format="ds9")
    items: list[RegionItem] = []

    for idx, reg in enumerate(parsed, 1):
        if isinstance(reg, astropy_regions.PixelRegion):
            if wcs is None:
                continue
            reg = reg.to_sky(wcs)

        include = bool(reg.meta.get("include", 1))
        name = str(reg.meta.get("text") or f"reg{idx}")

        if isinstance(reg, astropy_regions.CircleSkyRegion):
            fk5 = reg.center.fk5
            items.append(
                CircleRegion(
                    name=name,
                    ra_deg=float(fk5.ra.deg),
                    dec_deg=float(fk5.dec.deg),
                    radius_arcsec=float(reg.radius.to_value(u.arcsec)),
                    include=include,
                )
            )
        elif isinstance(reg, astropy_regions.RectangleSkyRegion):
            fk5 = reg.center.fk5
            items.append(
                BoxRegion(
                    name=name,
                    ra_deg=float(fk5.ra.deg),
                    dec_deg=float(fk5.dec.deg),
                    width_arcsec=float(reg.width.to_value(u.arcsec)),
                    height_arcsec=float(reg.height.to_value(u.arcsec)),
                    angle_deg=float(reg.angle.to_value(u.deg)),
                    include=include,
                )
            )
        elif isinstance(reg, astropy_regions.EllipseSkyRegion):
            fk5 = reg.center.fk5
            items.append(
                EllipseRegion(
                    name=name,
                    ra_deg=float(fk5.ra.deg),
                    dec_deg=float(fk5.dec.deg),
                    semi_major_arcsec=0.5 * float(reg.width.to_value(u.arcsec)),
                    semi_minor_arcsec=0.5 * float(reg.height.to_value(u.arcsec)),
                    angle_deg=float(reg.angle.to_value(u.deg)),
                    include=include,
                )
            )
        elif isinstance(reg, astropy_regions.PolygonSkyRegion):
            fk5 = reg.vertices.fk5
            verts = [
                (float(ra), float(dec))
                for ra, dec in zip(fk5.ra.deg, fk5.dec.deg)
            ]
            items.append(
                PolygonRegion(name=name, vertices_deg=verts, include=include)
            )

    return items
