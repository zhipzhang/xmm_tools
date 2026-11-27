"""Filtering and cross-epoch clustering for ``emllist.fits`` source catalogs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from collections.abc import Set as AbstractSet

from xmm_tools.srclist.models import (
    EpochCatalog,
    FilterParams,
    MergedSource,
    SourceDetection,
)

_DEFAULT_BG_MAP = 0.1
_DEFAULT_SCTS = 1.0


def angular_separation_arcsec(
    ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float
) -> float:
    """Exact spherical angular separation in arcseconds (haversine formula)."""
    ra1 = math.radians(ra1_deg)
    dec1 = math.radians(dec1_deg)
    ra2 = math.radians(ra2_deg)
    dec2 = math.radians(dec2_deg)
    d_ra = ra2 - ra1
    d_dec = dec2 - dec1
    a = (
        math.sin(0.5 * d_dec) ** 2
        + math.cos(dec1) * math.cos(dec2) * math.sin(0.5 * d_ra) ** 2
    )
    a = min(1.0, max(0.0, a))
    sep_rad = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return math.degrees(sep_rad) * 3600.0


def filter_detections(
    detections: Sequence[SourceDetection], params: FilterParams
) -> list[SourceDetection]:
    """Filter individual detections against ML, flux, separation, and extent rules."""
    kept: list[SourceDetection] = []
    for det in detections:
        if det.det_ml < params.min_det_ml:
            continue
        if det.flux < params.min_flux:
            continue
        if det.dist_nn_arcsec < params.min_dist_nn_arcsec:
            continue
        if params.exclude_extended and det.ext > 0.0 and det.ext_ml >= 6.0:
            continue
        kept.append(det)
    return kept


def _weighted_sky_position(
    members: Sequence[SourceDetection],
) -> tuple[float, float, float]:
    """Compute inverse-variance weighted ``(ra_deg, dec_deg, radec_err_arcsec)``."""
    if len(members) == 1:
        m = members[0]
        err = m.radec_err_arcsec if m.radec_err_arcsec > 0.0 else 0.5
        return m.ra_deg, m.dec_deg, err

    wx = 0.0
    wy = 0.0
    wz = 0.0
    w_sum = 0.0
    for m in members:
        sigma = max(m.radec_err_arcsec, 0.2)
        w = 1.0 / (sigma * sigma)
        ra_r = math.radians(m.ra_deg)
        dec_r = math.radians(m.dec_deg)
        cos_d = math.cos(dec_r)
        wx += w * cos_d * math.cos(ra_r)
        wy += w * cos_d * math.sin(ra_r)
        wz += w * math.sin(dec_r)
        w_sum += w

    norm = math.sqrt(wx * wx + wy * wy + wz * wz)
    if norm <= 0.0 or w_sum <= 0.0:
        m0 = members[0]
        return m0.ra_deg, m0.dec_deg, max(m0.radec_err_arcsec, 0.2)

    ra_deg = math.degrees(math.atan2(wy, wx)) % 360.0
    dec_deg = math.degrees(math.asin(max(-1.0, min(1.0, wz / norm))))
    combined_err = 1.0 / math.sqrt(w_sum)
    return ra_deg, dec_deg, combined_err


def _build_merged_source(
    merged_id: int, members: list[SourceDetection]
) -> MergedSource:
    ra_deg, dec_deg, radec_err = _weighted_sky_position(members)
    best_det = max(members, key=lambda m: (m.det_ml, m.scts))
    pos_both = [m for m in members if m.scts > 0.0 and m.bg_map > 0.0]
    if pos_both:
        best_snr = max(
            pos_both,
            key=lambda m: (m.scts / max(m.bg_map, 1e-4), m.det_ml),
        )
        scts = best_snr.scts
        bg_map = best_snr.bg_map
        rep_row = dict(best_snr.raw_row)
    else:
        pos_bg = [m for m in members if m.bg_map > 0.0]
        if pos_bg:
            best_bg = max(pos_bg, key=lambda m: (m.det_ml, m.scts))
            bg_map = best_bg.bg_map
            rep_row = dict(best_bg.raw_row)
        else:
            bg_map = _DEFAULT_BG_MAP
            rep_row = dict(best_det.raw_row)

        pos_scts = [m.scts for m in members if m.scts > 0.0]
        if pos_scts:
            scts = max(pos_scts)
        else:
            derived = [
                m.rate * m.exp_map
                for m in members
                if m.rate > 0.0 and m.exp_map > 0.0
            ]
            scts = max(derived) if derived else _DEFAULT_SCTS

    return MergedSource(
        merged_id=merged_id,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        radec_err_arcsec=radec_err,
        det_ml=max(m.det_ml for m in members),
        flux=max(m.flux for m in members),
        scts=scts,
        bg_map=bg_map,
        ext=best_det.ext,
        ext_ml=best_det.ext_ml,
        dist_nn_arcsec=best_det.dist_nn_arcsec,
        include=True,
        is_manual=False,
        members=members,
        representative_row=rep_row,
    )


def merge_catalogs(
    catalogs: Sequence[EpochCatalog],
    *,
    filter_params: FilterParams,
    match_radius_arcsec: float = 5.0,
    excluded_detection_keys: AbstractSet[tuple[str, int]] = frozenset(),
    excluded_sky_coords: Sequence[tuple[float, float]] = (),
    manual_sources: Sequence[MergedSource] = (),
) -> list[MergedSource]:
    """Filter enabled epochs, cluster detections without chaining, and merge."""
    filtered: list[SourceDetection] = []
    for cat in catalogs:
        if not cat.enabled:
            continue
        filtered.extend(filter_detections(cat.detections, filter_params))

    # Sort deterministically by DET_ML descending so brightest seeds define clusters.
    sorted_dets = sorted(
        filtered,
        key=lambda d: (-d.det_ml, d.epoch_id, d.ml_id_src),
    )

    clusters: list[tuple[list[SourceDetection], float, float]] = []
    if match_radius_arcsec <= 0.0:
        clusters = [([det], det.ra_deg, det.dec_deg) for det in sorted_dets]
    else:
        for det in sorted_dets:
            best_idx: int | None = None
            best_sep = match_radius_arcsec
            for idx, (_, seed_ra, seed_dec) in enumerate(clusters):
                sep = angular_separation_arcsec(
                    det.ra_deg, det.dec_deg, seed_ra, seed_dec
                )
                if sep <= best_sep:
                    best_sep = sep
                    best_idx = idx
            if best_idx is None:
                clusters.append(([det], det.ra_deg, det.dec_deg))
            else:
                # Keep the initial seed position fixed to prevent chaining.
                clusters[best_idx][0].append(det)

    merged: list[MergedSource] = [
        _build_merged_source(idx, members)
        for idx, (members, _, _) in enumerate(clusters, start=1)
    ]

    for msrc in manual_sources:
        merged.append(
            MergedSource(
                merged_id=len(merged) + 1,
                ra_deg=msrc.ra_deg,
                dec_deg=msrc.dec_deg,
                radec_err_arcsec=msrc.radec_err_arcsec,
                det_ml=msrc.det_ml,
                flux=msrc.flux,
                scts=msrc.scts if msrc.scts > 0.0 else _DEFAULT_SCTS,
                bg_map=msrc.bg_map if msrc.bg_map > 0.0 else _DEFAULT_BG_MAP,
                ext=msrc.ext,
                ext_ml=msrc.ext_ml,
                dist_nn_arcsec=msrc.dist_nn_arcsec,
                include=msrc.include,
                is_manual=True,
                manual_id=msrc.manual_id,
                members=list(msrc.members),
                representative_row=dict(msrc.representative_row),
            )
        )

    # Recompute nearest-neighbor separation across the merged catalog.
    n_tot = len(merged)
    if n_tot > 1:
        for i in range(n_tot):
            si = merged[i]
            min_sep = min(
                angular_separation_arcsec(
                    si.ra_deg, si.dec_deg, merged[j].ra_deg, merged[j].dec_deg
                )
                for j in range(n_tot)
                if j != i
            )
            si.dist_nn_arcsec = min_sep
    elif n_tot == 1:
        merged[0].dist_nn_arcsec = 999.0

    # Restore manual exclusions across live filter/radius updates.
    if excluded_detection_keys:
        for src in merged:
            if not src.is_manual and any(
                (m.epoch_id, m.ml_id_src) in excluded_detection_keys
                for m in src.members
            ):
                src.include = False

    if excluded_sky_coords:
        excl_tol = max(match_radius_arcsec, 0.5)
        for src in merged:
            if not src.include:
                continue
            for ex_ra, ex_dec in excluded_sky_coords:
                if (
                    angular_separation_arcsec(src.ra_deg, src.dec_deg, ex_ra, ex_dec)
                    <= excl_tol
                ):
                    src.include = False
                    break

    return merged
