"""Region specification expansion and DS9 region import for ESAS spectra."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xmm_tools.config import AnaConfig, SpectrumRegionSpec
from xmm_tools.exec import normalize_lexical_path


@dataclass
class ExpandedRegionJob:
    """Concrete region extraction job derived from a ``SpectrumRegionSpec``."""

    key: str
    group: str
    label: str
    mode: str
    region_dir: Path
    source_file: Path
    overwrite: bool
    elow: int
    ehigh: int
    sky_regions: list[Any] | None = None
    det_expr_text: str | None = None


def sanitize_label(raw_label: str, fallback: str) -> str:
    """Sanitize a DS9 region ``text`` label for filesystem path usage."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_label.strip()).strip("_")
    return cleaned or fallback


def _import_regions(regions_path: Path | str | None = None) -> tuple[Any, ...]:
    """Import ``astropy-regions``, falling back to ``regions_path`` if needed."""

    def _load() -> tuple[Any, ...]:
        from regions import (
            CircleSkyRegion,
            EllipseSkyRegion,
            PolygonSkyRegion,
            RectangleSkyRegion,
            Regions,
        )

        return (
            Regions,
            CircleSkyRegion,
            EllipseSkyRegion,
            RectangleSkyRegion,
            PolygonSkyRegion,
        )

    try:
        return _load()
    except ImportError as exc:
        fallback = regions_path or os.environ.get("XMM_REGIONS_PATH")
        if fallback and str(fallback).strip():
            rpath = str(Path(str(fallback).strip()))
            if rpath not in sys.path:
                sys.path.insert(0, rpath)
            try:
                return _load()
            except ImportError as inner_exc:
                raise ImportError(
                    f"Could not import 'regions' even after adding regions_path={rpath!r} "
                    "to sys.path. Install 'regions' or verify [env].regions_path in ana.toml."
                ) from inner_exc
        raise ImportError(
            "DS9 sky region conversion requires 'regions'. "
            "Install 'regions' or set [env].regions_path in ana.toml."
        ) from exc


def expand_region_spec(
    spec: SpectrumRegionSpec,
    cfg: AnaConfig,
    spectrum_dir: Path,
) -> list[ExpandedRegionJob]:
    """Expand one ``SpectrumRegionSpec`` into one or more ``ExpandedRegionJob`` items."""
    src_path = normalize_lexical_path(cfg.paths.resolve(spec.file))
    if not src_path.is_file():
        raise FileNotFoundError(
            f"Region file not found for {spec.name!r}: {src_path}"
        )

    elow = spec.elow if spec.elow is not None else cfg.spectrum.elow
    ehigh = spec.ehigh if spec.ehigh is not None else cfg.spectrum.ehigh

    if src_path.suffix.lower() == ".txt":
        expr = src_path.read_text(encoding="utf-8").strip()
        if not expr:
            raise ValueError(f"Empty SAS detector expression file: {src_path}")
        return [
            ExpandedRegionJob(
                key=spec.name,
                group=spec.name,
                label=spec.name,
                mode=spec.mode,
                region_dir=spectrum_dir / spec.name,
                source_file=src_path,
                overwrite=spec.overwrite,
                elow=elow,
                ehigh=ehigh,
                det_expr_text=expr,
            )
        ]

    Regions, *_ = _import_regions(cfg.env.regions_path)
    parsed = list(Regions.read(str(src_path), format="ds9"))
    if not parsed:
        raise ValueError(f"No DS9 regions parsed from {src_path}")

    if spec.mode == "single":
        if len(parsed) > 1:
            raise ValueError(
                f"Region {spec.name!r} ({src_path.name}) contains {len(parsed)} shapes, "
                "but mode='single'. Use mode='union' to merge them or mode='split' to batch."
            )
        return [
            ExpandedRegionJob(
                key=spec.name,
                group=spec.name,
                label=spec.name,
                mode="single",
                region_dir=spectrum_dir / spec.name,
                source_file=src_path,
                overwrite=spec.overwrite,
                elow=elow,
                ehigh=ehigh,
                sky_regions=parsed,
            )
        ]

    if spec.mode == "union":
        return [
            ExpandedRegionJob(
                key=spec.name,
                group=spec.name,
                label=spec.name,
                mode="union",
                region_dir=spectrum_dir / spec.name,
                source_file=src_path,
                overwrite=spec.overwrite,
                elow=elow,
                ehigh=ehigh,
                sky_regions=parsed,
            )
        ]

    jobs: list[ExpandedRegionJob] = []
    seen_labels: set[str] = set()
    for idx, reg in enumerate(parsed, start=1):
        meta_text = str(reg.meta.get("text", "") or "") if hasattr(reg, "meta") else ""
        label = sanitize_label(meta_text, f"r{idx}")
        if label in seen_labels:
            label = f"{label}_{idx}"
        seen_labels.add(label)
        jobs.append(
            ExpandedRegionJob(
                key=f"{spec.name}/{label}",
                group=spec.name,
                label=label,
                mode="split",
                region_dir=spectrum_dir / spec.name / label,
                source_file=src_path,
                overwrite=spec.overwrite,
                elow=elow,
                ehigh=ehigh,
                sky_regions=[reg],
            )
        )
    return jobs


def expand_region_specs(
    specs: Sequence[SpectrumRegionSpec],
    cfg: AnaConfig,
    spectrum_dir: Path,
) -> list[ExpandedRegionJob]:
    """Expand a sequence of ``SpectrumRegionSpec`` objects into region jobs."""
    norm_sdir = normalize_lexical_path(spectrum_dir)
    jobs: list[ExpandedRegionJob] = []
    seen_keys: set[str] = set()
    seen_dirs: set[Path] = set()
    for spec in specs:
        for job in expand_region_spec(spec, cfg, norm_sdir):
            norm_rdir = normalize_lexical_path(job.region_dir)
            if norm_rdir == norm_sdir or not norm_rdir.is_relative_to(norm_sdir):
                raise ValueError(
                    f"Expanded region {job.key!r} directory {norm_rdir} must reside "
                    f"strictly inside spectrum_dir {norm_sdir}"
                )
            if job.key in seen_keys:
                raise ValueError(
                    f"Duplicate expanded spectrum region key {job.key!r}"
                )
            if norm_rdir in seen_dirs:
                raise ValueError(
                    f"Duplicate expanded spectrum region directory {norm_rdir} "
                    f"for key {job.key!r}"
                )
            seen_keys.add(job.key)
            seen_dirs.add(norm_rdir)
            jobs.append(job)
    return jobs
