"""Data models and manifest serialization for ``xmm_tools.sas.qpb_image``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xmm_tools.exec import (
    MANIFEST_SCHEMA_VERSION,
    StageManifest,
    atomic_write_json,
    normalize_lexical_path,
    read_json,
    resolve_workspace_path,
    workspace_relative_path,
)


@dataclass
class SubbandProductPaths:
    """Combined ``eimagecombine`` outputs for a single energy sub-band."""

    index: int
    emin: int
    emax: int
    counts_image: Path
    exposure_image: Path
    smoothed_flux_image: Path | None = None

    def to_dict(
        self,
        base_dir: Path | str,
        *,
        workspace_root: Path | str | None = None,
    ) -> dict[str, Any]:
        """Serialize sub-band product paths relative to ``base_dir``."""
        data: dict[str, Any] = {
            "index": self.index,
            "emin": self.emin,
            "emax": self.emax,
            "counts_image": workspace_relative_path(
                self.counts_image, base_dir, workspace_root=workspace_root
            ),
            "exposure_image": workspace_relative_path(
                self.exposure_image, base_dir, workspace_root=workspace_root
            ),
        }
        if self.smoothed_flux_image is not None:
            data["smoothed_flux_image"] = workspace_relative_path(
                self.smoothed_flux_image, base_dir, workspace_root=workspace_root
            )
        return data

    @classmethod
    def from_dict(
        cls, base_dir: Path | str, raw: Mapping[str, Any]
    ) -> SubbandProductPaths:
        """Deserialize a ``SubbandProductPaths`` record relative to ``base_dir``."""
        smooth_raw = raw.get("smoothed_flux_image")
        return cls(
            index=int(raw["index"]),
            emin=int(raw["emin"]),
            emax=int(raw["emax"]),
            counts_image=resolve_workspace_path(base_dir, str(raw["counts_image"])),
            exposure_image=resolve_workspace_path(base_dir, str(raw["exposure_image"])),
            smoothed_flux_image=(
                resolve_workspace_path(base_dir, str(smooth_raw))
                if smooth_raw
                else None
            ),
        )


SubbandCombinedProducts = SubbandProductPaths


@dataclass
class BandProductPaths:
    """Structured combined outputs for one configured target energy band."""

    band_name: str
    emin: int
    emax: int
    flux_image: Path
    mask_image: Path | None = None
    subbands: list[SubbandProductPaths] = field(default_factory=list)

    def to_dict(
        self,
        base_dir: Path | str,
        *,
        workspace_root: Path | str | None = None,
    ) -> dict[str, Any]:
        """Serialize band product paths relative to ``base_dir``."""
        data: dict[str, Any] = {
            "emin": self.emin,
            "emax": self.emax,
            "flux_image": workspace_relative_path(
                self.flux_image, base_dir, workspace_root=workspace_root
            ),
            "subbands": [
                sb.to_dict(base_dir, workspace_root=workspace_root)
                for sb in self.subbands
            ],
        }
        if self.mask_image is not None:
            data["mask_image"] = workspace_relative_path(
                self.mask_image, base_dir, workspace_root=workspace_root
            )
        return data

    @classmethod
    def from_dict(
        cls,
        base_dir: Path | str,
        band_name: str,
        raw: Mapping[str, Any],
    ) -> BandProductPaths:
        """Deserialize a ``BandProductPaths`` record relative to ``base_dir``."""
        mask_raw = raw.get("mask_image")
        return cls(
            band_name=band_name,
            emin=int(raw["emin"]),
            emax=int(raw["emax"]),
            flux_image=resolve_workspace_path(base_dir, str(raw["flux_image"])),
            mask_image=(
                resolve_workspace_path(base_dir, str(mask_raw)) if mask_raw else None
            ),
            subbands=[
                SubbandProductPaths.from_dict(base_dir, sb)
                for sb in raw.get("subbands", [])
            ],
        )


BandCombinedProducts = BandProductPaths


@dataclass
class QPBImageResult:
    """Core artifacts produced by ``run_qpb_image``."""

    qpb_image_dir: Path
    manifest_path: Path
    qpb_files: dict[str, Path] = field(default_factory=dict)
    bands: dict[str, BandProductPaths] = field(default_factory=dict)
    band_flux_images: dict[str, Path] = field(default_factory=dict)
    subband_count_images: dict[str, list[Path]] = field(default_factory=dict)
    manifest: StageManifest | None = None
    workspace_root: Path | None = None

    def __post_init__(self) -> None:
        for band_name, band_prod in self.bands.items():
            self.band_flux_images.setdefault(band_name, band_prod.flux_image)

    def save_manifest(self) -> Path:
        """Persist a v2 ``manifest.json`` for standalone ``QPBImageResult`` instances."""
        ws_root = self.workspace_root
        products_payload: dict[str, Any] = {
            "qpb_files": {
                k: workspace_relative_path(p, self.qpb_image_dir, workspace_root=ws_root)
                for k, p in self.qpb_files.items()
            },
            "band_flux_images": {
                k: workspace_relative_path(p, self.qpb_image_dir, workspace_root=ws_root)
                for k, p in self.band_flux_images.items()
            },
            "bands": {
                bname: bprod.to_dict(self.qpb_image_dir, workspace_root=ws_root)
                for bname, bprod in self.bands.items()
            },
        }
        if self.subband_count_images:
            products_payload["subband_count_images"] = {
                bname: [
                    workspace_relative_path(p, self.qpb_image_dir, workspace_root=ws_root)
                    for p in paths
                ]
                for bname, paths in self.subband_count_images.items()
            }

        manifest = (
            StageManifest(
                stage=self.manifest.stage,
                status=self.manifest.status,
                schema_version=MANIFEST_SCHEMA_VERSION,
                config=dict(self.manifest.config),
                steps=dict(self.manifest.steps),
                products=products_payload,
                error=self.manifest.error,
            )
            if self.manifest is not None
            else StageManifest(
                stage="qpb_image",
                status="completed",
                schema_version=MANIFEST_SCHEMA_VERSION,
                products=products_payload,
            )
        )
        self.manifest = manifest
        return atomic_write_json(
            self.manifest_path,
            manifest.to_dict(self.qpb_image_dir, workspace_root=ws_root),
        )

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path | str,
        *,
        workspace_root: Path | str | None = None,
    ) -> QPBImageResult:
        """Load a ``QPBImageResult`` from a v2 or legacy v1 ``manifest.json``."""
        mpath = normalize_lexical_path(manifest_path)
        qdir = mpath.parent
        ws_root = (
            normalize_lexical_path(workspace_root)
            if workspace_root is not None
            else mpath.parent.parent
        )
        raw = read_json(mpath)

        stage_manifest: StageManifest | None = None
        if isinstance(raw, Mapping) and raw.get("schema_version") == MANIFEST_SCHEMA_VERSION:
            stage_manifest = StageManifest.from_dict(qdir, raw)
            payload: Mapping[str, Any] = stage_manifest.products
        else:
            payload = raw if isinstance(raw, Mapping) else {}

        raw_qpb = payload.get("qpb_files", {})
        raw_flux = payload.get("band_flux_images", {})
        raw_bands = payload.get("bands", {})
        raw_subcounts = payload.get("subband_count_images", {})

        qpb_files = (
            {
                str(k): resolve_workspace_path(qdir, str(v))
                for k, v in raw_qpb.items()
            }
            if isinstance(raw_qpb, Mapping)
            else {}
        )
        band_flux_images = (
            {
                str(k): resolve_workspace_path(qdir, str(v))
                for k, v in raw_flux.items()
            }
            if isinstance(raw_flux, Mapping)
            else {}
        )
        bands = (
            {
                str(bname): BandProductPaths.from_dict(qdir, str(bname), bdict)
                for bname, bdict in raw_bands.items()
                if isinstance(bdict, Mapping)
            }
            if isinstance(raw_bands, Mapping)
            else {}
        )
        subband_count_images = (
            {
                str(bname): [resolve_workspace_path(qdir, str(p)) for p in plist]
                for bname, plist in raw_subcounts.items()
                if isinstance(plist, list)
            }
            if isinstance(raw_subcounts, Mapping)
            else {}
        )
        return cls(
            qpb_image_dir=qdir,
            manifest_path=mpath,
            qpb_files=qpb_files,
            bands=bands,
            band_flux_images=band_flux_images,
            subband_count_images=subband_count_images,
            manifest=stage_manifest,
            workspace_root=ws_root,
        )
