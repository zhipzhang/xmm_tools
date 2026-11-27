"""Data models and manifest serialization for ``xmm_tools.esas.spectrum``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xmm_tools.config import AnaConfig
from xmm_tools.esas.naming import load_stage_manifest_payload, save_stage_manifest
from xmm_tools.exec import (
    StageManifest,
    is_nonempty_file,
    normalize_lexical_path,
    resolve_workspace_path,
    workspace_relative_path,
)


def _rel_opt(
    path: Path | None,
    base_dir: Path | str,
    *,
    workspace_root: Path | str | None,
) -> str | None:
    if path is None:
        return None
    resolved = resolve_workspace_path(base_dir, path)
    return workspace_relative_path(resolved, base_dir, workspace_root=workspace_root)


def _abs_opt(val: Any, base_dir: Path | str) -> Path | None:
    if val is None or not str(val).strip():
        return None
    return resolve_workspace_path(base_dir, str(val))


@dataclass
class InstrumentSpectrumProducts:
    """Extracted spectrum & background products for one instrument in one region."""

    fovt: Path | None = None
    fovtoot: Path | None = None
    fovtootsub: Path | None = None
    bkg: Path | None = None
    rmf: Path | None = None
    arf: Path | None = None
    qdp_plots: dict[str, Path] = field(default_factory=dict)
    region_file: Path | None = None
    event_count: int | None = None
    proton_area: float | None = None
    proton_ave_flux: float | None = None
    proton_scale: float | None = None
    skipped: bool = False
    skip_reason: str = ""

    def core_files_ready(self, inst: str) -> bool:
        """Return True when this instrument is skipped or all core files exist."""
        if self.skipped:
            return True
        required = [self.fovt, self.bkg, self.rmf, self.arf]
        if inst == "pn":
            required.extend([self.fovtoot, self.fovtootsub])
        return all(p is not None and is_nonempty_file(p) for p in required)

    def to_dict(
        self,
        base_dir: Path | str,
        *,
        workspace_root: Path | str | None = None,
    ) -> dict[str, Any]:
        """Serialize instrument products relative to ``base_dir``."""
        return {
            "fovt": _rel_opt(self.fovt, base_dir, workspace_root=workspace_root),
            "fovtoot": _rel_opt(self.fovtoot, base_dir, workspace_root=workspace_root),
            "fovtootsub": _rel_opt(
                self.fovtootsub, base_dir, workspace_root=workspace_root
            ),
            "bkg": _rel_opt(self.bkg, base_dir, workspace_root=workspace_root),
            "rmf": _rel_opt(self.rmf, base_dir, workspace_root=workspace_root),
            "arf": _rel_opt(self.arf, base_dir, workspace_root=workspace_root),
            "qdp_plots": {
                k: _rel_opt(v, base_dir, workspace_root=workspace_root)
                for k, v in self.qdp_plots.items()
            },
            "region_file": _rel_opt(
                self.region_file, base_dir, workspace_root=workspace_root
            ),
            "event_count": self.event_count,
            "proton_area": self.proton_area,
            "proton_ave_flux": self.proton_ave_flux,
            "proton_scale": self.proton_scale,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }

    def to_manifest_dict(
        self,
        base_dir: Path | str,
        *,
        workspace_root: Path | str | None = None,
    ) -> dict[str, Any]:
        return self.to_dict(base_dir, workspace_root=workspace_root)

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, Any], base_dir: Path | str
    ) -> InstrumentSpectrumProducts:
        """Deserialize instrument products relative to ``base_dir``."""
        qdp_raw = raw.get("qdp_plots", {})
        qdp_plots = (
            {
                str(k): p
                for k, v in qdp_raw.items()
                if (p := _abs_opt(v, base_dir)) is not None
            }
            if isinstance(qdp_raw, Mapping)
            else {}
        )
        return cls(
            fovt=_abs_opt(raw.get("fovt"), base_dir),
            fovtoot=_abs_opt(raw.get("fovtoot"), base_dir),
            fovtootsub=_abs_opt(raw.get("fovtootsub"), base_dir),
            bkg=_abs_opt(raw.get("bkg"), base_dir),
            rmf=_abs_opt(raw.get("rmf"), base_dir),
            arf=_abs_opt(raw.get("arf"), base_dir),
            qdp_plots=qdp_plots,
            region_file=_abs_opt(raw.get("region_file"), base_dir),
            event_count=raw.get("event_count"),
            proton_area=raw.get("proton_area"),
            proton_ave_flux=raw.get("proton_ave_flux"),
            proton_scale=raw.get("proton_scale"),
            skipped=bool(raw.get("skipped", False)),
            skip_reason=str(raw.get("skip_reason", "")),
        )

    @classmethod
    def from_manifest_dict(
        cls, raw: Mapping[str, Any], base_dir: Path | str
    ) -> InstrumentSpectrumProducts:
        return cls.from_dict(raw, base_dir)


InstrumentProducts = InstrumentSpectrumProducts


@dataclass
class RegionSpectrumResult:
    """Spectrum extraction result for a single region directory."""

    key: str
    group: str
    label: str
    mode: str
    region_dir: Path
    source_file: Path
    elow: int
    ehigh: int
    active: bool = True
    instruments: dict[str, InstrumentSpectrumProducts] = field(default_factory=dict)

    def all_ready(self) -> bool:
        return bool(self.instruments) and all(
            prod.core_files_ready(inst) for inst, prod in self.instruments.items()
        )

    def to_dict(
        self,
        spectrum_dir: Path | str,
        *,
        workspace_root: Path | str | None = None,
    ) -> dict[str, Any]:
        """Serialize region result with workspace-portable paths."""
        resolved_rdir = resolve_workspace_path(spectrum_dir, self.region_dir)
        src_str = str(self.source_file)
        if src_str and src_str != ".":
            resolved_src = resolve_workspace_path(spectrum_dir, self.source_file)
            portable_src = workspace_relative_path(
                resolved_src, spectrum_dir, workspace_root=workspace_root
            )
        else:
            portable_src = ""
        return {
            "group": self.group,
            "label": self.label,
            "mode": self.mode,
            "dir": workspace_relative_path(
                resolved_rdir, spectrum_dir, workspace_root=workspace_root
            ),
            "source_file": portable_src,
            "elow": self.elow,
            "ehigh": self.ehigh,
            "active": self.active,
            "instruments": {
                inst: prod.to_dict(resolved_rdir, workspace_root=workspace_root)
                for inst, prod in self.instruments.items()
            },
        }

    def to_manifest_dict(
        self,
        spectrum_dir: Path | str,
        *,
        workspace_root: Path | str | None = None,
    ) -> dict[str, Any]:
        return self.to_dict(spectrum_dir, workspace_root=workspace_root)

    @classmethod
    def from_dict(
        cls,
        key: str,
        raw: Mapping[str, Any],
        spectrum_dir: Path | str,
    ) -> RegionSpectrumResult:
        """Deserialize a ``RegionSpectrumResult`` relative to ``spectrum_dir``."""
        rdir = resolve_workspace_path(spectrum_dir, str(raw["dir"]))
        raw_src = str(raw.get("source_file", ""))
        source_file = (
            resolve_workspace_path(spectrum_dir, raw_src) if raw_src else Path("")
        )
        raw_insts = raw.get("instruments", {})
        insts = (
            {
                str(inst): InstrumentSpectrumProducts.from_dict(pdata, rdir)
                for inst, pdata in raw_insts.items()
                if isinstance(pdata, Mapping)
            }
            if isinstance(raw_insts, Mapping)
            else {}
        )
        return cls(
            key=key,
            group=str(raw.get("group", key)),
            label=str(raw.get("label", key)),
            mode=str(raw.get("mode", "single")),
            region_dir=rdir,
            source_file=source_file,
            elow=int(raw.get("elow", 400)),
            ehigh=int(raw.get("ehigh", 7200)),
            active=bool(raw.get("active", True)),
            instruments=insts,
        )

    @classmethod
    def from_manifest_dict(
        cls,
        key: str,
        raw: Mapping[str, Any],
        spectrum_dir: Path | str,
    ) -> RegionSpectrumResult:
        return cls.from_dict(key, raw, spectrum_dir)


@dataclass
class SpectrumResult:
    """Root manifest contract for ``spectrum/manifest.json``."""

    spectrum_dir: Path
    manifest_path: Path
    regions: dict[str, RegionSpectrumResult] = field(default_factory=dict)
    workspace_root: Path | None = None
    manifest: StageManifest | None = None

    def save_manifest(self) -> Path:
        """Persist a v2 ``spectrum/manifest.json`` preserving ``workspace_root``."""
        ws_root = self.workspace_root
        products_payload: dict[str, Any] = {
            "regions": {
                key: reg.to_dict(self.spectrum_dir, workspace_root=ws_root)
                for key, reg in sorted(self.regions.items())
            }
        }
        self.manifest, written = save_stage_manifest(
            self.manifest_path,
            self.spectrum_dir,
            stage="spectrum",
            products=products_payload,
            existing_manifest=self.manifest,
            workspace_root=ws_root,
        )
        return written

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path | str,
        *,
        workspace_root: Path | str | None = None,
        require_completed: bool = False,
    ) -> SpectrumResult:
        """Load a ``SpectrumResult`` from a v2 or legacy v1 ``manifest.json``."""
        mpath = normalize_lexical_path(manifest_path)
        sdir = mpath.parent
        ws_root = (
            normalize_lexical_path(workspace_root)
            if workspace_root is not None
            else mpath.parent.parent
        )
        stage_manifest, payload = load_stage_manifest_payload(
            mpath, stage_label="Spectrum", require_completed=require_completed
        )

        raw_regions = payload.get("regions", {})
        regions = (
            {
                str(key): RegionSpectrumResult.from_dict(str(key), rdata, sdir)
                for key, rdata in raw_regions.items()
                if isinstance(rdata, Mapping)
            }
            if isinstance(raw_regions, Mapping)
            else {}
        )
        return cls(
            spectrum_dir=sdir,
            manifest_path=mpath,
            regions=regions,
            workspace_root=ws_root,
            manifest=stage_manifest,
        )


def load_spectrum_result(
    cfg: AnaConfig,
    *,
    require_completed: bool = False,
) -> SpectrumResult:
    """Load ``spectrum`` outputs from ``spectrum/manifest.json``."""
    sdir = normalize_lexical_path(cfg.paths.resolve(cfg.paths.spectrum_dir))
    ws_root = normalize_lexical_path(cfg.paths.workdir)
    manifest_path = sdir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Spectrum manifest not found at {manifest_path}; run 'xmm-ana spectrum' first."
        )
    return SpectrumResult.from_manifest(
        manifest_path,
        workspace_root=ws_root,
        require_completed=require_completed,
    )
