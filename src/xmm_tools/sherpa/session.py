"""Joint Sherpa fitting session, parameter linking/auditing, and JSON export."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from xmm_tools.sherpa.components import (
    VALID_SCOPES,
    ModelBuilder,
    invoke_component_builder,
)
from xmm_tools.sherpa.dataset import SpectrumCollection, SpectrumDataset


def _require_sherpa_fit() -> dict[str, Any]:
    try:
        from sherpa.data import DataSimulFit
        from sherpa.fit import Fit
        from sherpa.models.model import SimulFitModel
        from sherpa.optmethods import LevMar, NelderMead
        from sherpa.stats import Chi2Gehrels, Chi2XspecVar, CStat, WStat
    except ImportError as exc:
        raise ImportError(
            "Sherpa fitting modules are required for JointFitSession."
        ) from exc
    return {
        "DataSimulFit": DataSimulFit,
        "Fit": Fit,
        "SimulFitModel": SimulFitModel,
        "LevMar": LevMar,
        "NelderMead": NelderMead,
        "CStat": CStat,
        "WStat": WStat,
        "Chi2Gehrels": Chi2Gehrels,
        "Chi2XspecVar": Chi2XspecVar,
    }


def _extract_model_tree(
    model: Any,
    err_map: dict[str, tuple[float | None, float | None]] | None = None,
) -> list[dict[str, Any]]:
    """Serialize all parameters of a Sherpa model expression cleanly."""
    if model is None:
        return []
    err_map = err_map or {}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for par in getattr(model, "pars", []):
        if par.fullname in seen:
            continue
        seen.add(par.fullname)
        err_lo, err_hi = err_map.get(par.fullname, (None, None))
        out.append(
            {
                "name": par.name,
                "fullname": par.fullname,
                "model": par.modelname,
                "val": float(par.val),
                "min": float(par.min),
                "max": float(par.max),
                "frozen": bool(par.frozen),
                "units": str(getattr(par, "units", "")),
                "linked_to": par.link.fullname if par.link is not None else None,
                "err_low": err_lo,
                "err_high": err_hi,
            }
        )
    return out


@dataclass
class FitReport:
    """Structured fit outcome & per-component parameter dump for JSON export."""

    stat_name: str
    stat_val: float
    dof: int
    reduced_stat: float
    succeeded: bool
    datasets: dict[str, dict[str, Any]] = field(default_factory=dict)
    components: dict[str, dict[str, list[dict[str, Any]]]] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stat_name": self.stat_name,
            "stat_val": self.stat_val,
            "dof": self.dof,
            "reduced_stat": self.reduced_stat,
            "succeeded": self.succeeded,
            "message": self.message,
            "datasets": self.datasets,
            "components": self.components,
        }

    def param(self, fullname: str) -> dict[str, Any]:
        """Look up one parameter dict by its Sherpa ``fullname``."""
        for ds_comps in self.components.values():
            for plist in ds_comps.values():
                for p in plist:
                    if p["fullname"] == fullname:
                        return p
        raise KeyError(f"Parameter {fullname!r} not found in FitReport")

    def summary(
        self,
        *,
        only_free: bool = True,
        include_linked: bool = False,
    ) -> str:
        """Return a concise human-readable table of fitted parameters."""
        lines = [
            (
                f"Fit Summary: {self.stat_name} = {self.stat_val:.3f} / {self.dof} dof "
                f"(reduced = {self.reduced_stat:.4f}, succeeded = {self.succeeded})"
            ),
            "-" * 78,
        ]
        seen: set[str] = set()
        for ds_comps in self.components.values():
            for plist in ds_comps.values():
                for p in plist:
                    fname = p["fullname"]
                    if fname in seen:
                        continue
                    seen.add(fname)
                    is_linked = p["linked_to"] is not None
                    if is_linked and not include_linked:
                        continue
                    if only_free and p["frozen"] and not is_linked:
                        continue
                    if is_linked:
                        val_str = f"-> {p['linked_to']} ({p['val']:.4g})"
                    elif p["err_low"] is not None and p["err_high"] is not None:
                        val_str = f"{p['val']:.4g} [{p['err_low']:+.3g}, {p['err_high']:+.3g}]"
                    else:
                        val_str = f"{p['val']:.4g}"
                    tag = " [frozen]" if p["frozen"] else ""
                    lines.append(f"  {fname:<42} {val_str}{tag}")
        return "\n".join(lines)

    def print_summary(
        self,
        *,
        only_free: bool = True,
        include_linked: bool = False,
    ) -> None:
        """Print ``summary()`` to stdout."""
        print(self.summary(only_free=only_free, include_linked=include_linked))

    def save_json(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        return out_path

    @classmethod
    def from_json(cls, path: str | Path) -> FitReport:
        """Load a ``FitReport`` previously saved by ``save_json()``."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            stat_name=str(raw.get("stat_name", "")),
            stat_val=float(raw.get("stat_val", 0.0)),
            dof=int(raw.get("dof", 0)),
            reduced_stat=float(raw.get("reduced_stat", 0.0)),
            succeeded=bool(raw.get("succeeded", False)),
            datasets=dict(raw.get("datasets", {})),
            components=dict(raw.get("components", {})),
            message=str(raw.get("message", "")),
        )

    def params_map(self) -> dict[str, dict[str, Any]]:
        """Return a flat mapping ``{fullname: param_dict}`` across all components."""
        out: dict[str, dict[str, Any]] = {}
        for ds_comps in self.components.values():
            for plist in ds_comps.values():
                for p in plist:
                    out.setdefault(p["fullname"], p)
        return out

    def component(
        self,
        component_name: str,
        *,
        region: str | None = None,
        instrument: str | None = None,
        full: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Return ``{tag: {param_name: val}}`` (or full param dicts if ``full=True``)."""
        for ds_id, ds_comps in self.components.items():
            meta = self.datasets.get(ds_id, {})
            if region is not None and meta.get("region") != region:
                continue
            if instrument is not None and meta.get("instrument") != instrument:
                continue
            if component_name not in ds_comps:
                continue
            grouped: dict[str, dict[str, Any]] = {}
            for p in ds_comps[component_name]:
                tag = str(p["model"]).rsplit("__", 1)[-1]
                grouped.setdefault(tag, {})[p["name"]] = p if full else p["val"]
            return grouped
        raise KeyError(
            f"Component {component_name!r} not found "
            f"(region={region!r}, instrument={instrument!r})"
        )

    def get(
        self,
        component_name: str,
        tag: str,
        param_name: str,
        *,
        region: str | None = None,
        instrument: str | None = None,
    ) -> float:
        """Shortcut to read one fitted parameter value from ``FitReport``."""
        comp = self.component(
            component_name, region=region, instrument=instrument, full=False
        )
        return float(comp[tag][param_name])


@dataclass
class _ComponentSpec:
    name: str
    spec: Any
    scope: str
    regions: set[str] | None = None
    per_arcmin2: bool = False
    use_proton_scale: bool = False
    response: str = "sky"


class JointFitSession:
    """Transparent OOP Sherpa joint-fitting session across XMM instruments and regions."""

    def __init__(self, collection: SpectrumCollection) -> None:
        self.collection = collection
        self._specs: dict[str, _ComponentSpec] = {}
        self.full_models: dict[str, Any] = {}
        self._assigned: dict[str, dict[str, Any]] = {}
        self.parts: dict[str, dict[str, dict[str, Any]]] = {}
        self._fit_obj: Any = None

    def _register(
        self,
        name: str,
        spec: Any,
        *,
        scope: str,
        regions: Iterable[str] | None = None,
        per_arcmin2: bool = False,
        use_proton_scale: bool = False,
        response: str = "sky",
    ) -> JointFitSession:
        if scope not in VALID_SCOPES:
            raise ValueError(f"Invalid scope {scope!r}; expected one of {VALID_SCOPES}")
        self._specs[name] = _ComponentSpec(
            name=name,
            spec=spec,
            scope=scope,
            regions=set(regions) if regions is not None else None,
            per_arcmin2=per_arcmin2,
            use_proton_scale=use_proton_scale,
            response=response,
        )
        return self

    def set_calibration(
        self,
        cal: Any,
        *,
        scope: str = "per_instrument",
        regions: Iterable[str] | None = None,
    ) -> JointFitSession:
        return self._register("calibration", cal, scope=scope, regions=regions)

    def set_sky_background(
        self,
        sky: Any,
        *,
        scope: str = "global",
        per_arcmin2: bool = True,
        regions: Iterable[str] | None = None,
    ) -> JointFitSession:
        return self._register(
            "sky_background",
            sky,
            scope=scope,
            regions=regions,
            per_arcmin2=per_arcmin2,
        )

    def set_source(
        self,
        source: Any,
        *,
        scope: str = "region",
        regions: Iterable[str] | None = None,
        per_arcmin2: bool = False,
    ) -> JointFitSession:
        return self._register(
            "source_emission",
            source,
            scope=scope,
            regions=regions,
            per_arcmin2=per_arcmin2,
        )

    def set_soft_proton(
        self,
        sp: Any,
        *,
        scope: str = "dataset",
        use_proton_scale: bool = False,
        regions: Iterable[str] | None = None,
    ) -> JointFitSession:
        return self._register(
            "soft_proton",
            sp,
            scope=scope,
            regions=regions,
            use_proton_scale=use_proton_scale,
            response="diagonal",
        )

    def set_instrument_lines(
        self,
        inst_lines: Any,
        *,
        scope: str = "dataset",
        response: str = "unit_arf",
        regions: Iterable[str] | None = None,
    ) -> JointFitSession:
        if response not in ("unit_arf", "sky_arf"):
            raise ValueError("response must be 'unit_arf' or 'sky_arf'")
        return self._register(
            "instrument_lines",
            inst_lines,
            scope=scope,
            regions=regions,
            response=response,
        )

    def _scope_key(self, comp: _ComponentSpec, ds: SpectrumDataset) -> str:
        if comp.scope == "global":
            return "global"
        if comp.scope == "region":
            return ds.region_key.replace("/", "__")
        if comp.scope == "per_instrument":
            return ds.instrument
        return ds.ds_id

    def build(self) -> JointFitSession:
        """Instantiate scoped models and assemble the 3-channel ``SimulFitModel``."""
        sh = _require_sherpa_fit()
        self.full_models.clear()
        self._assigned.clear()
        self.parts.clear()

        scope_cache: dict[tuple[str, str], tuple[Any, dict[str, Any]]] = {}
        phas: list[Any] = []
        models: list[Any] = []

        for ds in self.collection:
            ds_assigned: dict[str, Any] = {}
            ds_parts: dict[str, dict[str, Any]] = {}

            for cname, comp in self._specs.items():
                if comp.regions is not None and ds.region_key not in comp.regions:
                    continue
                skey = self._scope_key(comp, ds)
                cache_key = (cname, skey)
                if cache_key not in scope_cache:
                    mb = ModelBuilder(f"{skey}__{cname}")
                    mdl = invoke_component_builder(comp.spec, builder=mb, ds=ds)
                    scope_cache[cache_key] = (mdl, mb.parts)
                mdl, parts = scope_cache[cache_key]
                ds_assigned[cname] = mdl
                ds_parts[cname] = parts

            self._assigned[ds.ds_id] = ds_assigned
            self.parts[ds.ds_id] = ds_parts

            cal = ds_assigned.get("calibration")
            sky = ds_assigned.get("sky_background")
            src = ds_assigned.get("source_emission")
            sp = ds_assigned.get("soft_proton")
            inst = ds_assigned.get("instrument_lines")

            sky_sum = None
            if sky is not None:
                c_sky = self._specs["sky_background"]
                sky_term = ds.solid_angle * sky if c_sky.per_arcmin2 else sky
                sky_sum = sky_term
            if src is not None:
                c_src = self._specs["source_emission"]
                src_term = ds.solid_angle * src if c_src.per_arcmin2 else src
                sky_sum = src_term if sky_sum is None else (sky_sum + src_term)

            folded_terms: list[Any] = []
            if sky_sum is not None:
                celestial = cal * sky_sum if cal is not None else sky_sum
                folded_terms.append(ds.fold_sky(celestial))
            if sp is not None:
                c_sp = self._specs["soft_proton"]
                if c_sp.use_proton_scale:
                    if ds.proton_scale is None:
                        raise ValueError(
                            f"use_proton_scale=True requires proton_scale on {ds.ds_id!r}."
                        )
                    sp_term = ds.proton_scale * sp
                else:
                    sp_term = sp
                folded_terms.append(ds.fold_diagonal(sp_term))
            if inst is not None:
                c_inst = self._specs["instrument_lines"]
                if c_inst.response == "unit_arf":
                    folded_terms.append(ds.fold_unit_arf(inst))
                else:
                    folded_terms.append(ds.fold_sky(inst))

            if not folded_terms:
                raise ValueError(
                    f"No model components assigned to dataset {ds.ds_id!r}."
                )

            full = folded_terms[0]
            for term in folded_terms[1:]:
                full = full + term

            self.full_models[ds.ds_id] = full
            phas.append(ds.pha)
            models.append(full)

        self._data_simul = sh["DataSimulFit"]("xmm_data", phas)
        self._model_simul = sh["SimulFitModel"]("xmm_model", models)
        return self

    def folded_components(self, ds_id: str) -> dict[str, Any]:
        """Return each assigned component folded through its actual response."""
        if not self.full_models:
            self.build()
        ds = self.collection[ds_id]
        assigned = self._assigned[ds_id]
        cal = assigned.get("calibration")
        out: dict[str, Any] = {}

        if (sky := assigned.get("sky_background")) is not None:
            term = ds.solid_angle * sky if self._specs["sky_background"].per_arcmin2 else sky
            out["sky_background"] = ds.fold_sky(cal * term if cal is not None else term)

        if (src := assigned.get("source_emission")) is not None:
            term = ds.solid_angle * src if self._specs["source_emission"].per_arcmin2 else src
            out["source_emission"] = ds.fold_sky(cal * term if cal is not None else term)

        if (sp := assigned.get("soft_proton")) is not None:
            sp_term = (
                ds.proton_scale * sp
                if self._specs["soft_proton"].use_proton_scale and ds.proton_scale is not None
                else sp
            )
            out["soft_proton"] = ds.fold_diagonal(sp_term)

        if (inst := assigned.get("instrument_lines")) is not None:
            if self._specs["instrument_lines"].response == "unit_arf":
                out["instrument_lines"] = ds.fold_unit_arf(inst)
            else:
                out["instrument_lines"] = ds.fold_sky(inst)

        return out

    def plot(
        self,
        region: str,
        instrument: str | Iterable[str] | None = None,
        *,
        show_components: bool | None = None,
        per_kev: bool = True,
        title: str | None = None,
        save_path: str | Path | None = None,
        dpi: int = 300,
    ) -> tuple[Any, tuple[Any, Any]]:
        """Plot observed data, folded components, and Δχ residuals for ``region``."""
        from xmm_tools.sherpa.plotting import plot_fit

        return plot_fit(
            self,
            region,
            instrument=instrument,
            show_components=show_components,
            per_kev=per_kev,
            title=title,
            save_path=save_path,
            dpi=dpi,
        )

    def model(self, region: str, instrument: str, component: str) -> Any:
        """Return the live Sherpa composite model for ``(region, instrument, component)``."""
        ds = self.collection[region, instrument]
        return self._assigned[ds.ds_id][component]

    def part(self, region: str, instrument: str, component: str, tag: str) -> Any:
        """Return a named sub-model created via ``ModelBuilder``."""
        ds = self.collection[region, instrument]
        return self.parts[ds.ds_id][component][tag]

    def _resolve_param_path(self, path: str) -> Any:
        """Resolve ``'<region>.<inst>.<component>.<tag>.<param>'`` or ``'<region>.<component>.<tag>.<param>'``."""
        tokens = path.split(".")
        if len(tokens) == 4:
            reg, comp, tag, pname = tokens
            inst = next(
                (
                    ds.instrument
                    for ds in self.collection.by_region(reg)
                    if comp in self.parts[ds.ds_id] and tag in self.parts[ds.ds_id][comp]
                ),
                None,
            )
            if inst is None:
                raise KeyError(
                    f"Component '{comp}.{tag}' not found in region {reg!r} (path={path!r})"
                )
        elif len(tokens) == 5:
            reg, inst, comp, tag, pname = tokens
        else:
            raise ValueError(
                f"Invalid parameter path {path!r}. Expected "
                "'<region>.<component>.<tag>.<param>' or "
                "'<region>.<inst>.<component>.<tag>.<param>'."
            )
        sub = self.part(reg, inst, comp, tag)
        return getattr(sub, pname)

    def link(self, target_path: str, source_path: str) -> JointFitSession:
        """Link ``target_path`` parameter directly to ``source_path`` parameter."""
        if not self.full_models:
            self.build()
        target_par = self._resolve_param_path(target_path)
        source_par = self._resolve_param_path(source_path)
        target_par.link = source_par
        return self

    def audit_parameters(self) -> dict[str, list[dict[str, Any]]]:
        """Return deduplicated lists of free and linked parameters."""
        if not self.full_models:
            self.build()
        seen: set[str] = set()
        free: list[dict[str, Any]] = []
        linked: list[dict[str, Any]] = []
        for mdl in self.full_models.values():
            for par in mdl.pars:
                if par.fullname in seen:
                    continue
                seen.add(par.fullname)
                if par.link is not None:
                    linked.append(
                        {"param": par.fullname, "linked_to": par.link.fullname}
                    )
                elif not par.frozen:
                    free.append(
                        {
                            "param": par.fullname,
                            "val": float(par.val),
                            "min": float(par.min),
                            "max": float(par.max),
                        }
                    )
        return {"free": free, "linked": linked}

    def fit(
        self,
        *,
        stat: str = "wstat",
        method: str = "levmar",
        estimate_errors: bool = False,
    ) -> FitReport:
        """Run joint Sherpa fit and return a JSON-serializable ``FitReport``."""
        sh = _require_sherpa_fit()
        if not self.full_models:
            self.build()

        stat_map = {
            "wstat": sh["WStat"],
            "cstat": sh["CStat"],
            "chi2gehrels": sh["Chi2Gehrels"],
            "chi2xspecvar": sh["Chi2XspecVar"],
        }
        method_map = {
            "levmar": sh["LevMar"],
            "neldermead": sh["NelderMead"],
        }
        if stat not in stat_map:
            raise ValueError(f"Unsupported stat {stat!r}; choose from {tuple(stat_map)}")
        if method not in method_map:
            raise ValueError(f"Unsupported method {method!r}; choose from {tuple(method_map)}")

        self._fit_obj = sh["Fit"](
            self._data_simul,
            self._model_simul,
            stat=stat_map[stat](),
            method=method_map[method](),
        )
        res = self._fit_obj.fit()

        err_map: dict[str, tuple[float | None, float | None]] = {}
        if estimate_errors and res.succeeded:
            eres = self._fit_obj.est_errors()
            for pname, lo, hi in zip(eres.parnames, eres.parmins, eres.parmaxes):
                err_map[pname] = (
                    float(lo) if lo is not None else None,
                    float(hi) if hi is not None else None,
                )

        ds_meta: dict[str, dict[str, Any]] = {}
        comp_dump: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for ds in self.collection:
            ds_meta[ds.ds_id] = {
                "region": ds.region_key,
                "instrument": ds.instrument,
                "solid_angle_arcmin2": ds.solid_angle,
                "proton_scale": ds.proton_scale,
                "exposure_s": ds.exposure,
                "pha": str(ds.pha_path),
            }
            comp_dump[ds.ds_id] = {
                cname: _extract_model_tree(cmodel, err_map)
                for cname, cmodel in self._assigned[ds.ds_id].items()
                if cmodel is not None
            }

        return FitReport(
            stat_name=stat,
            stat_val=float(res.statval),
            dof=int(res.dof),
            reduced_stat=float(res.rstat) if res.rstat is not None else 0.0,
            succeeded=bool(res.succeeded),
            datasets=ds_meta,
            components=comp_dump,
            message=str(getattr(res, "message", "")),
        )
