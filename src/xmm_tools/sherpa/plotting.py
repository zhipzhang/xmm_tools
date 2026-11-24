"""Academic-style spectral fit + component breakdown + residual plots for Sherpa."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from xmm_tools.sherpa.dataset import SpectrumDataset

_INST_COLORS = {
    "mos1": "#1f77b4",
    "mos2": "#2ca02c",
    "pn": "#111111",
}

_COMP_STYLES = {
    "sky_background": {"color": "#7f7f7f", "ls": "--", "lw": 1.3, "label": "Sky Bkg"},
    "source_emission": {"color": "#d62728", "ls": "-", "lw": 1.5, "label": "Source"},
    "soft_proton": {"color": "#ff7f0e", "ls": "-.", "lw": 1.3, "label": "Soft Proton"},
    "instrument_lines": {"color": "#9467bd", "ls": ":", "lw": 1.3, "label": "Inst. Lines"},
}


def _extract_binned_spectrum(
    ds: SpectrumDataset,
    model: Any,
    *,
    per_kev: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x_mid, x_half, y_data, y_err, y_model)`` via Sherpa plot objects."""
    from sherpa.astro.plot import DataPHAPlot, ModelPHAHistogram

    dplot = DataPHAPlot()
    dplot.histo_prefs["rate"] = per_kev
    dplot.prepare(ds.pha)

    mplot = ModelPHAHistogram()
    mplot.histo_prefs["rate"] = per_kev
    mplot.prepare(ds.pha, model)

    x_mid = np.asarray(dplot.x, dtype=float)
    x_half = (
        0.5 * np.asarray(dplot.xerr, dtype=float)
        if dplot.xerr is not None
        else np.zeros_like(x_mid)
    )
    y_data = np.asarray(dplot.y, dtype=float)
    y_err = (
        np.asarray(dplot.yerr, dtype=float)
        if dplot.yerr is not None
        else np.zeros_like(y_data)
    )
    y_model = np.asarray(mplot.y, dtype=float)
    return x_mid, x_half, y_data, y_err, y_model


def plot_fit(
    session: Any,
    region: str,
    instrument: str | Iterable[str] | None = None,
    *,
    show_components: bool | None = None,
    per_kev: bool = True,
    title: str | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
) -> tuple[Any, tuple[Any, Any]]:
    """Plot observed spectrum, folded model components, and Δχ residuals.

    Parameters
    ----------
    session:
        Built/fitted ``JointFitSession``.
    region:
        Region key (e.g. ``"east_jet/e4"`` or ``"background"``).
    instrument:
        Single instrument (``"mos1"``), list of instruments (``["mos1", "pn"]``),
        or ``None`` to plot all active instruments in ``region``.
    show_components:
        Whether to overlay individual folded components (``sky_background``,
        ``source_emission``, ``soft_proton``, ``instrument_lines``). Defaults to
        ``True`` when plotting a single instrument and ``False`` for overlays.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    if not session.full_models:
        session.build()

    if instrument is None:
        datasets = session.collection.by_region(region)
    elif isinstance(instrument, str):
        datasets = [session.collection[region, instrument]]
    else:
        datasets = [session.collection[region, inst] for inst in instrument]

    if not datasets:
        raise ValueError(f"No active datasets found for region={region!r}")

    if show_components is None:
        show_components = len(datasets) == 1

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "axes.linewidth": 1.0,
        }
    )

    fig, (ax_spec, ax_res) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7.2, 5.6),
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.06},
    )

    for ds in datasets:
        color = _INST_COLORS.get(ds.instrument, "#333333")
        full_mdl = session.full_models[ds.ds_id]
        x, xh, y, ye, ym = _extract_binned_spectrum(ds, full_mdl, per_kev=per_kev)

        inst_label = ds.instrument.upper()
        ax_spec.errorbar(
            x,
            y,
            xerr=xh,
            yerr=ye,
            fmt="o",
            ms=3.2,
            lw=0.85,
            elinewidth=0.85,
            capsize=0,
            color=color,
            alpha=0.88,
            label=f"{inst_label} data",
            zorder=3,
        )
        ax_spec.step(
            x,
            ym,
            where="mid",
            color=color,
            lw=1.7,
            label=f"{inst_label} total",
            zorder=4,
        )

        # Residual: (Data - Model) / Error
        safe_err = np.where(ye > 0, ye, 1.0)
        resid = (y - ym) / safe_err
        ax_res.errorbar(
            x,
            resid,
            xerr=xh,
            yerr=np.ones_like(resid),
            fmt="o",
            ms=2.8,
            lw=0.8,
            elinewidth=0.8,
            capsize=0,
            color=color,
            alpha=0.85,
        )

        # Folded individual components
        if show_components:
            folded_comps = session.folded_components(ds.ds_id)
            for cname, fmdl in folded_comps.items():
                if cname not in _COMP_STYLES:
                    continue
                _, _, _, _, yc = _extract_binned_spectrum(ds, fmdl, per_kev=per_kev)
                if np.all(yc <= 0):
                    continue
                st = _COMP_STYLES[cname]
                prefix = f"{inst_label} " if len(datasets) > 1 else ""
                ax_spec.step(
                    x,
                    yc,
                    where="mid",
                    color=st["color"],
                    ls=st["ls"],
                    lw=st["lw"],
                    label=f"{prefix}{st['label']}",
                    zorder=2,
                )

    ax_spec.set_xscale("log")
    ax_spec.set_yscale("log")
    ylabel = (
        r"Normalized Rate ($\mathrm{counts\;s^{-1}\;keV^{-1}}$)"
        if per_kev
        else r"Counts ($\mathrm{counts\;bin^{-1}}$)"
    )
    ax_spec.set_ylabel(ylabel, fontsize=11)
    ax_spec.set_title(title or f"{region}", fontsize=12, pad=8)
    ax_spec.legend(
        frameon=False,
        fontsize=8.5,
        ncol=2 if show_components else 1,
        loc="best",
    )

    ax_res.axhline(0.0, color="0.35", ls="--", lw=0.9)
    ax_res.set_ylabel(r"$\Delta\chi$", fontsize=11)
    ax_res.set_xlabel("Energy (keV)", fontsize=11)
    ax_res.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax_res.xaxis.set_minor_formatter(FormatStrFormatter("%g"))

    for ax in (ax_spec, ax_res):
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=9.5)

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")

    return fig, (ax_spec, ax_res)
