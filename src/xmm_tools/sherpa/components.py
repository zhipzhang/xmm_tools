"""Transparent, scoped Sherpa model builder for XMM-Newton ESAS fitting."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class ModelBuilder:
    """Factory passed into user ``lambda m: ...`` / builder functions.

    Every Sherpa model created through ``m`` gets a unique, collision-free
    Sherpa name (``<prefix>__<tag>``) and is registered in ``m.parts[tag]``
    so parameters can be inspected or linked by path.
    """

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.parts: dict[str, Any] = {}

    def create(
        self,
        model_cls: type[Any],
        tag: str,
        *,
        frozen: bool | Iterable[str] = False,
        bounds: dict[str, tuple[float | None, float | None]] | None = None,
        **params: float,
    ) -> Any:
        """Instantiate ``model_cls(f'{prefix}__{tag}')`` and configure parameters."""
        if tag in self.parts:
            raise ValueError(f"Duplicate component tag {tag!r} in scope {self.prefix!r}")
        mdl = model_cls(f"{self.prefix}__{tag}")

        for pname, val in params.items():
            par = getattr(mdl, pname)
            par.val = val

        if bounds:
            for pname, (lo, hi) in bounds.items():
                par = getattr(mdl, pname)
                if lo is not None:
                    par.min = lo
                if hi is not None:
                    par.max = hi

        if frozen is True:
            for par in mdl.pars:
                par.freeze()
        elif isinstance(frozen, (list, tuple, set)):
            for pname in frozen:
                getattr(mdl, pname).freeze()

        self.parts[tag] = mdl
        return mdl

    # ---- Native Sherpa models (no HEADAS required) ----
    def const1d(self, tag: str = "const", **kwargs: Any) -> Any:
        from sherpa.models.basic import Const1D

        return self.create(Const1D, tag, **kwargs)

    def powlaw1d(self, tag: str = "pl", **kwargs: Any) -> Any:
        from sherpa.models.basic import PowLaw1D

        return self.create(PowLaw1D, tag, **kwargs)

    def gauss1d(self, tag: str = "gauss", **kwargs: Any) -> Any:
        from sherpa.models.basic import Gauss1D

        return self.create(Gauss1D, tag, **kwargs)

    # ---- XSPEC models (available when HEADAS is initialized) ----
    def constant(self, tag: str = "cal", **kwargs: Any) -> Any:
        from sherpa.astro.xspec import XSconstant

        return self.create(XSconstant, tag, **kwargs)

    def tbabs(self, tag: str = "tbabs", **kwargs: Any) -> Any:
        from sherpa.astro.xspec import XSTBabs

        return self.create(XSTBabs, tag, **kwargs)

    def apec(self, tag: str = "apec", **kwargs: Any) -> Any:
        from sherpa.astro.xspec import XSapec

        return self.create(XSapec, tag, **kwargs)

    def vapec(self, tag: str = "vapec", **kwargs: Any) -> Any:
        from sherpa.astro.xspec import XSvapec

        return self.create(XSvapec, tag, **kwargs)

    def vnei(self, tag: str = "vnei", **kwargs: Any) -> Any:
        from sherpa.astro.xspec import XSvnei

        return self.create(XSvnei, tag, **kwargs)

    def powerlaw(self, tag: str = "pl", **kwargs: Any) -> Any:
        from sherpa.astro.xspec import XSpowerlaw

        return self.create(XSpowerlaw, tag, **kwargs)

    def bknpowerlaw(self, tag: str = "bknpl", **kwargs: Any) -> Any:
        from sherpa.astro.xspec import XSbknpower

        return self.create(XSbknpower, tag, **kwargs)

    def gaussian(self, tag: str = "gauss", **kwargs: Any) -> Any:
        from sherpa.astro.xspec import XSgaussian

        return self.create(XSgaussian, tag, **kwargs)


VALID_SCOPES = ("global", "region", "per_instrument", "dataset")


def invoke_component_builder(
    spec: Any,
    *,
    builder: ModelBuilder,
    ds: Any,
) -> Any:
    """Invoke a 1-arg ``fn(m)``, 2-arg ``fn(m, ds)``, or return a direct Sherpa model."""
    if spec is None:
        return None
    if callable(spec) and not hasattr(spec, "pars"):
        import inspect

        sig = inspect.signature(spec)
        if len(sig.parameters) >= 2:
            return spec(builder, ds)
        return spec(builder)
    return spec
