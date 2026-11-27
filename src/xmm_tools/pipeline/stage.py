"""Shared stage utilities and ``StageSpec`` contract."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from xmm_tools.exec.paths import all_nonempty, read_log_tail

if TYPE_CHECKING:
    from xmm_tools.pipeline.context import PipelineContext

__all__ = ["StageSpec", "all_nonempty", "read_log_tail"]


@dataclass(frozen=True)
class StageSpec:
    """Declarative pipeline stage specification with explicit upstream ``requires``."""

    name: str
    requires: tuple[str, ...]
    run: Callable[[PipelineContext], Any]
    should_skip: Callable[[PipelineContext, set[str]], str | None] | None = None
