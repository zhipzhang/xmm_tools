"""Shared execution context for pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xmm_tools.exec.runner import SASRunner
from xmm_tools.pipeline.config import AnaConfig


@dataclass
class PipelineContext:
    """Mutable state passed through ordered pipeline stages."""

    config: AnaConfig
    dry_run: bool = False
    force: bool = False
    docker_container: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)

    def make_runner(self, *, skip_if_exists: bool | None = None) -> SASRunner:
        """Create a configured ``SASRunner`` for a stage."""
        skip = (not self.force) if skip_if_exists is None else skip_if_exists
        return SASRunner(
            env=self.config.env,
            dry_run=self.dry_run,
            skip_if_exists=skip,
            docker_container=self.docker_container,
        )
