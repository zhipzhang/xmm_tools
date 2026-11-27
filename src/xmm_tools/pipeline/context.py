"""Shared execution context for pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xmm_tools.config import AnaConfig
from xmm_tools.exec.runner import SASRunner


@dataclass
class PipelineContext:
    """Mutable state passed through ordered pipeline stages."""

    config: AnaConfig
    dry_run: bool = False
    force: bool = False
    docker_container: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)

    def make_runner(self, *, skip_if_exists: bool = False) -> SASRunner:
        """Create a configured ``SASRunner`` for a stage."""
        return SASRunner(
            env=self.config.env,
            dry_run=self.dry_run,
            skip_if_exists=skip_if_exists,
            docker_container=self.docker_container,
        )
