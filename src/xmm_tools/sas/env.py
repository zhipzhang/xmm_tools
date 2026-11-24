"""SAS/HEASoft environment configuration and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import shlex
from typing import Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]


_SAS_ENV_KEYS: tuple[tuple[str, str], ...] = (
    ("sas_dir", "SAS_DIR"),
    ("sas_ccf", "SAS_CCF"),
    ("sas_ccfpath", "SAS_CCFPATH"),
    ("sas_odf", "SAS_ODF"),
    ("caldb", "CALDB"),
)


@dataclass
class SASEnv:
    """Container for XMM-Newton SAS environment variables."""

    sas_dir: Path | str | None = None
    sas_ccf: Path | str | None = None
    sas_ccfpath: Path | str | None = None
    sas_odf: Path | str | None = None
    caldb: Path | str | None = None
    extra_env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_environ(cls, environ: Mapping[str, str] | None = None) -> SASEnv:
        """Create a ``SASEnv`` from ``os.environ`` (or a custom mapping)."""
        src = os.environ if environ is None else environ
        return cls(
            sas_dir=src.get("SAS_DIR"),
            sas_ccf=src.get("SAS_CCF"),
            sas_ccfpath=src.get("SAS_CCFPATH"),
            sas_odf=src.get("SAS_ODF"),
            caldb=src.get("CALDB"),
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> SASEnv:
        """Load ``[env]`` from a TOML configuration file."""
        toml_path = Path(path)
        data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        env_sec = data.get("env", {})
        if not isinstance(env_sec, dict):
            raise ValueError(f"Expected [env] table in {toml_path}")

        known_keys = {attr for attr, _ in _SAS_ENV_KEYS}
        extra: dict[str, str] = {}
        for k, v in env_sec.items():
            if k not in known_keys and isinstance(v, (str, int, float, bool)):
                extra[str(k)] = str(v)

        return cls(
            sas_dir=env_sec.get("sas_dir"),
            sas_ccf=env_sec.get("sas_ccf"),
            sas_ccfpath=env_sec.get("sas_ccfpath"),
            sas_odf=env_sec.get("sas_odf"),
            caldb=env_sec.get("caldb"),
            extra_env=extra,
        )

    def to_env_dict(
        self, base_env: Mapping[str, str] | None = None
    ) -> dict[str, str]:
        """Return a merged environment dictionary suitable for ``subprocess``."""
        merged = dict(os.environ if base_env is None else base_env)
        for attr, env_name in _SAS_ENV_KEYS:
            val = getattr(self, attr)
            if val is not None and str(val) != "":
                merged[env_name] = str(val)
        for k, v in self.extra_env.items():
            merged[k] = str(v)
        return merged

    def validate(
        self,
        *,
        required: tuple[str, ...] = ("SAS_CCF", "SAS_ODF"),
        check_paths: bool = True,
    ) -> None:
        """Validate that required SAS variables are set and optionally exist."""
        env_map = {
            env_name: getattr(self, attr)
            for attr, env_name in _SAS_ENV_KEYS
        }
        missing = [
            name for name in required if not env_map.get(name)
        ]
        if missing:
            raise ValueError(
                f"Missing required SAS environment variables: {', '.join(missing)}"
            )

        if check_paths:
            for name in required:
                p = Path(str(env_map[name]))
                if not p.exists():
                    raise FileNotFoundError(
                        f"SAS path for {name} does not exist: {p}"
                    )

    def shell_exports(self) -> list[str]:
        """Return ``export KEY=VALUE`` lines for shell script generation."""
        lines: list[str] = []
        for attr, env_name in _SAS_ENV_KEYS:
            val = getattr(self, attr)
            if val is not None and str(val) != "":
                lines.append(f"export {env_name}={shlex.quote(str(val))}")
        for k, v in self.extra_env.items():
            lines.append(f"export {k}={shlex.quote(str(v))}")
        return lines
