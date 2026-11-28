"""Portable path normalization, SHA-256 identity, symlinks, and atomic I/O."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal


def normalize_lexical_path(path: Path | str) -> Path:
    """Return an absolute path with ``..`` collapsed without resolving symlinks."""
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _to_posix(rel: str | Path) -> str:
    return PurePosixPath(str(rel).replace(os.sep, "/")).as_posix()


def _real_parent_path(path: Path) -> Path:
    """Resolve ancestor directory symlinks without dereferencing the leaf entry."""
    return Path(os.path.realpath(path.parent)) / path.name


def workspace_relative_path(
    path: Path | str,
    base_dir: Path | str,
    *,
    workspace_root: Path | str | None = None,
) -> str:
    """Return a portable POSIX path relative to ``base_dir`` when within workspace.

    When ``path`` resides under ``base_dir``, returns the path relative to
    ``base_dir``. When ``workspace_root`` is provided and both ``path`` and
    ``base_dir`` reside inside ``workspace_root``, returns a relative POSIX path
    (e.g. ``"../soft_proton_flare/mos1S001-gti.fits"``) so sibling-stage
    references never embed host or container absolute prefixes. Otherwise falls
    back to the lexically normalized POSIX path.
    """
    norm_path = normalize_lexical_path(path)
    norm_base = normalize_lexical_path(base_dir)
    norm_root = (
        normalize_lexical_path(workspace_root)
        if workspace_root is not None
        else None
    )

    if norm_path.is_relative_to(norm_base):
        return _to_posix(os.path.relpath(norm_path, norm_base))

    if (
        norm_root is not None
        and norm_path.is_relative_to(norm_root)
        and norm_base.is_relative_to(norm_root)
    ):
        return _to_posix(os.path.relpath(norm_path, norm_base))

    # Secondary check for OS-level ancestor directory symlinks (e.g. macOS
    # /var -> /private/var), resolving parent directories only so leaf artifact
    # symlinks are never dereferenced.
    real_path = _real_parent_path(norm_path)
    real_base = Path(os.path.realpath(norm_base))
    if real_path.is_relative_to(real_base):
        return _to_posix(os.path.relpath(real_path, real_base))

    if norm_root is not None:
        real_root = Path(os.path.realpath(norm_root))
        if real_path.is_relative_to(real_root) and real_base.is_relative_to(real_root):
            return _to_posix(os.path.relpath(real_path, real_base))

    return norm_path.as_posix()


def resolve_workspace_path(base_dir: Path | str, raw_path: Path | str) -> Path:
    """Resolve ``raw_path`` lexically against ``base_dir`` unless already absolute."""
    expanded = Path(os.path.expanduser(str(raw_path)))
    if expanded.is_absolute():
        return normalize_lexical_path(expanded)
    return normalize_lexical_path(Path(base_dir) / expanded)


def serialize_portable_value(
    value: Any,
    base_dir: Path | str,
    *,
    workspace_root: Path | str | None = None,
    sort_keys: bool = False,
    drop_none: bool = False,
) -> Any:
    """Recursively convert ``Path`` objects into workspace-portable strings.

    Relative ``Path`` values are resolved lexically against ``base_dir`` (never
    process ``os.getcwd()``) before computing workspace-relative POSIX paths.
    """
    norm_base = normalize_lexical_path(base_dir)
    norm_root = (
        normalize_lexical_path(workspace_root)
        if workspace_root is not None
        else None
    )
    return _serialize_portable_inner(
        value,
        norm_base,
        norm_root,
        sort_keys=sort_keys,
        drop_none=drop_none,
    )


def _serialize_portable_inner(
    value: Any,
    norm_base: Path,
    norm_root: Path | None,
    *,
    sort_keys: bool,
    drop_none: bool,
) -> Any:
    if isinstance(value, Path):
        resolved = resolve_workspace_path(norm_base, value)
        return workspace_relative_path(
            resolved, norm_base, workspace_root=norm_root
        )
    if isinstance(value, Mapping):
        items = (
            sorted(value.items(), key=lambda kv: str(kv[0]))
            if sort_keys
            else value.items()
        )
        return {
            str(k): _serialize_portable_inner(
                v,
                norm_base,
                norm_root,
                sort_keys=sort_keys,
                drop_none=drop_none,
            )
            for k, v in items
            if not (drop_none and v is None)
        }
    if isinstance(value, (list, tuple)):
        return [
            _serialize_portable_inner(
                item,
                norm_base,
                norm_root,
                sort_keys=sort_keys,
                drop_none=drop_none,
            )
            for item in value
        ]
    return value


def is_nonempty_file(path: Path | str) -> bool:
    """Return ``True`` iff ``path`` resolves to a regular file with size > 0."""
    p = Path(path)
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def all_nonempty(paths: Mapping[str, Path | str]) -> bool:
    """Return ``True`` when ``paths`` is non-empty and every entry is a non-empty file."""
    return bool(paths) and all(is_nonempty_file(p) for p in paths.values())


def read_log_tail(log_path: Path | str, limit: int = 1500) -> str:
    """Return up to the last ``limit`` characters of ``log_path`` (or ``""``)."""
    p = Path(log_path)
    if limit <= 0 or not p.is_file():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")[-limit:]


def compute_sha256(path: Path | str) -> str:
    """Compute the SHA-256 hex digest of ``path`` (following symlinks)."""
    with Path(path).open("rb") as fh:
        if hasattr(hashlib, "file_digest"):
            return hashlib.file_digest(fh, "sha256").hexdigest()
        digest = hashlib.sha256()
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
        return digest.hexdigest()


@dataclass(frozen=True)
class FileIdentity:
    """Content-addressed identity of an input or output file on disk."""

    path: Path
    size_bytes: int
    sha256: str

    @classmethod
    def capture(cls, path: Path | str) -> FileIdentity:
        """Capture lexical path, byte size, and SHA-256 digest of ``path``."""
        norm = normalize_lexical_path(path)
        if not norm.is_file():
            raise FileNotFoundError(f"File not found or not a regular file: {norm}")
        size_bytes = norm.stat().st_size
        return cls(path=norm, size_bytes=size_bytes, sha256=compute_sha256(norm))

    def matches_disk(self) -> bool:
        """Return ``True`` iff ``self.path`` exists with matching size and SHA-256."""
        if not self.path.is_file():
            return False
        try:
            st = self.path.stat()
        except OSError:
            return False
        if st.st_size != self.size_bytes:
            return False
        return compute_sha256(self.path) == self.sha256

    def to_dict(
        self,
        base_dir: Path | str,
        *,
        workspace_root: Path | str | None = None,
    ) -> dict[str, Any]:
        """Serialize with a workspace-portable relative path."""
        return {
            "path": workspace_relative_path(
                self.path, base_dir, workspace_root=workspace_root
            ),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, base_dir: Path | str, raw: Mapping[str, Any]) -> FileIdentity:
        """Reconstruct a ``FileIdentity`` from serialized manifest metadata."""
        return cls(
            path=resolve_workspace_path(base_dir, str(raw["path"])),
            size_bytes=int(raw["size_bytes"]),
            sha256=str(raw["sha256"]),
        )


def create_relative_symlink(
    source: Path | str,
    link_path: Path | str,
    *,
    overwrite_file: bool = False,
) -> Path:
    """Create a portable relative symlink at ``link_path`` pointing to ``source``.

    Existing symlinks are replaced atomically. Existing non-symlink regular
    files raise ``FileExistsError`` unless ``overwrite_file=True``. Directories
    are never overwritten.
    """
    src = normalize_lexical_path(source)
    dst = normalize_lexical_path(link_path)
    rel_target = _to_posix(os.path.relpath(src, dst.parent))

    if dst.is_symlink():
        try:
            if os.readlink(dst) == rel_target:
                return dst
        except OSError:
            pass
        replace_existing = True
    elif dst.exists():
        if dst.is_dir():
            raise FileExistsError(f"Cannot replace directory with symlink: {dst}")
        if not overwrite_file:
            raise FileExistsError(
                f"Refusing to overwrite existing regular file with symlink: {dst}"
            )
        replace_existing = True
    else:
        replace_existing = False

    dst.parent.mkdir(parents=True, exist_ok=True)
    if replace_existing:
        fd, tmp_name = tempfile.mkstemp(
            dir=dst.parent, prefix=f".{dst.name}.", suffix=".tmp"
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            tmp_path.unlink()
            tmp_path.symlink_to(rel_target)
            os.replace(tmp_path, dst)
        except BaseException:
            if tmp_path.is_symlink() or tmp_path.exists():
                tmp_path.unlink()
            raise
    else:
        dst.symlink_to(rel_target)
    return dst


def publish_artifact(
    source: Path | str,
    destination: Path | str,
    *,
    mode: Literal["symlink", "copy"] = "symlink",
) -> Path:
    """Publish a non-empty artifact atomically to ``destination``."""
    src = normalize_lexical_path(source)
    dst = normalize_lexical_path(destination)

    if not is_nonempty_file(src):
        raise FileNotFoundError(f"Artifact source is missing or empty: {src}")
    if src == dst:
        return dst
    if dst.is_dir() and not dst.is_symlink():
        raise IsADirectoryError(f"Destination is a directory: {dst}")

    if mode == "symlink":
        return create_relative_symlink(src, dst, overwrite_file=True)

    if mode == "copy":
        dst.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=dst.parent, prefix=f".{dst.name}.", suffix=".tmp"
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as out_fh, src.open("rb") as in_fh:
                shutil.copyfileobj(in_fh, out_fh)
                out_fh.flush()
                os.fsync(out_fh.fileno())
            os.chmod(tmp_path, 0o644)
            os.replace(tmp_path, dst)
        except BaseException:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        return dst

    raise ValueError(f"Unsupported publish mode {mode!r}; expected 'symlink' or 'copy'")


def atomic_write_json(path: Path | str, payload: Any) -> Path:
    """Write deterministic indented JSON atomically via ``fsync`` + ``os.replace``."""
    target = normalize_lexical_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = target.stat().st_mode & 0o777 if target.exists() else 0o644

    content = json.dumps(payload, indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, target)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return target


def read_json(path: Path | str) -> Any:
    """Read and parse a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
