"""Pluggable storage backend for run artifacts.

Today only LocalStorage is shipped. Cloud backends (S3/GCS) will land later
behind the same Storage interface without breaking existing configs.
"""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator


class Storage(ABC):
    """Storage interface used by every stage to publish artifacts.

    A stage hands the orchestrator {logical_name: local_path}; the orchestrator
    calls `put` to publish each path under the run's namespace, stores the
    returned URI in the StageResult, and downstream stages resolve via env
    var injection (CGML_DEP_<STAGE>_<ARTIFACT>) in SLURM mode.
    """

    @abstractmethod
    def put(self, local_path: Path, key: str) -> str: ...

    @abstractmethod
    def get(self, key: str, local_path: Path) -> Path: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def list(self, prefix: str) -> Iterator[str]: ...

    @abstractmethod
    def resolve(self, uri: str) -> Path: ...


class LocalStorage(Storage):
    """Local-filesystem backend. Symlinks rather than copies so multi-GB
    artifacts aren't duplicated."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _full(self, key: str) -> Path:
        # Reject ".." traversal so a misconfigured key can't write outside root.
        p = (self.root / key).resolve()
        if not str(p).startswith(str(self.root)):
            raise ValueError(f"key {key!r} escapes storage root {self.root}")
        return p

    def put(self, local_path: Path, key: str) -> str:
        local_path = Path(local_path).resolve()
        dst = self._full(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        os.symlink(local_path, dst)
        return str(dst)

    def get(self, key: str, local_path: Path) -> Path:
        src = self._full(key)
        if not src.exists():
            raise FileNotFoundError(src)
        return src

    def exists(self, key: str) -> bool:
        return self._full(key).exists()

    def list(self, prefix: str) -> Iterator[str]:
        base = self._full(prefix)
        if not base.exists():
            return
        for p in base.rglob("*"):
            yield str(p.relative_to(self.root))

    def resolve(self, uri: str) -> Path:
        return Path(uri)


def from_config(cfg) -> Storage:
    if cfg.backend == "local":
        return LocalStorage(root=cfg.root)
    raise ValueError(f"unknown storage backend: {cfg.backend}")
