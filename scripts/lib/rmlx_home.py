"""Resolve the runtime home the way the binary does."""

from __future__ import annotations

import os
from pathlib import Path


def rmlx_home(repo_root: Path) -> Path:
    """`$RMLX_HOME` when set, which must be absolute, else `<repo_root>/.rmlx`."""
    raw = os.environ.get("RMLX_HOME")
    if raw is None:
        return repo_root / ".rmlx"
    if not Path(raw).is_absolute():
        raise SystemExit(f"RMLX_HOME={raw} must be an absolute path")
    return Path(raw)
