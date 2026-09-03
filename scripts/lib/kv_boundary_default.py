"""The engine's default KV boundary-layer counts, read from the engine.

`rmlx_core::kv_boundary` is the one definition of these numbers — it exists so
that `rmlx-models` (which applies them) and `rmlx-metrics` (which has to
recognise a `decode_config` spelling them out) cannot drift apart. A Python
copy would be a third spelling, and a wrong one is not visible: it files a
non-default sweep under the default cell, permanently, in an append-only table.

So this reads the constants rather than restating them, and raises when it
cannot. A parse that silently falls back to a literal would be the copy again.

`scripts/check_kv_boundary_default_parity.sh` (in `make ci`) checks the other
direction: that the CLI help and `docs/CLI.md` still name the same pair.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_REPO_ROOT = Path(
    os.environ.get("RMLX_REPO_ROOT", str(Path(__file__).resolve().parents[2]))
)
SOURCE = _REPO_ROOT / "crates" / "rmlx-core" / "src" / "kv_boundary.rs"

_CONST = "pub const DEFAULT_BOUNDARY_{}_N: usize = (?P<value>[0-9]+);"


def _const(name: str, text: str) -> int:
    match = re.search(_CONST.format(name), text)
    if match is None:
        raise SystemExit(
            f"cannot find DEFAULT_BOUNDARY_{name}_N in {SOURCE}. The engine's "
            "boundary default is defined there and is not restated here; fix "
            "the path or the constant's name rather than hard-coding a value."
        )
    return int(match.group("value"))


def kv_boundary_default() -> tuple[int, int]:
    """`(head_n, tail_n)` as the engine ships them."""
    try:
        text = SOURCE.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"cannot read {SOURCE}: {exc}") from exc
    return _const("HEAD", text), _const("TAIL", text)
