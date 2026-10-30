"""Refuse a recorder that cannot state a run class.

An `rmlx` built before migration 9 has no `--run-class` on `metrics record`.
Handed a record that states one, it ignores the key, stores a row with no
class, and rebuilds `bests` without the class filter on its way in
(`docs/METRICS_DB.md` §13, rule 23). The flag's presence in the help text is
the fact the record depends on, so that is what is read.
"""

from __future__ import annotations

import subprocess


def require_run_class_recorder(rmlx_bin: str) -> None:
    """Exit naming `rmlx_bin` unless its `metrics record --help` names `--run-class`."""
    try:
        out = subprocess.run(
            [rmlx_bin, "metrics", "record", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise SystemExit(f"cannot run the recorder {rmlx_bin}: {exc}") from None
    if out.returncode != 0:
        raise SystemExit(
            f"`{rmlx_bin} metrics record --help` failed (exit {out.returncode}): "
            f"{out.stderr.strip()}"
        )
    if "--run-class" not in out.stdout:
        raise SystemExit(
            f"the recorder {rmlx_bin} has no `metrics record --run-class`: it was built "
            "before the run class, and would store these rows with no class and rebuild "
            "bests without the class filter (docs/METRICS_DB.md §13, rule 23). Pass "
            "--rmlx-bin with a binary built at or after migration 9."
        )
