#!/usr/bin/env python3
"""Wikitext-2 perplexity harness for rMLX.

Downloads the wikitext-2-raw test split, persists it to a temp file, and
invokes ``rmlx eval ppl`` to compute sliding-window perplexity using the
native scorer in ``rmlx_models::ppl``.

Why a CLI wrapper, not an HTTP call:
    Option A (HTTP ``echo``+``logprobs`` with ``max_tokens=0``) would require
    exposing per-prompt-position logits across every architecture's forward
    path — engine refactor out of scope — so this uses Option B: a standalone
    ``rmlx eval ppl`` subcommand.

Dependencies:
    Python 3.9+ standard library only (``urllib``, ``zipfile``, ``json``,
    ``subprocess``, ``tempfile``, ``argparse``, ``pathlib``).  No ``requests``,
    no ``datasets``.

Exit codes:
    0  success -- prints the rMLX JSON line on stdout
    1  argument / I/O / subprocess error
    2  PPL value outside the plausible band (sanity gate, opt-in via
       ``--plausibility``)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Direct mirror of the wikitext-2-raw test split.  The original Salesforce S3
# bucket now returns a permanent 301 with no Location header, so we resolve
# against cosmo.zip's stable mirror.  Pre-extracted raw file -- no zip.
WIKITEXT2_TEST_URL = "https://cosmo.zip/pub/datasets/wikitext-2-raw/wiki.test.raw"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run wikitext-2 perplexity against an rMLX model via "
        "the `rmlx eval ppl` CLI subcommand.",
    )
    p.add_argument(
        "--server-url",
        default="",
        help="Reserved for the follow-up HTTP path (currently unused; "
        "Option B invokes the CLI directly). Kept for forward-compatibility.",
    )
    p.add_argument(
        "--model",
        required=True,
        help="Absolute path to an MLX model snapshot directory "
        "(Qwen3, Gemma4 or Qwen3.5).",
    )
    p.add_argument(
        "--ctx-window",
        type=int,
        default=4096,
        help="Sliding-window length in tokens. Default 4096.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=2048,
        help="Window stride. Default 2048 (50%% overlap).",
    )
    p.add_argument(
        "--corpus",
        default="wikitext-2",
        choices=["wikitext-2"],
        help="Corpus identifier. Only wikitext-2 is supported.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Optional cap on the corpus token count (0 = no cap). Used by "
        "the smoke run to keep wall-clock manageable on small models.",
    )
    p.add_argument(
        "--device",
        default="gpu",
        choices=["gpu", "cpu"],
        help="Device passed to `rmlx eval ppl`. Default gpu.",
    )
    p.add_argument(
        "--rmlx-binary",
        default="",
        help="Path to the rmlx binary. Default: auto-detect "
        "(`./target/release-perf/rmlx` then `./target/release/rmlx` then "
        "`rmlx` on PATH).",
    )
    p.add_argument(
        "--cache-dir",
        default="",
        help="Cache directory for the downloaded wikitext-2 archive. Default: "
        "`$RMLX_HOME/cache/wikitext-2/` or `$HOME/.rmlx/cache/wikitext-2/`.",
    )
    p.add_argument(
        "--git-sha",
        default="",
        help="Commit SHA stamped on the emitted record's `git_sha` column. "
        "Provenance the caller supplies -- the binary does not derive it, so "
        "an omitted flag leaves the column NULL.",
    )
    p.add_argument(
        "--plausibility",
        action="store_true",
        help="Gate the result: exit 2 when PPL is outside (1.5, 50.0).",
    )
    return p.parse_args()


def resolve_cache_dir(arg_cache: str) -> Path:
    if arg_cache:
        return Path(arg_cache).expanduser().resolve()
    rmlx_home = os.environ.get("RMLX_HOME")
    if rmlx_home:
        return Path(rmlx_home).expanduser() / "cache" / "wikitext-2"
    return Path.home() / ".rmlx" / "cache" / "wikitext-2"


def download_wikitext2_test(cache_dir: Path) -> Path:
    """Download (or load from cache) the wikitext-2-raw test split.

    Returns the path to a UTF-8 text file containing the full split.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    text_path = cache_dir / "wiki.test.raw.txt"
    if text_path.exists() and text_path.stat().st_size > 0:
        print(
            f"wikitext-2: cache hit at {text_path} "
            f"({text_path.stat().st_size} bytes)",
            file=sys.stderr,
        )
        return text_path
    print(
        f"wikitext-2: downloading {WIKITEXT2_TEST_URL} -> {text_path}",
        file=sys.stderr,
    )
    try:
        with urllib.request.urlopen(WIKITEXT2_TEST_URL, timeout=60) as resp:
            data = resp.read()
    except urllib.error.URLError as e:
        print(f"wikitext-2: download failed: {e}", file=sys.stderr)
        raise
    text_path.write_bytes(data)
    print(
        f"wikitext-2: wrote {text_path} ({len(data)} bytes)",
        file=sys.stderr,
    )
    return text_path


def resolve_rmlx_binary(arg_binary: str) -> Path:
    if arg_binary:
        p = Path(arg_binary).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"rmlx binary not found: {p}")
        return p
    workspace = Path(__file__).resolve().parents[2]
    for candidate in (
        workspace / "target" / "release-perf" / "rmlx",
        workspace / "target" / "release" / "rmlx",
        workspace / "target" / "debug" / "rmlx",
    ):
        if candidate.exists():
            return candidate
    # Fall back to PATH lookup.
    from shutil import which

    on_path = which("rmlx")
    if on_path:
        return Path(on_path)
    raise FileNotFoundError(
        "rmlx binary not found in ./target/{release-perf,release,debug}/ "
        "or on PATH; pass --rmlx-binary"
    )


def run_ppl(args: argparse.Namespace, text_path: Path, rmlx: Path) -> dict:
    cmd = [
        str(rmlx),
        "eval",
        "ppl",
        "--model",
        str(args.model),
        "--text-file",
        str(text_path),
        "--ctx-window",
        str(args.ctx_window),
        "--stride",
        str(args.stride),
        "--corpus",
        args.corpus,
        "--device",
        args.device,
        "--max-tokens",
        str(args.max_tokens),
    ]
    if args.git_sha:
        cmd += ["--git-sha", args.git_sha]
    print(f"wikitext-2: running: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(
            f"wikitext-2: rmlx eval ppl failed (exit {proc.returncode}):\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    # The CLI emits informational tracing to stderr and one JSON line on
    # stdout.  Parse the last non-empty stdout line.
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        print(
            "wikitext-2: rmlx eval ppl produced no stdout JSON",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return json.loads(lines[-1])


def main() -> int:
    args = parse_args()
    if args.server_url:
        print(
            "wikitext-2: --server-url is reserved for the follow-up "
            "HTTP path; this script invokes the CLI directly. Ignoring.",
            file=sys.stderr,
        )
    cache_dir = resolve_cache_dir(args.cache_dir)
    text_path = download_wikitext2_test(cache_dir)
    rmlx = resolve_rmlx_binary(args.rmlx_binary)
    result = run_ppl(args, text_path, rmlx)
    # Forward the rMLX JSON line to stdout verbatim so calling scripts can
    # parse one line, full audit fields included.
    print(json.dumps(result))
    ppl = float(result.get("ppl", float("nan")))
    print(
        f"wikitext-2: PPL = {ppl:.4f} over {result.get('scored_tokens')} "
        f"tokens in {result.get('windows')} window(s); "
        f"ctx_window={result.get('ctx_window')} stride={result.get('stride')}",
        file=sys.stderr,
    )
    if args.plausibility:
        if not (1.5 < ppl < 50.0):
            print(
                f"wikitext-2: PPL {ppl} outside plausible band (1.5, 50.0); "
                "expected range is (1.5, 50.0).",
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
