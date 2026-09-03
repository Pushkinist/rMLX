#!/usr/bin/env python3
"""Ingest a `scripts/perf_ab.sh` result into the rMLX metrics DB.

`perf_ab.sh` deliberately never touches `runs.db` — an A/B run is an experiment
and the store is append-only. This is the separate, explicit step that promotes
one accepted comparison into two §8.5 RunRecords, one per arm.

Sibling of `llama_ab_ingest.py`, which does the same job for
`bench_llama_ab.sh`. The two inputs are different shapes and different
backends: this one reads `rmlx baseline` slots, so the identity block comes
from the measured binary itself (`rmlx metrics identity --json`) instead of
being caller-supplied, and the residency column is `kv_cache_bytes` off the
harness's own per-slot parse rather than a server's KV-buffer log line.

Usage:
    # Inspect the records without writing anything:
    python3 scripts/ingest/perf_ab_ingest.py --dry-run RESULT.json \
        --model Qwen3.8-27B-mxfp8 --weight-quant mxfp8 \
        --arm-a-kv-quant none --arm-b-kv-quant mixed_k8g64_v4g64 \
        --prompt-tokens 130848

    # Write buffer files and ingest them:
    python3 scripts/ingest/perf_ab_ingest.py --record RESULT.json ...

A run the harness marked TAINTED is refused unless `--accept-tainted` is given,
and the taint text is then carried into `notes`: on a host where the quiescence
gate never clears, an unmarked row is a claim of a clean measurement that was
never made. A run made with `--synthetic-arms` is refused with no waiver at
all: its arms were stub binaries, so there is no measurement in it to accept.

`--arm-*-kv-quant` is required and then *checked* against the `--kv-quant` in
that arm's recorded arguments. It is not parsed out of them: a result whose arm
carries no `--kv-quant` at all is a different failure from one that carries a
different codec, and collapsing the two is how a row lands under the wrong cell
key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent / "lib"))
from kv_boundary_default import kv_boundary_default  # noqa: E402

RMLX_REPO_ROOT = Path(os.environ.get("RMLX_REPO_ROOT", str(_SCRIPT_DIR.parents[1])))
# Runtime state lives under a single root (CLAUDE.md). `RMLX_HOME` wins; the
# in-repo `.rmlx/` is the dev default.
RMLX_HOME = Path(os.environ.get("RMLX_HOME", str(RMLX_REPO_ROOT / ".rmlx")))
BUFFER_PENDING = RMLX_HOME / "metrics" / "buffer" / "pending"
# Records are assembled HERE first and only moved into `pending/` for the
# moment the recorder is actually invoked. Anything sitting in `pending/` is,
# by contract, work the next `rmlx metrics record --replay-pending` will claim
# — and that sweep quarantines whatever it cannot ingest into `buffer/failed/`.
STAGING = RMLX_HOME / "bench" / "perf_ab" / "records"

_KV_FLAG = re.compile(r"--kv-quant[= ]+(\S+)")
_KV_BOUNDARY_FLAG = re.compile(r"--kv-boundary-layers[= ]+(\d+),(\d+)")


def _decode_config_of(arm_args: str) -> str | None:
    """The `decode_config` cell term this arm's own arguments imply.

    Read off the recorded argument string rather than taken from a flag on this
    script: the arguments are what ran, and an operator-typed term can describe
    a configuration the slot never used. `None` is the engine at its defaults,
    which is what keeps a default-boundary arm ranking against every row
    recorded before the flag existed.

    The LAST occurrence wins, matching how clap resolves a repeated flag.
    """
    matches = _KV_BOUNDARY_FLAG.findall(arm_args or "")
    if not matches:
        return None
    head, tail = (int(x) for x in matches[-1])
    if (head, tail) == kv_boundary_default():
        return None
    return f"kv_boundary/head={head},kv_boundary/tail={tail}"


def _kv_quant_of(arm_args: str, declared: str, arm: str) -> str:
    """Check the declared codec against the arm's recorded arguments.

    The LAST occurrence wins, because that is how clap resolves a repeated
    flag: taking the first would confirm a codec the slot did not run.
    """
    matches = _KV_FLAG.findall(arm_args or "")
    m = matches[-1] if matches else None
    if not m:
        raise SystemExit(
            f"arm {arm} recorded no --kv-quant in its arguments ({arm_args!r}), so "
            f"the declared '{declared}' cannot be confirmed. Both arms of a codec "
            "comparison must name their codec explicitly."
        )
    if m != declared:
        raise SystemExit(
            f"arm {arm} ran --kv-quant {m} but --arm-{arm.lower()}-kv-quant "
            f"says '{declared}'. The cell key would not describe the measurement."
        )
    return declared


def _kv_bytes(stats: dict[str, Any]) -> tuple[int | None, str]:
    """Resident KV for one arm, in bytes, plus a provenance note.

    `perf_ab.sh` carries the exact byte count. A handful of result files on this
    machine carry only the report table's 0.1 MB display field, because they
    were produced by an intermediate revision of the harness that was never
    committed — no released or committed version of `perf_ab.sh` emits that
    shape. They are still ingestable, and the row then states where its
    precision came from rather than presenting a rounded number as an exact one.
    """
    exact = stats.get("median_kv_cache_bytes")
    if exact is not None:
        return int(exact), ""
    rounded = stats.get("median_kv_cache_mb")
    if rounded is None:
        return None, ""
    return (
        round(rounded * 1e6),
        "kv_cache_bytes derived from the harness report table's 0.1 MB display "
        "field (this result predates median_kv_cache_bytes), so it is exact to "
        "+/-50 kB, not to the byte",
    )


def _verify_binary(binary: str, sha256_16: str, arm: str) -> None:
    """Refuse when the binary on disk is not the one the run measured.

    `_identity` asks a PATH who it is, and the run recorded a digest. Between
    the run and the ingest that path can be rebuilt, and then `backend_version`
    / `build_profile` describe one binary while the row's notes assert another
    — permanently, in an append-only table. The digest is the only thing that
    ties the two together, so it is checked rather than pasted.
    """
    path = Path(binary)
    if not path.is_file():
        raise SystemExit(
            f"arm {arm}'s binary {binary} no longer exists, so its identity "
            f"(recorded digest sha256:{sha256_16}) cannot be confirmed."
        )
    actual = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    if actual != sha256_16:
        raise SystemExit(
            f"arm {arm}'s binary {binary} now hashes sha256:{actual}, but the run "
            f"recorded sha256:{sha256_16}. It was rebuilt after the measurement, so "
            "its version and build profile no longer describe the numbers in this "
            "result. Rebuild at the measured commit, or re-run the comparison."
        )


def _identity(binary: str) -> dict[str, Any]:
    """§8.5 identity block, asked of the binary that produced the numbers."""
    out = subprocess.run(
        [binary, "metrics", "identity", "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise SystemExit(f"'{binary} metrics identity --json' failed:\n{out.stderr}")
    # The command logs to stderr and prints one JSON object to stdout.
    return json.loads(out.stdout.strip().splitlines()[-1])


def _arm_record(
    result: dict[str, Any],
    cell: dict[str, Any],
    arm: str,
    args: argparse.Namespace,
    prompt_body: str,
    prompt_tokens: int,
    prompt_name: str,
) -> dict[str, Any]:
    """Build one §8.5 RunRecord from one arm of an A/B result."""
    key = "arm_a" if arm == "A" else "arm_b"
    stats = cell[key]
    top = result[key]
    declared = args.arm_a_kv_quant if arm == "A" else args.arm_b_kv_quant
    kv_quant = _kv_quant_of(top.get("args", ""), declared, arm)

    kv_bytes, kv_note = _kv_bytes(stats)
    tainted = cell.get("taint") or ""

    # The interleave pattern is read from the result file, never asserted: an
    # `--invert` leg is BAAB and a row that calls it ABBA describes a protocol
    # the run did not follow. A result file written before the field existed
    # says so rather than being given a plausible value.
    pattern = result.get("pattern") or "pattern-unrecorded"
    notes = (
        f"perf_ab.sh {pattern} n={stats['n']}/arm; decode_tps median over slots, "
        f"min={stats['min_tps']:.3f} max={stats['max_tps']:.3f}; "
        f"verdict={cell['verdict']}; "
        f"paired arm {top['label']!r} vs "
        f"{result['arm_b' if arm == 'A' else 'arm_a']['label']!r}; "
        f"binary sha256:{top['sha256_16']}"
    )
    if tainted:
        notes += f"; TAINTED: {tainted}"
    waived = sorted(k for k, v in (result.get("waivers") or {}).items() if v)
    if waived:
        notes += f"; guards waived: {', '.join(waived)}"

    metrics: list[dict[str, Any]] = [
        {
            "name": "decode_tps_warm",
            "value": stats["median_tps"],
            "stddev": stats["sd_tps"] if stats["n"] > 1 else None,
        }
    ]
    # `null` is the supported way to say "not measured"; a 0 here would rank in
    # `bests` as the smallest cache ever recorded.
    metrics.append({"name": "kv_cache_bytes", "value": kv_bytes})
    if kv_note:
        notes += f"; {kv_note}"

    _verify_binary(top["binary"], top["sha256_16"], arm)
    return {
        **_identity(top["binary"]),
        "schema_version": 1,
        "model_namespace": args.model_namespace,
        "model": args.model,
        "weight_quant": args.weight_quant,
        "kv_quant": kv_quant,
        "decode_config": _decode_config_of(top.get("args", "")),
        "ctx_max": result["shape"]["max_ctx"],
        "prompt": {
            "name": prompt_name,
            "body": prompt_body,
            "notes": args.prompt_notes,
        },
        "ts_utc": _iso(result["ts_utc"]),
        "git_sha": args.git_sha,
        "prompt_tokens": prompt_tokens,
        "max_tokens": result["shape"]["max_tokens"],
        "temperature": 0.0,
        "seed": 0,
        "n_warmups": 1,
        "n_measure": stats["n"],
        "notes": notes,
        "description": args.description,
        "metrics": metrics,
    }


def _iso(stamp: str) -> str:
    """`perf_ab.sh` stamps `YYYYMMDDTHHMMSSZ`; §8.5 wants ISO-8601."""
    if re.fullmatch(r"\d{8}T\d{6}Z", stamp):
        return (
            f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]}T"
            f"{stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}Z"
        )
    return stamp


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result", help="JSON emitted by perf_ab.sh")
    p.add_argument("--model", required=True, help="§5.1 model label")
    p.add_argument("--model-namespace", default="mlx-community")
    p.add_argument("--weight-quant", required=True, help="§5.2 whitelist value")
    p.add_argument("--arm-a-kv-quant", required=True)
    p.add_argument("--arm-b-kv-quant", required=True)
    p.add_argument(
        "--prompt-tokens",
        type=int,
        default=None,
        help="tokenized prompt length. The harness measures it and this becomes a "
        "cross-check that errors on mismatch; it is only required for a result "
        "written before the harness recorded one. Note the fixture's NAME "
        "(--prompt-tokens 131072) is not its token count.",
    )
    p.add_argument("--prompt-name", default=None)
    p.add_argument("--prompt-file", default=None)
    p.add_argument("--prompt-notes", default=None)
    p.add_argument(
        "--git-sha",
        default=None,
        help="caller-supplied provenance (§8.5.1); the binary does no git.",
    )
    p.add_argument("--description", default=None)
    p.add_argument("--accept-tainted", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--record", action="store_true", help="ingest via `rmlx metrics record --file`"
    )
    p.add_argument(
        "--rmlx-bin", default=str(RMLX_REPO_ROOT / "target" / "release" / "rmlx")
    )
    args = p.parse_args()

    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    cells = result["results"]
    if len(cells) != 1:
        raise SystemExit(
            f"{args.result} compares {len(cells)} models in one run. The model "
            "label, namespace and weight quant are per model and this script "
            "takes one of each — re-run perf_ab.sh with a single --model, or "
            "ingest each cell from its own result file."
        )
    cell = cells[0]

    # A run made with --synthetic-arms drove stub binaries, so its numbers
    # describe nothing that exists. That is not a host condition anyone can
    # choose to accept, so this refusal has no waiver -- unlike taint, which is
    # a real measurement taken under interference.
    waivers = result.get("waivers") or {}
    if waivers.get("synthetic_arms"):
        print(
            "refusing: the run was made with --synthetic-arms, so its arms were "
            "stub binaries\n"
            "  and its numbers measure nothing. There is no waiver for this: "
            "re-run the\n"
            "  comparison against real binaries.",
            file=sys.stderr,
        )
        return 2

    if cell.get("taint") and not args.accept_tainted:
        print(
            f"refusing: run is TAINTED ({cell['taint']}).\n"
            "  Re-run on a quiet host, or pass --accept-tainted to record it with\n"
            "  the taint carried into `notes`.",
            file=sys.stderr,
        )
        return 2

    # A raised --busy-pct does not taint: it removes the gate that would have
    # tainted. The result then looks clean for the one reason a result must
    # never look clean, and the taint check above cannot see it.
    if waivers.get("busy_pct_raised") and not args.accept_tainted:
        print(
            "refusing: the run raised --busy-pct above the default, so the "
            "interference\n"
            "  gate was weakened and an empty taint means the gate did not fire, "
            "not that\n"
            "  the host was quiet. Re-run at the default threshold, or pass "
            "--accept-tainted.",
            file=sys.stderr,
        )
        return 2

    # Prompt: the harness names a canonical fixture by requested size, and the
    # fixture's rendered token count is a different number.
    requested = result["shape"]["prompt_tokens"]
    prompt_name = args.prompt_name or f"longctx_{requested // 1024}k"
    prompt_path = (
        Path(args.prompt_file)
        if args.prompt_file
        else RMLX_REPO_ROOT / "prompts" / f"{prompt_name}.json"
    )
    if not prompt_path.is_file():
        raise SystemExit(f"no prompt body at {prompt_path} (pass --prompt-file)")
    prompt_body = prompt_path.read_text(encoding="utf-8")

    measured = cell.get("prompt_tokens")
    if measured is None:
        if args.prompt_tokens is None:
            raise SystemExit(
                "this result carries no measured prompt_tokens (it predates the "
                "harness recording one). Pass --prompt-tokens with the count the "
                "run's `baseline: ... prompt_tokens=` line reported; the fixture's "
                "requested size is not it."
            )
        prompt_tokens = args.prompt_tokens
    else:
        # The harness measured it. A supplied value is then a cross-check, never
        # an override: the cell key must be what ran, not what was expected.
        if args.prompt_tokens is not None and args.prompt_tokens != measured:
            raise SystemExit(
                f"--prompt-tokens {args.prompt_tokens} disagrees with the "
                f"{measured} the run tokenized. The recorded cell key would not "
                "describe the measurement."
            )
        prompt_tokens = measured

    records = [
        _arm_record(result, cell, arm, args, prompt_body, prompt_tokens, prompt_name)
        for arm in ("A", "B")
    ]

    if args.dry_run:
        for rec in records:
            trimmed = dict(rec)
            trimmed["prompt"] = dict(rec["prompt"], body=f"<{len(prompt_body)} chars>")
            print(json.dumps(trimmed, indent=2))
        return 0

    stamp = result["ts_utc"].replace("-", "").replace(":", "")
    STAGING.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for rec in records:
        path = STAGING / f"{stamp}-{rec['kv_quant']}-{uuid.uuid4().hex[:8]}.json"
        path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        staged.append(path)
        print(f"staged {path}")

    if not args.record:
        print(
            "not ingested (pass --record). The files above are OUTSIDE\n"
            f"  {BUFFER_PENDING},\n"
            "  so no --replay-pending sweep can claim them. Inspect, then re-run\n"
            "  with --record."
        )
        return 0

    # Move into the live queue one file at a time, immediately before the
    # recorder claims it, and take it back out either way.
    BUFFER_PENDING.mkdir(parents=True, exist_ok=True)
    failed = 0
    for path in staged:
        queued = BUFFER_PENDING / path.name
        path.rename(queued)
        try:
            proc = subprocess.run(
                [args.rmlx_bin, "metrics", "record", "--file", str(queued)], check=False
            )
        finally:
            if queued.exists():
                queued.rename(path)
        if proc.returncode != 0:
            print(
                f"ingest FAILED for {path} (exit {proc.returncode}); "
                "record kept in staging, not in the pending queue",
                file=sys.stderr,
            )
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
