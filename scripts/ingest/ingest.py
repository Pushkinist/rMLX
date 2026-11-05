#!/usr/bin/env python3
"""Promote a harness result into §8.5 records and file them with `rmlx metrics record`.

Usage:  ingest.py ab [--harness rmlx|llama] RESULT.json [flags]
        ingest.py codec-inertness CSV --backend-version V --build-profile P --models-root DIR [flags]
        ingest.py llama-bench FILE|- [flags]
        ingest.py published RESULT.json [flags]
        ingest.py <mode> -h lists the flags of one mode
Exit:   0 records written, printed or filed; 1 the recorder refused a record, or the input
        cannot be read or is refused; 2 a usage error, a result the mode refuses to record,
        or a CSV with no usable row
Not a mode of: bench/codec_inertness_probe.sh — that one runs a model and measures; this one
        promotes a finished result into records

`ab` takes a `perf_ab.sh` result (`--harness rmlx`, the default) or a `bench_llama_ab.sh` result
(`--harness llama`). Both harnesses write each slot as an `ab_arm` row, which reaches no champion
and no gate; `--keep-arm A|B` promotes the surviving arm as one `gate` record. The llama harness
runs this itself, without `--keep-arm`, after every real run: both arms are then `ab_arm`.

`ab`, `codec-inertness` and `published` stage each record outside `buffer/pending/` and move it in
only for the moment the recorder claims it: anything left in `pending/` is work the next
`--replay-pending` sweep takes and quarantines when it cannot ingest it. `llama-bench` queues in
`pending/`. `--dry-run` prints the records and writes nothing; `--record` files them through
`--rmlx-bin`, which must state a run class.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "lib"))
import llama_bench  # noqa: E402
import published  # noqa: E402
from kv_boundary_default import kv_boundary_default  # noqa: E402
from recorder_check import require_run_class_recorder  # noqa: E402
from rmlx_home import rmlx_home  # noqa: E402
from snapshot_identity import (  # noqa: E402
    UnlabelledSnapshot,
    snapshot_weight_quant,
    split_model_dir,
)

BENCH_DIR = rmlx_home(REPO_ROOT) / "bench"
BUFFER_PENDING = rmlx_home(REPO_ROOT) / "metrics" / "buffer" / "pending"
LLAMA_MIB = 1024 * 1024
DEFAULT_RMLX_BIN = REPO_ROOT / "target" / "release" / "rmlx"


# ── the tail every staging mode shares ────────────────────────────────────────


def boundary_term(head: int, tail: int) -> str | None:
    """The `decode_config` cell term for a head/tail boundary; `None` at the engine's default.

    `None` is what keeps a default-boundary row ranking against every row recorded before the
    flag existed.
    """
    if (head, tail) == kv_boundary_default():
        return None
    return f"kv_boundary/head={head},kv_boundary/tail={tail}"


def print_dry_run(records: list[dict[str, Any]]) -> None:
    for rec in records:
        body = rec["prompt"]["body"]
        size = len(body) if isinstance(body, str) else len(json.dumps(body))
        print(json.dumps(dict(rec, prompt=dict(rec["prompt"], body=f"<{size} chars>")), indent=2))


def announce_each(staged: list[Path]) -> None:
    for path in staged:
        print(f"staged {path}")


def stage_records(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    staging: Path,
    name_key: str | None,
    announce: Callable[[list[Path]], None],
    compact: bool = False,
) -> list[Path] | None:
    """Print (`--dry-run`) or stage `records`; the staged paths when they are to be recorded.

    A file is named `<ts_utc>[-<record[name_key]>]-<8 hex>.json`. `None` means nothing is left
    to record: a dry run, or staged without `--record`.
    """
    if args.dry_run:
        print_dry_run(records)
        return None
    if args.record:
        require_run_class_recorder(args.rmlx_bin)

    staging.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for rec in records:
        stem = rec["ts_utc"].replace("-", "").replace(":", "")
        if name_key:
            stem += f"-{rec[name_key]}"
        path = staging / f"{stem}-{uuid.uuid4().hex[:8]}.json"
        text = json.dumps(rec, ensure_ascii=False) if compact else json.dumps(rec, indent=2)
        path.write_text(text, encoding="utf-8")
        staged.append(path)
    announce(staged)

    if not args.record:
        print(
            "not ingested (pass --record). The files above are OUTSIDE\n"
            f"  {BUFFER_PENDING},\n"
            "  so no --replay-pending sweep can claim them. Inspect, then re-run\n"
            "  with --record."
        )
        return None
    return staged


def record_staged(staged: list[Path], rmlx_bin: str, *, capture: bool = False) -> list[Path]:
    """File each staged record through the recorder; the paths it refused.

    A record is moved into the queue one at a time, immediately before the recorder claims it,
    and taken back out either way: the recorder deletes it on success, and on failure it is ours
    to retrieve, not the next sweep's to quarantine.
    """
    BUFFER_PENDING.mkdir(parents=True, exist_ok=True)
    failed: list[Path] = []
    for path in staged:
        queued = BUFFER_PENDING / path.name
        path.rename(queued)
        try:
            proc = subprocess.run(
                [rmlx_bin, "metrics", "record", "--file", str(queued)],
                check=False,
                capture_output=capture,
                text=True,
            )
        finally:
            if queued.exists():
                queued.rename(path)
        if proc.returncode != 0:
            detail = f": {proc.stderr.strip()}" if capture else ""
            print(
                f"ingest FAILED for {path} (exit {proc.returncode}){detail}; "
                "record kept in staging, not in the pending queue",
                file=sys.stderr,
            )
            failed.append(path)
    return failed


# ── ab ────────────────────────────────────────────────────────────────────────

_KV_FLAG = re.compile(r"--kv-quant[= ]+(\S+)")
_KV_BOUNDARY_FLAG = re.compile(r"--kv-boundary-layers[= ]+(\d+),(\d+)")


def run_ab(argv: list[str]) -> int:
    chooser = argparse.ArgumentParser(prog="ingest.py ab", add_help=False, allow_abbrev=False)
    chooser.add_argument("--harness", choices=("rmlx", "llama"), default="rmlx")
    chosen, rest = chooser.parse_known_args(argv)
    run = ab_rmlx if chosen.harness == "rmlx" else ab_llama
    return run(rest)


def ab_rmlx(argv: list[str]) -> int:
    """Promote the surviving arm of a `perf_ab.sh` result into one `gate` record.

    The result is `rmlx baseline` slots, so the identity block comes from the measured binary
    itself (`rmlx metrics identity --json`) and the residency column is the harness's own
    per-slot `kv_cache_bytes`. The other arm is not written again: its slots are its record.

    A run the harness marked TAINTED is refused unless `--accept-tainted` is given, and the
    taint text is then carried into `notes`: on a host where the quiescence gate never clears,
    an unmarked row is a claim of a clean measurement that was never made. A run made with
    `--synthetic-arms` is refused with no waiver at all: its arms were stub binaries.

    `--arm-*-kv-quant` is required and then checked against the `--kv-quant` in that arm's
    recorded arguments, not parsed out of them: a result whose arm carries no `--kv-quant` is a
    different failure from one that carries a different codec, and collapsing the two is how a
    row lands under the wrong cell key.
    """
    p = argparse.ArgumentParser(prog="ingest.py ab --harness rmlx", description=ab_rmlx.__doc__)
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
    p.add_argument(
        "--keep-arm",
        required=True,
        choices=["A", "B"],
        help="the surviving arm, written as one gate record (docs/METRICS_DB.md "
        "§8.5). Both arms' slots are already in runs.db as ab_arm rows.",
    )
    p.add_argument("--accept-tainted", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--record", action="store_true", help="ingest via `rmlx metrics record --file`"
    )
    p.add_argument("--rmlx-bin", default=str(DEFAULT_RMLX_BIN))
    args = p.parse_args(argv)

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

    # No waiver: unlike taint, a stub-armed run is not a real measurement taken under
    # interference.
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

    # A raised --busy-pct does not taint: it removes the gate that would have tainted.
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

    # The harness names a canonical fixture by requested size, and the fixture's rendered
    # token count is a different number.
    requested = result["shape"]["prompt_tokens"]
    prompt_name = args.prompt_name or f"longctx_{requested // 1024}k"
    prompt_path = (
        Path(args.prompt_file)
        if args.prompt_file
        else REPO_ROOT / "prompts" / f"{prompt_name}.json"
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
        # A supplied value is a cross-check, never an override: the cell key must be what
        # ran, not what was expected.
        if args.prompt_tokens is not None and args.prompt_tokens != measured:
            raise SystemExit(
                f"--prompt-tokens {args.prompt_tokens} disagrees with the "
                f"{measured} the run tokenized. The recorded cell key would not "
                "describe the measurement."
            )
        prompt_tokens = measured

    # The kept row's notes name the arm it was paired with, so both declared codecs are held
    # to the arguments that ran.
    for arm, declared in (("A", args.arm_a_kv_quant), ("B", args.arm_b_kv_quant)):
        _kv_quant_of(result["arm_a" if arm == "A" else "arm_b"].get("args", ""), declared, arm)
    records = [
        _perf_ab_record(result, cell, args.keep_arm, args, prompt_body, prompt_tokens, prompt_name)
    ]

    staged = stage_records(
        records,
        args,
        staging=BENCH_DIR / "perf_ab" / "records",
        name_key="kv_quant",
        announce=announce_each,
    )
    if staged is None:
        return 0
    return 1 if record_staged(staged, args.rmlx_bin) else 0


def _decode_config_of(arm_args: str) -> str | None:
    """The `decode_config` cell term this arm's own arguments imply.

    Read off the recorded argument string rather than taken from a flag on this script: the
    arguments are what ran, and an operator-typed term can describe a configuration the slot
    never used. The LAST occurrence wins, matching how clap resolves a repeated flag.
    """
    matches = _KV_BOUNDARY_FLAG.findall(arm_args or "")
    if not matches:
        return None
    head, tail = (int(x) for x in matches[-1])
    return boundary_term(head, tail)


def _kv_quant_of(arm_args: str, declared: str, arm: str) -> str:
    """Check the declared codec against the arm's recorded arguments.

    The LAST occurrence wins, because that is how clap resolves a repeated flag: taking the
    first would confirm a codec the slot did not run.
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

    A result file written by a harness revision that emitted only the report table's 0.1 MB
    display field is still ingestable, and its row then states where its precision came from.
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

    `_identity` asks a path who it is, and the run recorded a digest. Between the run and the
    ingest the path can be rebuilt, and then `backend_version` / `build_profile` describe one
    binary while the row's notes assert another, permanently. The digest ties the two together.
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


def _iso(stamp: str) -> str:
    """`perf_ab.sh` stamps `YYYYMMDDTHHMMSSZ`; §8.5 wants ISO-8601."""
    if re.fullmatch(r"\d{8}T\d{6}Z", stamp):
        return (
            f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]}T"
            f"{stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}Z"
        )
    return stamp


def _perf_ab_record(
    result: dict[str, Any],
    cell: dict[str, Any],
    arm: str,
    args: argparse.Namespace,
    prompt_body: str,
    prompt_tokens: int,
    prompt_name: str,
) -> dict[str, Any]:
    """Build one §8.5 RunRecord from one arm of a perf_ab result."""
    key = "arm_a" if arm == "A" else "arm_b"
    stats = cell[key]
    top = result[key]
    declared = args.arm_a_kv_quant if arm == "A" else args.arm_b_kv_quant
    kv_quant = _kv_quant_of(top.get("args", ""), declared, arm)

    kv_bytes, kv_note = _kv_bytes(stats)
    tainted = cell.get("taint") or ""

    # The interleave pattern is read from the result file, never asserted: an `--invert` leg is
    # BAAB, and a result written before the field existed says so.
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
    # `null` is the supported way to say "not measured"; a 0 would rank in `bests` as the
    # smallest cache ever recorded.
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
        "run_class": "gate",
        "metrics": metrics,
    }


def ab_llama(argv: list[str]) -> int:
    """Write the arms of a `bench_llama_ab.sh` result as records.

    Without `--keep-arm` both arms are written as `ab_arm`, one record each, which is what the
    harness does after every real run. With `--keep-arm A|B` that arm alone is written, as
    `gate`: the surviving arm of an accepted comparison.

    Every identity field a cross-backend row needs is caller-supplied and none is guessed.
    `backend_version` is free-form for non-rMLX backends (§8.5.1): pass the build commit. A run
    the harness marked TAINTED is refused unless `--accept-tainted` is given, and the taint text
    is then carried into `notes`.
    """
    p = argparse.ArgumentParser(prog="ingest.py ab --harness llama", description=ab_llama.__doc__)
    p.add_argument("result", nargs="?", help="JSON emitted by bench_llama_ab.sh")
    p.add_argument("--prompt-file", required=True, help="the prompt body that was sent")
    p.add_argument("--prompt-name", required=True)
    p.add_argument("--prompt-notes", default=None)
    p.add_argument("--model", required=True, help="§5.1 model label")
    p.add_argument("--model-namespace", default="hf")
    p.add_argument("--weight-quant", required=True, help="§5.2 whitelist value, e.g. q8_0")
    p.add_argument("--arm-a-backend", required=True, help="§5.4 whitelist value")
    p.add_argument("--arm-b-backend", required=True)
    p.add_argument("--arm-a-version", default=None, help="free-form for non-rMLX backends")
    p.add_argument("--arm-b-version", default=None)
    p.add_argument("--arm-a-kv-quant", required=True)
    p.add_argument("--arm-b-kv-quant", required=True)
    p.add_argument("--arm-a-git-sha", default=None, help="commit of arm A's binary")
    p.add_argument("--arm-b-git-sha", default=None, help="commit of arm B's binary")
    p.add_argument("--hardware-tag", default="m5_max_128gb")
    p.add_argument(
        "--build-profile",
        default=None,
        help="how the measured binary was built, e.g. 'release'. Not guessed: an "
        "unset value records NULL rather than a plausible-looking default.",
    )
    p.add_argument(
        "--git-sha",
        default=None,
        help="caller-supplied provenance (§8.5.1). For a cross-backend arm this is "
        "the commit of the MEASURED backend, not of rMLX — a column that means "
        "'the rMLX tree that drove the bench' on some rows and 'the binary under "
        "test' on others cannot be queried. Pass the arm's own commit, or omit it.",
    )
    p.add_argument("--description", default=None)
    p.add_argument(
        "--keep-arm",
        choices=["A", "B"],
        default=None,
        help="write only this arm, as gate: the surviving arm of an accepted "
        "comparison (docs/METRICS_DB.md §8.5). Without it both arms are written "
        "as ab_arm, which is what bench_llama_ab.sh does after every real run.",
    )
    p.add_argument("--accept-tainted", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--record", action="store_true", help="ingest via `rmlx metrics record --file`")
    p.add_argument("--rmlx-bin", default=str(DEFAULT_RMLX_BIN))
    p.add_argument(
        "--check-args",
        action="store_true",
        help="parse the arguments, and with --record check the recorder, then exit 0 "
        "without reading a result: bench_llama_ab.sh runs this before its first slot",
    )
    args = p.parse_args(argv)

    if args.check_args:
        if args.record:
            require_run_class_recorder(args.rmlx_bin)
        return 0
    if args.result is None:
        p.error("the following arguments are required: result")

    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    # No waiver: a stub-armed run is not a real measurement taken under interference.
    if result.get("synthetic_arms"):
        print(
            "refusing: the run was made with --synthetic-arms, so its arms were "
            "stub servers\n"
            "  and its numbers measure nothing. There is no waiver for this: "
            "re-run the\n"
            "  comparison against real binaries.",
            file=sys.stderr,
        )
        return 2

    if result.get("tainted") and not args.accept_tainted:
        print(
            f"refusing: run is TAINTED ({result['tainted']}).\n"
            "  Re-run on a quiet host, or pass --accept-tainted to record it with the\n"
            "  taint carried into `notes`.",
            file=sys.stderr,
        )
        return 2

    prompt_body = Path(args.prompt_file).read_text(encoding="utf-8")
    if args.keep_arm:
        records = [_llama_ab_record(result, args.keep_arm, args, prompt_body, "gate")]
    else:
        records = [_llama_ab_record(result, arm, args, prompt_body, "ab_arm") for arm in ("A", "B")]

    staged = stage_records(
        records,
        args,
        staging=BENCH_DIR / "llama_ab" / "records",
        name_key="backend",
        announce=announce_each,
    )
    if staged is None:
        return 0
    failed = record_staged(staged, args.rmlx_bin)
    if failed:
        # The other records landed, so re-running this ingest would write them a second time;
        # the ones that did not are recorded one by one.
        for path in failed:
            print(f"not recorded: {path}", file=sys.stderr)
        print(
            f"  record each with `{args.rmlx_bin} metrics record --file <path>`; do not "
            f"re-run this ingest, which would write the {len(staged) - len(failed)} "
            "record(s) that landed again",
            file=sys.stderr,
        )
    return 1 if failed else 0


def _llama_ab_record(
    result: dict[str, Any],
    arm: str,
    args: argparse.Namespace,
    prompt_body: str,
    run_class: str,
) -> dict[str, Any]:
    """Build one §8.5 RunRecord from one arm of a llama A/B result."""
    slots = [s for s in result["slots"] if s["arm"] == arm]
    if not slots:
        raise SystemExit(f"no slots for arm {arm} in {args.result}")

    tps = [s["decode_tps"] for s in slots]
    backend = args.arm_a_backend if arm == "A" else args.arm_b_backend
    version = args.arm_a_version if arm == "A" else args.arm_b_version
    kv_quant = args.arm_a_kv_quant if arm == "A" else args.arm_b_kv_quant
    # `git_sha` describes the binary that produced the number, so it is per arm.
    per_arm_sha = args.arm_a_git_sha if arm == "A" else args.arm_b_git_sha

    # One measurement per slot, so a single-slot arm has no sample stddev.
    stddev = statistics.stdev(tps) if len(tps) > 1 else None

    # Differing values across an arm's slots mean the arm was not one configuration.
    kv_mib = {s["kv_mib"] for s in slots}
    if len(kv_mib) != 1:
        raise SystemExit(f"arm {arm} slots disagree on kv_mib: {sorted(kv_mib)}")

    prompt_n = {s["prompt_n"] for s in slots}
    if len(prompt_n) != 1:
        raise SystemExit(f"arm {arm} slots disagree on prompt_n: {sorted(prompt_n)}")

    predicted_n = {s["predicted_n"] for s in slots}
    if len(predicted_n) != 1:
        raise SystemExit(
            f"arm {arm} slots disagree on predicted_n: {sorted(predicted_n)}"
        )

    notes = (
        f"bench_llama_ab.sh ABBA n={len(slots)}/arm; "
        f"decode_tps median over slots, min={min(tps):.3f} max={max(tps):.3f}; "
        f"verdict={result['verdict']}"
    )
    if result.get("tainted"):
        notes += f"; TAINTED: {result['tainted']}"

    return {
        "schema_version": 1,
        "backend": backend,
        "backend_version": version,
        "model_namespace": args.model_namespace,
        "model": args.model,
        "weight_quant": args.weight_quant,
        "kv_quant": kv_quant,
        "ctx_max": result["n_ctx"],
        "prompt": {"name": args.prompt_name, "body": prompt_body, "notes": args.prompt_notes},
        "ts_utc": result["ts_utc"],
        "git_sha": per_arm_sha or args.git_sha,
        "build_profile": args.build_profile,
        "hardware_tag": args.hardware_tag,
        "prompt_tokens": slots[0]["prompt_n"],
        "max_tokens": result["n_predict"],
        "temperature": 0.0,
        "seed": 0,
        "n_warmups": 1,
        "n_measure": len(slots),
        "output_first_64": slots[0].get("output_first_64") or None,
        "notes": notes,
        "description": args.description,
        "run_class": run_class,
        "metrics": [
            {"name": "decode_tps_warm", "value": statistics.median(tps), "stddev": stddev},
            {
                "name": "prefill_tps",
                "value": statistics.median([s["prompt_tps"] for s in slots]),
            },
            {"name": "kv_cache_bytes", "value": kv_mib.pop() * LLAMA_MIB},
            {"name": "peak_rss_mb", "value": statistics.median([s["peak_rss_mb"] for s in slots])},
        ],
    }


# ── codec-inertness ───────────────────────────────────────────────────────────

# Column the probe records its `--kv-boundary-layers` value in.
KV_BOUNDARY_FIELD = "kv_boundary"
# Column the probe sets on a row whose run it recorded itself.
RECORDED_FIELD = "recorded_run_class"
# Where `csv.DictReader` puts fields the header does not name. Checked, not ignored: a
# non-empty bucket means the columns are shifted.
EXTRA_FIELDS_KEY = "__unnamed_fields__"


def run_codec_inertness(argv: list[str]) -> int:
    """Write the residency cells of a `codec_inertness_probe.sh` CSV as `diagnostic` records.

    The probe records its own runs and marks such a CSV row in `recorded_run_class`; that row is
    skipped here, so no run's `kv_cache_bytes` is written twice. Every other usable row becomes
    one record per (model, context, codec).

    **Only `kv_cache_bytes` is recorded.** The probe's throughput columns are single unpaired
    runs on a shared host, where the same binary and flags have read 11% apart thirty minutes
    apart; promoting one into an append-only table would make an unusable number permanent.
    Throughput belongs to `perf_ab.sh` and `ab`. The greedy token-id digest travels in `notes`: it
    is an identity, not a measurement, and `observations` stores reals.

    A sweep at non-default `--kv-boundary-layers` lands in a cell of its own: the probe's
    `kv_boundary` column becomes the record's `decode_config` (`docs/METRICS_SCHEMA.md` §3.2).
    """
    p = argparse.ArgumentParser(prog="ingest.py codec-inertness", description=run_codec_inertness.__doc__)
    p.add_argument("csv", help="codec_inertness.csv written by the probe")
    p.add_argument(
        "--backend-version",
        required=True,
        help="semver of the rmlx binary that produced the cells (§8.5.1). Required, "
        "not inferred: a wrong identity value is worse than no value.",
    )
    p.add_argument(
        "--build-profile",
        required=True,
        help="how the measured binary was built, e.g. 'release-perf'.",
    )
    p.add_argument("--git-sha", default=None, help="caller-supplied provenance (§8.5.1)")
    p.add_argument("--hardware-tag", default="m5_max_128gb")
    p.add_argument("--description", default=None)
    p.add_argument("--prompts-dir", default=str(REPO_ROOT / "prompts"))
    p.add_argument(
        "--models-root",
        required=True,
        help="the directory holding the snapshots the CSV's model column names; each "
        "row's weight label is read from <models-root>/<model>/config.json",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--record", action="store_true", help="ingest via `rmlx metrics record --file`")
    p.add_argument("--rmlx-bin", default=str(REPO_ROOT / "target" / "release-perf" / "rmlx"))
    args = p.parse_args(argv)

    prompts_dir = Path(args.prompts_dir)
    with Path(args.csv).open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, restkey=EXTRA_FIELDS_KEY)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    # A row wider than the header means the file was appended to under a header of a different
    # shape. The reader maps by NAME, so the surplus lands in the leftover bucket and whichever
    # column it belonged to reads as absent: for `kv_boundary`, that files a non-default sweep
    # under the DEFAULT cell, permanently. Refuse rather than guess which column shifted.
    surplus = [i for i, r in enumerate(rows, start=2) if r.get(EXTRA_FIELDS_KEY)]
    if surplus:
        raise SystemExit(
            f"{args.csv} line(s) {surplus[:5]} carry more fields than the header names "
            f"({len(fieldnames)}: {','.join(fieldnames)}). The file was appended to under "
            "a header of a different shape, so every column past the first mismatch is "
            "read under the wrong name. Refusing; split the sweep into its own file."
        )
    if KV_BOUNDARY_FIELD not in fieldnames:
        # A sweep from before the probe recorded the column, when the boundary could not be
        # anything but the default; `_codec_decode_config` says so, this only makes it visible.
        print(
            f"note: {args.csv} has no `{KV_BOUNDARY_FIELD}` column, so it predates the "
            "flag and every row is recorded at the default boundary"
        )
    recorded = [r for r in rows if (r.get(RECORDED_FIELD) or "").strip()]
    if recorded:
        print(
            f"skipping {len(recorded)} row(s) the probe already recorded, or whose record is "
            f"pending a replay (`{RECORDED_FIELD}` is set); ingesting them would write each "
            "byte count twice"
        )
    rows = [r for r in rows if not (r.get(RECORDED_FIELD) or "").strip()]
    # A failed cell emits an empty `kv_cache_bytes` and an empty digest, the sha256 of nothing,
    # which reads exactly like a real result. The exit code tells the two apart.
    usable = [r for r in rows if r["exit_code"] == "0" and r["kv_cache_bytes"]]
    dropped = len(rows) - len(usable)
    if dropped:
        print(f"skipping {dropped} row(s) with a non-zero exit code or no byte count")
    if not usable:
        print("no usable rows", file=sys.stderr)
        return 2

    records = [_codec_record(r, args, prompts_dir) for r in usable]

    staging = BENCH_DIR / "codec_inertness" / "records"
    staged = stage_records(
        records,
        args,
        staging=staging,
        name_key="kv_quant",
        announce=lambda staged: print(f"staged {len(staged)} record(s) in {staging}"),
    )
    if staged is None:
        return 0
    failed = record_staged(staged, args.rmlx_bin)
    print(f"ingested {len(staged) - len(failed)} of {len(staged)}")
    return 1 if failed else 0


def _codec_decode_config(row: dict[str, str]) -> str | None:
    """The `decode_config` cell term this row's boundary-layer counts imply.

    Read off the `kv_boundary` column the probe writes from its own argument, so the term
    describes the sweep that ran. `None` is the engine at its defaults, which is also the right
    answer for a CSV written before the probe had the column: there was no way to run at anything
    else then.
    """
    raw = (row.get(KV_BOUNDARY_FIELD) or "").strip()
    if not raw:
        return None
    try:
        head, tail = (int(x) for x in raw.split(","))
    except ValueError:
        raise SystemExit(
            f"unparseable kv_boundary column {raw!r}: expected '<head>,<tail>'. "
            "Refusing rather than recording the row under the default cell — a "
            "sweep filed in the wrong cell is permanent."
        ) from None
    return boundary_term(head, tail)


def _codec_weight_quant(models_root: Path, model_dir: str) -> str:
    """The weight label the snapshot's `config.json` describes, as `rmlx baseline --record` reads it.

    A label guessed from the name would file the row in another cell than the probe's own record.
    """
    try:
        return snapshot_weight_quant(str(models_root / model_dir))
    except UnlabelledSnapshot as exc:
        raise SystemExit(f"cannot label the row's weights: {exc}") from None


def _codec_prompt(fixture_tokens: int, prompts_dir: Path) -> tuple[str, str]:
    """Resolve the canonical bench prompt the probe's `--prompt-tokens` names.

    The body is the fixture file's own text, which is what `ab` records, so the `prompts` row
    this record keys on is the one every other cell at that length already keys on.
    """
    name = f"longctx_{fixture_tokens // 1024}k"
    path = prompts_dir / f"{name}.json"
    if not path.is_file():
        raise SystemExit(f"prompt fixture not found: {path}")
    return name, path.read_text(encoding="utf-8")


def _codec_record(row: dict[str, str], args: argparse.Namespace, prompts_dir: Path) -> dict[str, Any]:
    namespace, model = split_model_dir(row["model"])
    fixture_tokens = int(row["prompt_tokens"])
    # The fixture NAME is not a token count: the chat template pushes a `longctx_32k` prompt to
    # 34 355 tokens on gemma-4. Refuse rather than substitute: a plausible wrong number is the
    # failure mode here.
    measured = row.get("prompt_tokens_measured") or ""
    if not measured:
        raise SystemExit(
            "this CSV has no `prompt_tokens_measured` column (it predates the "
            "probe recording one). Re-run the probe; the fixture name in "
            "`prompt_tokens` is not a token count and is not substituted."
        )
    prompt_tokens = int(measured)
    # `ctx_max` is a PK column and the probe accepts `--max-ctx`: re-deriving it from its default
    # formula would silently record a plausible wrong ceiling for a sweep that overrode one.
    raw_ctx = row.get("max_ctx") or ""
    if not raw_ctx:
        raise SystemExit(
            "this CSV has no `max_ctx` column (it predates the probe recording "
            "one). Re-run the probe; the default ceiling formula is not "
            "re-derived here, because a sweep that passed --max-ctx would then "
            "be recorded under a ceiling it never ran at."
        )
    max_ctx = int(raw_ctx)
    prompt_name, prompt_body = _codec_prompt(fixture_tokens, prompts_dir)
    return {
        "schema_version": 1,
        "backend": "rmlx",
        "backend_version": args.backend_version,
        "model_namespace": namespace,
        "model": model,
        "weight_quant": _codec_weight_quant(Path(args.models_root), row["model"]),
        "kv_quant": row["codec"],
        "decode_config": _codec_decode_config(row),
        "ctx_max": max_ctx,
        "prompt": {"name": prompt_name, "body": prompt_body, "notes": None},
        "ts_utc": row["timestamp"],
        "git_sha": args.git_sha,
        "build_profile": args.build_profile,
        "hardware_tag": args.hardware_tag,
        "prompt_tokens": prompt_tokens,
        "max_tokens": int(row["max_tokens"]),
        "temperature": 0.0,
        "seed": 0,
        "n_warmups": 0,
        "n_measure": 1,
        "output_first_64": None,
        "notes": (
            # `any_layer_skipped_store` is per-RUN, not per-codec: the layer-adaptive head/tail
            # promotion types some layers `K8V8`, so a store-reading codec sets it too. Rows
            # written before the rename spell it `packed_store_skipped` and carry no caveat;
            # `observations` is append-only, so a query over the column accepts either.
            f"codec_inertness_probe; binary_sha={row['binary_sha']}; "
            f"token_ids_sha256_16={row['ids_sha']}; "
            f"any_layer_skipped_store={row['store_skipped']}; "
            "residency only — the probe's throughput columns are single unpaired "
            "runs on a shared host and are deliberately not recorded"
        ),
        "description": args.description,
        "run_class": "diagnostic",
        "metrics": [{"name": "kv_cache_bytes", "value": float(row["kv_cache_bytes"])}],
    }


# ── published ─────────────────────────────────────────────────────────────────


def run_published(argv: list[str]) -> int:
    args = published.parse_args(argv)
    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    try:
        records = published.build_records(result, args)
    except published.Refused as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"{len(records)} records", file=sys.stderr)
    staging = Path(args.result).resolve().parent / "records"
    staged = stage_records(
        records,
        args,
        staging=staging,
        name_key=None,
        announce=lambda staged: print(f"staged {len(staged)} records under {staging}"),
        compact=True,
    )
    if staged is None:
        return 0
    failed = record_staged(staged, args.rmlx_bin, capture=True)
    print(f"ingested {len(staged) - len(failed)}/{len(staged)} records")
    if failed:
        return 1
    return published.requery(args)


# ── dispatch ──────────────────────────────────────────────────────────────────

MODES: dict[str, Callable[[list[str]], int]] = {
    "ab": run_ab,
    "codec-inertness": run_codec_inertness,
    "llama-bench": llama_bench.main,
    "published": run_published,
}


def main(argv: list[str]) -> int:
    if argv[:1] in (["-h"], ["--help"]):
        print(__doc__)
        return 0
    if not argv or argv[0] not in MODES:
        print(f"usage: ingest.py {{{'|'.join(MODES)}}} [flags]; ingest.py -h lists the modes", file=sys.stderr)
        return 2
    return MODES[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
