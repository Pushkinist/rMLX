#!/usr/bin/env python3
"""Promote a `scripts/spec_bench_published.sh` result into `runs.db`.

The harness deliberately writes no DB row — a measurement and a record are
separate acts, and the store is append-only, so a row written by the thing that
produced it cannot be taken back. This is the explicit second step.

WHAT IS RECORDED, AND WHAT IS NOT

One §8.5 record per measured thing, and the measured things are the samples:

  per (cell, sample)   `decode_tps_warm` and `ttft_warm_ms`, each the mean of
                       the three passes with the paired stddev, `n_measure=3`.
                       `prompt_id` is that sample's own body, so the row joins
                       to the exact prompt it was measured on. The speculative
                       arm adds its per-round figures.
  the fixed prompt     one record: output speed, input speed, TTFT and the two
                       resident peaks, on the prompt fitted for this checkpoint.

The per-dataset and macro averages are NOT recorded. They are means over the
rows above, so storing them would put a derived number beside the numbers it is
derived from, where the two can disagree and nothing says which is right. The
Markdown emitter takes them from the result file, which is where the harness
computed them once.

A dataset mean the harness refused for its run-to-run range is likewise not a
reason to refuse these rows. The refusal is of a mean; the rows are the
measurements it would have been a mean of, each carrying its own three pass
readings in `notes`.

WHAT IS REFUSED

  synthetic arms       the server was a stub. There is no measurement in it and
                       no waiver for that.
  unverified samples   `--samples-root` named a copy that is not the pinned one,
                       so the run is not a published measurement.
  a taint              a host that was busy, a window nobody could sample, a
                       thermal state that was throttled or unreadable. Waivable
                       with `--accept-tainted`, and then carried into `notes`.
  a moved binary       the file the run measured is re-hashed here. Between the
                       run and the ingest a rebuild can put a different binary
                       at that path, and then `backend_version` and
                       `build_profile` describe one binary while the row asserts
                       another — permanently, in an append-only table.
  a moved sample set   the published root is re-verified against the pins in
                       `published_samples.py`. Same failure, other input.
  a body that moved    HARD. The digest recorded with each measurement, the
                       digest in the sample manifest, and the digest this
                       recorder gives the body about to be submitted must be one
                       value. A mismatch attributes a row to a prompt that is
                       not the one measured, and the join splits with nothing
                       saying so — which is exactly the failure `prompt_id`
                       exists to prevent.

Usage:
    python3 scripts/ingest/published_ingest.py --dry-run RESULT.json
    python3 scripts/ingest/published_ingest.py --record RESULT.json --git-sha <sha>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))
sys.path.insert(0, str(_SCRIPT_DIR.parent / "lib"))
from published_samples import body_sha256  # noqa: E402

RMLX_REPO_ROOT = Path(os.environ.get("RMLX_REPO_ROOT", str(_SCRIPT_DIR.parents[1])))
# Runtime state lives under a single root (CLAUDE.md). `RMLX_HOME` wins; the
# in-repo `.rmlx/` is the dev default.
RMLX_HOME = Path(os.environ.get("RMLX_HOME", str(RMLX_REPO_ROOT / ".rmlx")))
BUFFER_PENDING = RMLX_HOME / "metrics" / "buffer" / "pending"
# Records are assembled HERE and only moved into `pending/` for the moment the
# recorder claims one: anything sitting in `pending/` is, by contract, work the
# next `--replay-pending` sweep will take, and that sweep quarantines what it
# cannot ingest.
STAGING = RMLX_HOME / "bench" / "spec_bench_published" / "records"

SCHEMA_VERSION = 1
# The identity fields the run stamped at emit time. §8.5.1: they are never
# re-derived at ingest, so a buffer replayed later keeps the identity of the
# build that produced it.
IDENTITY_FIELDS = ("backend", "backend_version", "build_profile", "hardware_tag")
# Per-round figures a speculative row carries. Each is its own registry metric.
ROUND_METRICS = ("tokens_per_round", "accepted_per_step", "accept_rate")


class Refused(Exception):
    """The result describes something that must not become a row."""


# ── provenance ────────────────────────────────────────────────────────────────


def check_binary(result: dict[str, Any]) -> str:
    """Re-hash the binary the run measured, and check the markers again."""
    binary = result.get("binary")
    if not binary:
        raise Refused(
            "this result carries no binary identity, so the row's "
            "backend_version and build_profile would describe a build nothing "
            "recorded. Re-run on a harness that records one."
        )
    path = Path(binary["path"])
    if not path.is_file():
        raise Refused(
            f"the binary the run measured is gone from {path}, so its recorded "
            f"identity (sha256:{binary['sha256'][:16]}) cannot be confirmed"
        )
    blob = path.read_bytes()
    actual = hashlib.sha256(blob).hexdigest()
    if actual != binary["sha256"]:
        raise Refused(
            f"the binary at {path} now hashes sha256:{actual[:16]} where the run "
            f"recorded sha256:{binary['sha256'][:16]}. It was rebuilt after the "
            "measurement, so its version and build profile no longer describe "
            "these numbers."
        )
    absent = [
        m
        for m, count in binary["markers"].items()
        if count and not blob.count(m.encode("utf-8"))
    ]
    if absent:
        raise Refused(
            f"the binary at {path} no longer contains {', '.join(map(repr, absent))}, "
            "which the run recorded it containing; the digest and the contents "
            "disagree"
        )
    return binary["sha256"]


def check_samples_root(result: dict[str, Any], repo_root: Path) -> None:
    """Re-verify that the pinned sample sets still re-derive from their pins."""
    root = Path(result["samples_root"])
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "published_samples.py"),
            "verify",
            "--root",
            str(root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise Refused(
            f"the sample sets at {root} no longer re-derive from what "
            f"published_samples.py pins:\n{proc.stdout}{proc.stderr}"
        )


def manifest_bodies(result: dict[str, Any]) -> dict[str, tuple[str, list]]:
    """`{sample_id: (body_sha256, messages)}` over every checked-in sample."""
    root = Path(result["samples_root"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    bodies: dict[str, tuple[str, list]] = {}
    for entry in manifest["datasets"]:
        doc = json.loads((root / entry["file"]).read_text(encoding="utf-8"))
        for sample in doc["samples"]:
            bodies[sample["id"]] = (sample["body_sha256"], sample["messages"])
    return bodies


def assert_one_body(
    sample_id: str, measured: str, manifest_digest: str, messages
) -> None:
    """The three digests of one prompt body must be one value.

    HARD, and not a warning: a row whose `prompt_id` resolves to a body other
    than the measured one is attributed to a prompt that was never sent, and
    every later query joining on it splits silently between the two.
    """
    recorded = body_sha256(messages)
    if not (measured == manifest_digest == recorded):
        raise Refused(
            f"sample {sample_id}: the measurement was recorded against "
            f"{measured[:16]}, the manifest holds {manifest_digest[:16]}, and "
            f"this recorder gives the body it is about to submit "
            f"{recorded[:16]}. They are not one prompt, so the row would be "
            "attributed to a body that was not measured."
        )


# ── records ───────────────────────────────────────────────────────────────────


def mean_stddev(values: list[float]) -> tuple[float, float | None]:
    return statistics.fmean(values), (
        statistics.stdev(values) if len(values) > 1 else None
    )


def base_record(result: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    sampling = (result.get("protocol") or {}).get("sampling_resolved") or {}
    record = {
        **{name: result[name] for name in IDENTITY_FIELDS},
        "schema_version": SCHEMA_VERSION,
        "model_namespace": result["model_namespace"],
        "model": result["model"],
        "weight_quant": result["weight_quant"],
        "kv_quant": result["kv_quant"],
        "ctx_max": result["ctx_max"],
        "ts_utc": result["ts_utc"],
        "git_sha": args.git_sha,
        "n_warmups": result["protocol"]["warmups_per_pass"],
        "n_measure": result["protocol"]["passes"],
        "temperature": sampling.get("temperature"),
        # The engine's own resolved seed, not the harness's idea of it. Every
        # pass ran under this one value, which is what makes the run-to-run
        # spread in `notes` a reading of machine variance.
        "seed": sampling.get("seed"),
        "description": args.description,
    }
    # `decode_config` is cell identity: absent means every engine setting at its
    # default, and the empty string is not a spelling of that.
    if result.get("decode_config"):
        record["decode_config"] = result["decode_config"]
    return record


def run_notes(result: dict[str, Any], binary_sha: str) -> str:
    """What every row from this run says about how it was taken."""
    protocol = result["protocol"]
    parts = [
        f"spec_bench_published.sh, {protocol['passes']} passes, "
        f"{protocol['warmups_per_pass']} untimed warmup each",
        f"sampling {protocol['thinking']}, seed {protocol['seed_policy']}",
        f"binary sha256:{binary_sha[:16]}",
    ]
    host = result.get("host") or {}
    if host.get("thermal"):
        parts.append(f"thermal ({host['thermal_source']}): {'; '.join(host['thermal'])}")
    if host.get("taint"):
        parts.append(f"TAINTED: {host['taint'].strip().rstrip(';')}")
    return "; ".join(parts)


def sample_records(
    result: dict[str, Any],
    args: argparse.Namespace,
    bodies: dict[str, tuple[str, list]],
    notes: str,
) -> list[dict[str, Any]]:
    """One record per (cell, sample), over that sample's three passes."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in result["samples"]:
        grouped.setdefault((row["cell"], row["sample_id"]), []).append(row)

    records = []
    for (cell, sample_id), rows in grouped.items():
        passes = result["protocol"]["passes"]
        if len(rows) != passes:
            raise Refused(
                f"{cell}/{sample_id} was measured {len(rows)} times where the "
                f"protocol reports the mean of {passes}; a mean over the passes "
                "that produced a row is not a mean over the passes"
            )
        if sample_id not in bodies:
            raise Refused(
                f"{cell}/{sample_id} was measured but the sample sets hold no "
                "sample of that id, so there is no body to record it against"
            )
        manifest_digest, messages = bodies[sample_id]
        measured = {r["body_sha256"] for r in rows}
        if len(measured) != 1:
            raise Refused(
                f"{cell}/{sample_id}: the passes recorded {sorted(measured)} as "
                "the body they measured; they did not measure one prompt"
            )
        assert_one_body(sample_id, measured.pop(), manifest_digest, messages)

        prompt_tokens = {r["prompt_tokens"] for r in rows}
        if len(prompt_tokens) != 1:
            raise Refused(
                f"{cell}/{sample_id}: the server counted this prompt at "
                f"{sorted(prompt_tokens)} tokens across the passes; one body "
                "does not have two lengths"
            )

        dataset = rows[0]["dataset"]
        tps_mean, tps_sd = mean_stddev([r["decode_tps"] for r in rows])
        metrics = [
            {"name": "decode_tps_warm", "value": tps_mean, "stddev": tps_sd},
            {
                "name": "ttft_warm_ms",
                "value": statistics.fmean([r["ttft_ms"] for r in rows]),
            },
        ]
        for name in ROUND_METRICS:
            present = [r[name] for r in rows if name in r]
            # `null` is the supported way to say "not measured"; a 0 on the
            # plain arm would rank as a measured round figure.
            metrics.append(
                {
                    "name": name,
                    "value": statistics.fmean(present) if present else None,
                }
            )

        lengths = {r["completion_tokens"] for r in rows}
        row_notes = notes + (
            f"; {cell} sample {sample_id}; pass decode_tps "
            + " ".join(f"{r['decode_tps']:.3f}" for r in rows)
        )
        if len(lengths) > 1:
            row_notes += (
                f"; the passes generated {sorted(lengths)} tokens, so this "
                "sample's spread carries sampling variance as well as machine "
                "variance"
            )

        records.append(
            {
                **base_record(result, args),
                "prompt": {
                    "name": f"published/{dataset}/{sample_id}",
                    "body": messages,
                    "notes": (
                        f"{dataset} sample {sample_id} from prompts/published/, "
                        "pinned by scripts/published_samples.py"
                    ),
                },
                "prompt_tokens": prompt_tokens.pop(),
                "max_tokens": rows[0]["max_tokens"],
                "notes": row_notes,
                "metrics": metrics,
            }
        )
    return records


def fixed_record(
    result: dict[str, Any], args: argparse.Namespace, notes: str
) -> dict[str, Any] | None:
    """The fixed-length-prompt block, as one record. `None` when it was not run."""
    block = result.get("fixed_prompt")
    if not block:
        return None
    if body_sha256(block["messages"]) != block["body_sha256"]:
        raise Refused(
            "the fixed prompt's body does not hash to the digest the run "
            "recorded for it, so the row would be attributed to a prompt that "
            "was not measured"
        )
    rates = {name: block[name] for name in ("decode_tps", "prefill_tps")}
    return {
        **base_record(result, args),
        "prompt": {
            "name": f"published/fixed_{block['target_tokens']}",
            "body": block["messages"],
            "notes": (
                f"fitted to {block['prompt_tokens']} tokens on this checkpoint's "
                f"tokenizer: the first {block['words']} words of "
                f"{block['corpus']} (sha256:{block['corpus_sha256'][:16]}) plus "
                f"{block['filler_reps']} x '{block['filler_word']}', behind the "
                "instruction pinned in scripts/lib/published_fixed_prompt.py"
            ),
        },
        "prompt_tokens": block["prompt_tokens"],
        "max_tokens": block["max_tokens"],
        "notes": (
            notes
            + f"; fixed prompt, plain decode; run decode_tps "
            + " ".join(f"{v:.3f}" for v in rates["decode_tps"]["pass_means"])
            + f"; range {rates['decode_tps']['range_pct']:.2f}%"
            + "; resident figures are the peak of a gauge sampled every "
            + f"{block['memory_poll_ms']} ms, so they are a lower bound on the "
            "true peak"
        ),
        "metrics": [
            {
                "name": "decode_tps_warm",
                "value": rates["decode_tps"]["mean"],
                "stddev": statistics.stdev(rates["decode_tps"]["pass_means"]),
            },
            {
                "name": "prefill_tps",
                "value": rates["prefill_tps"]["mean"],
                "stddev": statistics.stdev(rates["prefill_tps"]["pass_means"]),
            },
            {"name": "ttft_warm_ms", "value": block["ttft_ms"]["mean"]},
            {
                "name": "peak_phys_footprint_mb",
                "value": block["phys_footprint_bytes"]["max"] / 1e6,
            },
            {"name": "peak_rss_mb", "value": block["rss_bytes"]["max"] / 1e6},
        ],
    }


# ── driver ────────────────────────────────────────────────────────────────────


def gate(result: dict[str, Any], args: argparse.Namespace) -> None:
    if result.get("synthetic_arms"):
        raise Refused(
            "the run was made with --synthetic-arms, so its server was a stub "
            "and its numbers measure nothing. There is no waiver for this: "
            "re-run against a real server."
        )
    if result.get("unverified_samples"):
        raise Refused(
            "the run measured a sample root that is not the pinned one "
            "(--samples-root), so it is not a published measurement. There is no "
            "waiver for this: re-run against prompts/published/."
        )
    taint = (result.get("host") or {}).get("taint")
    if taint and not args.accept_tainted:
        raise Refused(
            f"run is TAINTED ({taint.strip()}).\n"
            "  Re-run on a quiet host, or pass --accept-tainted to record it "
            "with the taint carried into `notes`."
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("result", help="JSON emitted by spec_bench_published.sh")
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
    try:
        gate(result, args)
        binary_sha = check_binary(result)
        check_samples_root(result, RMLX_REPO_ROOT)
        bodies = manifest_bodies(result)
        notes = run_notes(result, binary_sha)
        records = sample_records(result, args, bodies, notes)
        fixed = fixed_record(result, args, notes)
    except Refused as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 2
    except (KeyError, OSError, ValueError) as exc:
        print(
            f"refusing: {args.result} is missing something every record needs "
            f"({exc!r}); it was not written by this harness",
            file=sys.stderr,
        )
        return 2
    if fixed:
        records.append(fixed)

    if args.dry_run:
        for rec in records:
            trimmed = dict(rec)
            trimmed["prompt"] = dict(
                rec["prompt"], body=f"<{len(json.dumps(rec['prompt']['body']))} chars>"
            )
            print(json.dumps(trimmed, indent=2))
        print(f"{len(records)} records", file=sys.stderr)
        return 0

    stamp = result["ts_utc"].replace("-", "").replace(":", "")
    STAGING.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for rec in records:
        path = STAGING / f"{stamp}-{uuid.uuid4().hex[:8]}.json"
        path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        staged.append(path)
    print(f"staged {len(staged)} records under {STAGING}")

    if not args.record:
        print(
            "not ingested (pass --record). The files above are OUTSIDE\n"
            f"  {BUFFER_PENDING},\n"
            "  so no --replay-pending sweep can claim them. Inspect, then re-run\n"
            "  with --record."
        )
        return 0

    BUFFER_PENDING.mkdir(parents=True, exist_ok=True)
    failed = 0
    for path in staged:
        queued = BUFFER_PENDING / path.name
        path.rename(queued)
        try:
            proc = subprocess.run(
                [args.rmlx_bin, "metrics", "record", "--file", str(queued)],
                check=False,
                capture_output=True,
                text=True,
            )
        finally:
            if queued.exists():
                queued.rename(path)
        if proc.returncode != 0:
            print(
                f"ingest FAILED for {path} (exit {proc.returncode}): "
                f"{proc.stderr.strip()}; record kept in staging, not in the "
                "pending queue",
                file=sys.stderr,
            )
            failed += 1
    print(f"ingested {len(staged) - failed}/{len(staged)} records")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
