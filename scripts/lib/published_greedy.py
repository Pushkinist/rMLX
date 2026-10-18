#!/usr/bin/env python3
"""Seal a greedy pass, and judge a speculative one against a plain one.

The speed passes of the published protocol sample under the checkpoint's own
defaults, so two arms never answer the same question there. Judging whether a
drafter changed the answer needs a pass of its own: every sample once, at temperature 0, unstreamed and with
logprobs, because `rmlx eval spec-answer` reads the margins of a saved answer. The
speed passes never send a logprob request, which reads the whole vocabulary back
to the host at every step.

  seal     check the directory a greedy pass left (an index of answers and the
           saved responses) and write `meta.json` beside it: the identity of the
           pass, and a digest of the sample set it covers.
  cold     check the log of the server that answered the pass: one prefill event
           for each request, and not one of them took a token from a prompt cache.
  compare  put a speculative pass beside a plain one. Both must be sealed, of one
           checkpoint at one KV codec and one context ceiling, and over one sample
           set. Per sample, equal answers are identical; a pair that differs is
           judged by the divergence oracle, `rmlx eval spec-answer`, and its
           verdict is filed as it came: `near_tie`, `refused` or `unjudgeable`.

THE MATCH IS TOKEN FOR TOKEN. A sample matches when the speculative answer, its
reasoning included, is the plain answer to the last character: the digest the
streamed and the unstreamed readers both take. A divergence at a near tie is not
that. It is the verify pass scoring a block in one forward and flipping a decision
the verifier was nearly tied on, so it is counted apart and reported in a rate
beside the identical one. The oracle reads the first divergence only: a near tie says
that decision was close, not that the rest of the two answers agrees. A
refused or unjudgeable pair is published, not stopped on: the table exists to say
how often an arm answered another question.

The oracle's exit code is its verdict's other spelling. A pair that disagree, or a
verdict this script does not know, is not a judgement and stops the run.

Exit codes: 0 — sealed or judged; 1 — the two passes are not of one measurement,
or the oracle gave an answer that is not a judgement; 2 — an input could not be
read.
"""

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import published_run_log  # noqa: E402

INDEX = "index.tsv"
META = "meta.json"
COLUMNS = ("cell", "dataset", "max_tokens", "sample_id", "body_sha256", "answer_sha256", "file")
VERDICTS = {"near_tie": 0, "refused": 1, "unjudgeable": 2}
# What a greedy pass is: the same for every arm, and stated in the meta so a
# pass made under another setting cannot be put beside this one.
PROTOCOL = {"temperature": 0, "logprobs": True, "top_logprobs": 2, "stream": False}
# Fields of the meta that both passes must share.
SHARED = ("model_namespace", "model", "weight_quant", "kv_quant", "ctx_max", "set_sha256")


class PassError(Exception):
    """The two passes are not of one measurement, or a pass is not whole."""


def read_index(directory):
    rows = []
    path = directory / INDEX
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split("\t")
        if len(parts) != len(COLUMNS):
            raise PassError(f"{path}:{number}: {len(parts)} columns, expected {len(COLUMNS)}")
        row = dict(zip(COLUMNS, parts))
        digest = row["answer_sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise PassError(f"{path}:{number}: {digest!r} is not a sha256 digest")
        if not (directory / row["file"]).is_file():
            raise PassError(f"{path}:{number}: the saved response {row['file']} is missing")
        rows.append(row)
    if not rows:
        raise PassError(f"{path} lists no answer")
    keys = [(r["cell"], r["sample_id"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise PassError(f"{path} lists a sample twice")
    return rows


def set_digest(rows):
    triples = sorted((r["cell"], r["sample_id"], r["body_sha256"]) for r in rows)
    return hashlib.sha256(json.dumps(triples).encode("utf-8")).hexdigest()


def seal(args):
    directory = pathlib.Path(args.dir)
    rows = read_index(directory)
    meta = {
        "arm": args.arm,
        "model_namespace": args.model_namespace,
        "model": args.model,
        "weight_quant": args.weight_quant,
        "kv_quant": args.kv_quant,
        "ctx_max": args.ctx_max,
        "decode_config": args.decode_config,
        "binary_sha256": args.binary_sha256,
        "ts_utc": args.ts_utc,
        "samples": len(rows),
        "set_sha256": set_digest(rows),
        "protocol": PROTOCOL,
    }
    (directory / META).write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(f"published_greedy: sealed {len(rows)} answers in {directory}")


def load_pass(directory, role, arm):
    meta_path = directory / META
    if not meta_path.is_file():
        raise PassError(
            f"{directory} holds no {META}: the {role} pass was not sealed, so it is "
            "not a greedy pass of the harness"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["arm"] != arm:
        raise PassError(f"the {role} pass is of the {meta['arm']} arm, not the {arm} one")
    if meta["protocol"] != PROTOCOL:
        raise PassError(f"the {role} pass ran under {meta['protocol']}, not {PROTOCOL}")
    rows = read_index(directory)
    if meta["set_sha256"] != set_digest(rows):
        raise PassError(
            f"the {role} pass's index no longer matches the sample set it was sealed over"
        )
    return meta, rows


def cold(args):
    """The reference answers come from a full prefill, whichever arm gave them.

    A speculative round loop prefills the whole prompt every time. A plain server
    keeps a prompt cache, and an answer decoded from a resumed prefill agrees with
    a cold one to rounding and not always token for token, so a plain sample that
    was resumed would be a reference the speculative answer is not comparable to.
    The pass is therefore run with the cache off, and this holds the engine to it.
    """
    events = published_run_log.events_matching(
        args.log, published_run_log.PREFILL_MESSAGE, exact=True
    )
    if len(events) != args.expect:
        raise PassError(
            f"the greedy server's log holds {len(events)} prefill events for "
            f"{args.expect} requests: the pass cannot be attributed answer by answer"
        )
    for number, fields in enumerate(events, 1):
        try:
            row = published_run_log.prefill_of(fields)
        except published_run_log.LogError as exc:
            raise PassError(f"greedy request {number}: {exc}") from exc
        if row["reused_tokens"]:
            raise PassError(
                f"the engine took {row['reused_tokens']} tokens of greedy request "
                f"{number} from a prompt cache, so its answer is not from a full "
                "prefill and is not a reference to judge another arm against"
            )
    print(f"published_greedy: {len(events)} greedy requests, all from a full prefill")


def judge(rmlx_bin, plain_file, spec_file):
    """The oracle's verdict for one pair, as `(verdict, fields)`."""
    done = subprocess.run(
        [rmlx_bin, "eval", "spec-answer", "--plain", str(plain_file), "--spec", str(spec_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    fields = {}
    for line in done.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            fields[key] = value
    verdict = fields.get("verdict")
    if verdict not in VERDICTS:
        raise PassError(
            f"`rmlx eval spec-answer` exited {done.returncode} with verdict "
            f"{verdict!r}, which is not a judgement of {spec_file.name}: "
            f"{done.stderr.strip()[-200:]}"
        )
    if done.returncode != VERDICTS[verdict]:
        raise PassError(
            f"`rmlx eval spec-answer` exited {done.returncode} but said "
            f"verdict={verdict} for {spec_file.name}: the exit code and the verdict "
            "are one answer spelled twice, and these disagree"
        )
    return verdict, fields


def compare(args):
    plain_dir, spec_dir = pathlib.Path(args.plain), pathlib.Path(args.spec)
    plain_meta, plain_rows = load_pass(plain_dir, "plain", "plain")
    spec_meta, spec_rows = load_pass(spec_dir, "speculative", "speculative")
    for name in SHARED:
        if plain_meta[name] != spec_meta[name]:
            raise PassError(
                f"the passes differ in {name} ({plain_meta[name]!r} against "
                f"{spec_meta[name]!r}); they did not answer one set of questions"
            )
    plain_by = {(r["cell"], r["sample_id"]): r for r in plain_rows}

    datasets = {}
    detail = []
    for row in spec_rows:
        other = plain_by[(row["cell"], row["sample_id"])]
        entry = datasets.setdefault(
            row["dataset"],
            {"samples": 0, "identical": 0, "near_tie": 0, "refused": 0, "unjudgeable": 0},
        )
        entry["samples"] += 1
        if row["answer_sha256"] == other["answer_sha256"]:
            entry["identical"] += 1
            continue
        verdict, fields = judge(args.rmlx_bin, plain_dir / other["file"], spec_dir / row["file"])
        entry[verdict] += 1
        detail.append(
            {
                "dataset": row["dataset"],
                "sample_id": row["sample_id"],
                "verdict": verdict,
                **{k: fields[k] for k in ("position", "margin", "rank") if k in fields},
                "reason": fields.get("reason", ""),
            }
        )
    for entry in datasets.values():
        entry["identical_rate"] = entry["identical"] / entry["samples"]
        entry["identical_or_near_tie_rate"] = (entry["identical"] + entry["near_tie"]) / entry["samples"]
    total = {k: sum(e[k] for e in datasets.values()) for k in ("samples", "identical", "near_tie")}
    result = {
        "against": {
            "arm": "plain",
            "ts_utc": plain_meta["ts_utc"],
            "binary_sha256": plain_meta["binary_sha256"],
        },
        "protocol": PROTOCOL,
        "datasets": datasets,
        "identical_rate": total["identical"] / total["samples"],
        "identical_or_near_tie_rate": (total["identical"] + total["near_tie"]) / total["samples"],
        "differing": detail,
    }
    pathlib.Path(args.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(
        f"published_greedy: {total['identical']} of {total['samples']} answers identical, "
        f"{total['near_tie']} differ at a near tie"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    sealer = sub.add_parser("seal", help="check a greedy pass and write its meta")
    sealer.add_argument("--dir", required=True)
    sealer.add_argument("--arm", required=True, choices=("plain", "speculative"))
    sealer.add_argument("--model-namespace", required=True)
    sealer.add_argument("--model", required=True)
    sealer.add_argument("--weight-quant", required=True)
    sealer.add_argument("--kv-quant", required=True)
    sealer.add_argument("--ctx-max", type=int, required=True)
    sealer.add_argument("--decode-config", default="")
    sealer.add_argument("--binary-sha256", required=True)
    sealer.add_argument("--ts-utc", required=True)

    colder = sub.add_parser("cold", help="check that no greedy answer came from a cache")
    colder.add_argument("--log", required=True, help="the greedy server's run log")
    colder.add_argument("--expect", type=int, required=True, help="requests it answered")

    comparer = sub.add_parser("compare", help="judge a speculative pass against a plain one")
    comparer.add_argument("--plain", required=True, help="the plain run's greedy directory")
    comparer.add_argument("--spec", required=True, help="the speculative run's greedy directory")
    comparer.add_argument("--rmlx-bin", required=True)
    comparer.add_argument("--out", required=True)

    args = ap.parse_args()
    try:
        {"seal": seal, "cold": cold, "compare": compare}[args.command](args)
    except PassError as exc:
        print(f"published_greedy: {exc}", file=sys.stderr)
        return 1
    except (OSError, KeyError, ValueError) as exc:
        print(f"published_greedy: cannot read an input: {exc!r}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
