#!/usr/bin/env python3
"""published_samples.py — build and verify the published-protocol sample sets.

The published speculative-decoding protocol reports output speed as a
macro-average over MT-Bench, MATH-500 and HumanEval subsets. Those subsets are
checked in under ``prompts/published/`` so a number can be traced back to the
exact bytes that produced it, on a machine with no network and no dataset
credentials.

Two subcommands, one implementation of everything they share:

  build   fetch the three upstream files at their pinned revisions (or read
          them from --sources), draw the samples, and write the sample files
          plus prompts/published/manifest.json.
  verify  re-derive from the manifest what build produced and fail loudly on
          any disagreement. Offline by default; --sources additionally checks
          the upstream files still hash to the pinned revision.

The manifest is the recorded truth and this script is its only producer, so
build and verify share the selector, the renderer and the digest functions
below rather than each carrying a copy.

Exit codes:
  0  everything agrees
  1  a recorded fact and the tree disagree
  2  the tree or the manifest could not be read at all (fail closed)
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pathlib
import re
import sys
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PUBLISHED_DIR = REPO_ROOT / "prompts" / "published"
MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1

# One seed for every drawn dataset. The selector mixes the seed with the sample
# id, so disjoint pools do not need disjoint seeds.
SEED = 1729

SELECTION_ALGORITHM = (
    "sha256-rank: order the pool ids by (sha256('<seed>:<id>').hexdigest(), id) "
    "ascending and take the first <count>"
)
DIGEST_CONVENTION = (
    "sha256 of the compact JSON encoding (no spaces, non-ASCII unescaped, object "
    "keys in the order written) of the sample's messages array — the content "
    "address rmlx_metrics gives a prompt body, so a recorded observation and a "
    "checked-in sample share one id. Key order is load-bearing: the workspace "
    "builds serde_json with preserve_order, so re-emitting a message as "
    "{content, role} yields a different id than the {role, content} written here."
)

# Upstream sources, pinned by revision. `pool_id` names the field each upstream
# record is identified by; `user_template` is rendered against that record to
# produce the single user message a request sends.
SOURCES = [
    {
        "key": "mt_bench",
        "name": "mt-bench-80",
        "notes": (
            "MT-Bench question set, first turn of each of the 80 questions. The "
            "whole set is used, so there is nothing to draw."
        ),
        "kind": "github-raw",
        "repo": "lm-sys/FastChat",
        "path": "fastchat/llm_judge/data/mt_bench/question.jsonl",
        "revision": "b494d0c6b4e7935f1764f8439e75da3e66beccc7",
        "encoding": "jsonl",
        "cache_file": "mt_bench.question.jsonl",
        "pool_id": "question_id",
        "count": 80,
        "mode": "all",
        "user_template": "{turns[0]}",
    },
    {
        "key": "math_500",
        "name": "math-500-128",
        "notes": "MATH-500 test split, 128 problems drawn from the 500.",
        "kind": "huggingface-dataset",
        "repo": "HuggingFaceH4/MATH-500",
        "path": "test.jsonl",
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "encoding": "jsonl",
        "cache_file": "math_500.test.jsonl",
        "pool_id": "unique_id",
        "count": 128,
        "mode": "sample",
        "user_template": "{problem}",
    },
    {
        "key": "humaneval",
        "name": "humaneval-128",
        "notes": (
            "HumanEval, 128 problems drawn from the 164. The upstream record is "
            "kept whole so the canonical tests are available to a pass@1 column."
        ),
        "kind": "github-raw",
        "repo": "openai/human-eval",
        "path": "data/HumanEval.jsonl.gz",
        "revision": "463c980b59e818ace59f6f9803cd92c749ceae61",
        "encoding": "jsonl.gz",
        "cache_file": "humaneval.HumanEval.jsonl.gz",
        "pool_id": "task_id",
        "count": 128,
        "mode": "sample",
        "user_template": (
            "Complete the following Python function. Return only the completed "
            "function in a single Python code block, with no explanation.\n\n"
            "```python\n{prompt}\n```"
        ),
    },
]


# ── shared primitives ─────────────────────────────────────────────────────────


def source_url(src: dict) -> str:
    if src["kind"] == "github-raw":
        return f"https://raw.githubusercontent.com/{src['repo']}/{src['revision']}/{src['path']}"
    if src["kind"] == "huggingface-dataset":
        return f"https://huggingface.co/datasets/{src['repo']}/resolve/{src['revision']}/{src['path']}"
    raise ValueError(f"unknown source kind: {src['kind']}")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def body_sha256(messages: list) -> str:
    """Content address of a prompt body, matching the metrics recorder."""
    canonical = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return sha256_hex(canonical.encode("utf-8"))


def select_ids(pool_ids: list, seed: int, count: int) -> list:
    """Draw `count` ids out of `pool_ids`, deterministically, from `seed`.

    Ranking by a hash of seed and id keeps the draw reproducible in any language
    without depending on a particular RNG implementation.
    """
    ranked = sorted(
        pool_ids, key=lambda i: (hashlib.sha256(f"{seed}:{i}".encode()).hexdigest(), i)
    )
    return ranked[:count]


PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?\}")


def render_user_message(template: str, record: dict) -> str:
    """Substitute `{field}` / `{field[i]}` in `template` from `record`.

    Only the template is scanned, so braces inside a value (LaTeX, code) are
    never interpreted.
    """

    def one(m: re.Match) -> str:
        field, index = m.group(1), m.group(2)
        if field not in record:
            raise KeyError(f"template field {field!r} absent from the upstream record")
        value = record[field]
        if index is not None:
            value = value[int(index)]
        return str(value)

    return PLACEHOLDER.sub(one, template)


def decode_records(raw: bytes, encoding: str) -> list:
    if encoding == "jsonl.gz":
        raw = gzip.decompress(raw)
    elif encoding != "jsonl":
        raise ValueError(f"unknown encoding: {encoding}")
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def build_sample(dataset_key: str, source_id: str, record: dict, template: str) -> dict:
    messages = [{"role": "user", "content": render_user_message(template, record)}]
    return {
        "id": f"{dataset_key}/{source_id}",
        "source_id": source_id,
        "messages": messages,
        "body_sha256": body_sha256(messages),
        "source_record": record,
    }


def sample_file_bytes(src: dict, samples: list) -> bytes:
    doc = {
        "name": src["name"],
        "class": "published",
        "dataset": src["key"],
        "notes": src["notes"],
        "samples": samples,
    }
    return (json.dumps(doc, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


# ── build ─────────────────────────────────────────────────────────────────────


def read_upstream(src: dict, sources_dir: pathlib.Path | None) -> bytes:
    if sources_dir is not None:
        path = sources_dir / src["cache_file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.read_bytes()
    with urllib.request.urlopen(source_url(src), timeout=120) as resp:
        return resp.read()


def cmd_build(args: argparse.Namespace) -> int:
    out_dir = pathlib.Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sources_dir = pathlib.Path(args.sources).resolve() if args.sources else None

    datasets = []
    for src in SOURCES:
        raw = read_upstream(src, sources_dir)
        records = decode_records(raw, src["encoding"])
        pool_ids = [str(r[src["pool_id"]]) for r in records]
        if len(set(pool_ids)) != len(pool_ids):
            print(f"ERROR: {src['key']}: upstream ids are not unique", file=sys.stderr)
            return 2

        if src["mode"] == "all":
            if len(pool_ids) != src["count"]:
                print(
                    f"ERROR: {src['key']}: mode 'all' wants {src['count']} records, "
                    f"upstream has {len(pool_ids)}",
                    file=sys.stderr,
                )
                return 2
            selected = list(pool_ids)
        else:
            selected = select_ids(pool_ids, SEED, src["count"])

        by_id = {pid: rec for pid, rec in zip(pool_ids, records)}
        samples = [
            build_sample(src["key"], sid, by_id[sid], src["user_template"])
            for sid in selected
        ]
        blob = sample_file_bytes(src, samples)
        sample_path = out_dir / f"{src['key']}.json"
        sample_path.write_bytes(blob)

        entry = {
            "key": src["key"],
            "name": src["name"],
            "file": sample_path.name,
            "file_bytes": len(blob),
            "file_sha256": sha256_hex(blob),
            "count": src["count"],
            "source": {
                "kind": src["kind"],
                "repo": src["repo"],
                "path": src["path"],
                "revision": src["revision"],
                "url": source_url(src),
                "encoding": src["encoding"],
                "cache_file": src["cache_file"],
                "sha256": sha256_hex(raw),
            },
            "pool_id_field": src["pool_id"],
            "pool_size": len(pool_ids),
            "pool_ids": pool_ids,
            "sampling": (
                {"mode": "all"}
                if src["mode"] == "all"
                else {"mode": "sample", "seed": SEED}
            ),
            "user_template": src["user_template"],
            "selected_ids": selected,
        }
        datasets.append(entry)
        print(f"build: {src['key']}: {len(samples)} samples, {len(blob)} bytes")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "selection_algorithm": SELECTION_ALGORITHM,
        "digest_convention": DIGEST_CONVENTION,
        "datasets": datasets,
    }
    manifest_path = out_dir / MANIFEST_NAME
    manifest_path.write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )
    print(f"build: wrote {manifest_path.name} ({len(datasets)} datasets)")
    return 0


# ── verify ────────────────────────────────────────────────────────────────────


class Report:
    def __init__(self) -> None:
        self.mismatches = 0

    def bad(self, msg: str) -> None:
        print(f"ERROR: {msg}", file=sys.stderr)
        self.mismatches += 1


def verify_dataset(
    entry: dict, root: pathlib.Path, sources_dir: pathlib.Path | None, rep: Report
) -> int:
    """Check one manifest entry. Returns the number of samples checked."""
    key = entry["key"]
    path = root / entry["file"]
    if not path.is_file():
        print(f"ERROR: {key}: sample file {entry['file']} not found", file=sys.stderr)
        return -1
    blob = path.read_bytes()

    if len(blob) != entry["file_bytes"]:
        rep.bad(
            f"{key}: {entry['file']} size changed: manifest records "
            f"{entry['file_bytes']} bytes, file has {len(blob)}"
        )
    elif sha256_hex(blob) != entry["file_sha256"]:
        rep.bad(
            f"{key}: {entry['file']} digest mismatch: manifest records "
            f"{entry['file_sha256']}, file hashes to {sha256_hex(blob)}"
        )

    try:
        doc = json.loads(blob.decode("utf-8"))
        samples = doc["samples"]
    except (ValueError, KeyError, UnicodeDecodeError) as e:
        print(f"ERROR: {key}: {entry['file']} is unreadable: {e}", file=sys.stderr)
        return -1

    pool_ids = entry["pool_ids"]
    if len(pool_ids) != entry["pool_size"]:
        rep.bad(
            f"{key}: pool_ids holds {len(pool_ids)} ids, pool_size records "
            f"{entry['pool_size']}"
        )
    if len(set(pool_ids)) != len(pool_ids):
        rep.bad(f"{key}: pool_ids contains a duplicate id")

    mode = entry["sampling"]["mode"]
    if mode == "all":
        want = list(pool_ids)
    else:
        want = select_ids(pool_ids, entry["sampling"]["seed"], entry["count"])
    if want != entry["selected_ids"]:
        rep.bad(
            f"{key}: selection does not re-derive from the recorded seed — "
            f"{sum(1 for a, b in zip(want, entry['selected_ids']) if a != b)} of "
            f"{entry['count']} positions differ, first divergence at "
            f"{first_divergence(want, entry['selected_ids'])}"
        )

    if len(samples) != entry["count"]:
        rep.bad(
            f"{key}: {entry['file']} holds {len(samples)} samples, manifest records "
            f"{entry['count']}"
        )

    file_ids = [s.get("source_id") for s in samples]
    if len(set(file_ids)) != len(file_ids):
        rep.bad(f"{key}: {entry['file']} contains a duplicate sample id")
    for sid in file_ids:
        if sid not in entry["selected_ids"]:
            rep.bad(
                f"{key}: sample id {sid!r} in {entry['file']} is not in the selected set"
            )
    for sid in entry["selected_ids"]:
        if sid not in file_ids:
            rep.bad(f"{key}: selected id {sid!r} is absent from {entry['file']}")

    for sample in samples:
        sid = sample.get("source_id")
        try:
            rendered = render_user_message(entry["user_template"], sample["source_record"])
        except (KeyError, IndexError, TypeError) as e:
            rep.bad(f"{key}/{sid}: cannot render the recorded template: {e}")
            continue
        want_messages = [{"role": "user", "content": rendered}]
        if sample.get("messages") != want_messages:
            rep.bad(
                f"{key}/{sid}: messages do not render from the recorded template and "
                f"the upstream record"
            )
        got = body_sha256(sample.get("messages", []))
        if got != sample.get("body_sha256"):
            rep.bad(
                f"{key}/{sid}: body digest mismatch: sample records "
                f"{sample.get('body_sha256')}, messages hash to {got}"
            )

    if sources_dir is not None:
        src_path = sources_dir / entry["source"]["cache_file"]
        if not src_path.is_file():
            print(
                f"ERROR: {key}: upstream file {entry['source']['cache_file']} not found "
                f"under --sources",
                file=sys.stderr,
            )
            return -1
        raw = src_path.read_bytes()
        got = sha256_hex(raw)
        if got != entry["source"]["sha256"]:
            rep.bad(
                f"{key}: upstream {entry['source']['cache_file']} differs from the "
                f"pinned revision {entry['source']['revision']}: manifest records "
                f"{entry['source']['sha256']}, file hashes to {got}"
            )
        else:
            records = decode_records(raw, entry["source"]["encoding"])
            upstream_ids = [str(r[entry["pool_id_field"]]) for r in records]
            if upstream_ids != pool_ids:
                rep.bad(
                    f"{key}: pool ids do not match the upstream file at the pinned "
                    f"revision"
                )

    return len(samples)


def first_divergence(a: list, b: list) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"index {i} ({x!r} vs {y!r})"
    return f"index {min(len(a), len(b))} (length {len(a)} vs {len(b)})"


def cmd_verify(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root).resolve()
    sources_dir = pathlib.Path(args.sources).resolve() if args.sources else None
    manifest_path = root / MANIFEST_NAME

    if not manifest_path.is_file():
        print(f"ERROR: manifest {manifest_path} not found", file=sys.stderr)
        return 2
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        print(f"ERROR: manifest is unreadable: {e}", file=sys.stderr)
        return 2

    if manifest.get("schema_version") != SCHEMA_VERSION:
        print(
            f"ERROR: manifest schema_version is {manifest.get('schema_version')!r}, "
            f"this script reads {SCHEMA_VERSION}",
            file=sys.stderr,
        )
        return 2

    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        print("ERROR: manifest declares no datasets — nothing would be checked", file=sys.stderr)
        return 2

    declared = {s["key"] for s in SOURCES}
    listed = {d.get("key") for d in datasets}
    if listed != declared:
        print(
            f"ERROR: manifest datasets {sorted(listed)} are not the sources this "
            f"script builds {sorted(declared)}",
            file=sys.stderr,
        )
        return 2

    rep = Report()
    checked = 0
    for entry in datasets:
        try:
            n = verify_dataset(entry, root, sources_dir, rep)
        except (KeyError, TypeError) as e:
            print(f"ERROR: {entry.get('key')}: manifest entry is malformed: {e}", file=sys.stderr)
            return 2
        if n < 0:
            return 2
        checked += n

    if rep.mismatches:
        print(
            f"check-published-samples: {rep.mismatches} mismatch(es) across "
            f"{len(datasets)} datasets",
            file=sys.stderr,
        )
        return 1

    scope = "offline" if sources_dir is None else "offline + upstream"
    print(
        f"check-published-samples: ok ({checked} samples across "
        f"{len(datasets)} datasets, {scope})"
    )
    return 0


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="fetch, draw and write the sample sets")
    b.add_argument("--out", default=str(PUBLISHED_DIR), help="output directory")
    b.add_argument(
        "--sources",
        default=None,
        help="read the upstream files from this directory instead of the network",
    )
    b.set_defaults(func=cmd_build)

    v = sub.add_parser("verify", help="re-derive the recorded facts and fail on drift")
    v.add_argument("--root", default=str(PUBLISHED_DIR), help="directory to verify")
    v.add_argument(
        "--sources",
        default=None,
        help="also check the upstream files here still hash to the pinned revision",
    )
    v.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
