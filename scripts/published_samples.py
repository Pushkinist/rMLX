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
  verify  re-derive what build produced and fail loudly on any disagreement.
          Offline by default; --sources additionally checks the upstream files
          against the pinned revision, record for record.

WHERE THE ANCHOR LIVES
  A manifest sitting beside the data it describes anchors nothing: an editor who
  changes a sample file can re-bless the manifest in the same edit and every
  internal equality still holds. So the facts that matter — the seed, the
  revision, the template, the sample count, and the digest of each built file —
  are pinned in SOURCES and PINS below, in this script, where changing one is a
  diff someone reads. `verify` checks the manifest against those constants
  before it checks anything against the manifest.

  What that still does not reach: an editor who changes the data *and* the pins
  in one commit. Nothing in a repository can catch that; review can.

Exit codes:
  0  everything agrees
  1  a recorded or pinned fact and the tree disagree
  2  the tree, the manifest or an upstream file could not be read at all
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
SCRIPT_NAME = pathlib.Path(__file__).name
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
        "license": {
            "spdx": "Apache-2.0",
            "holder": "LM-SYS / FastChat contributors",
            "url": "https://github.com/lm-sys/FastChat/blob/main/LICENSE",
            "redistribution": "permitted with attribution and licence text",
        },
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
        "license": {
            "spdx": "MIT",
            "holder": "OpenAI (prm800k split) over Hendrycks et al. MATH",
            "url": "https://github.com/openai/prm800k/blob/main/LICENSE",
            "redistribution": (
                "permitted with attribution and licence text. The Hugging Face "
                "card for HuggingFaceH4/MATH-500 declares no licence of its own; "
                "it republishes the prm800k test split (MIT), which subsets "
                "hendrycks/math (MIT). Both upstream licences permit "
                "redistribution."
            ),
        },
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
        "license": {
            "spdx": "MIT",
            "holder": "OpenAI",
            "url": "https://github.com/openai/human-eval/blob/master/LICENSE",
            "redistribution": "permitted with attribution and licence text",
        },
    },
]

# Digests of what `build` last produced, and of the upstream bytes it read.
# These are the anchor: a sample file can be edited and its manifest re-blessed
# in one move, but not these, without a diff. `build` prints a replacement block
# whenever what it produced no longer matches.
PINS = {
    "mt_bench": {
        "file_bytes": 101830,
        "file_sha256": "5b0a2664bdfbdaaaac6f56ee8734e26bfddc55e34bf56226671843b2d2492041",
        "source_sha256": "119565adbab82227089cefdb44c8d7e2cf04dc0a0ec233634c82e7d4e2a944f7",
    },
    "math_500": {
        "file_bytes": 197309,
        "file_sha256": "1983202141e46f535616f9a024d8050d9f717b4d10f818f2daa8039b6b804764",
        "source_sha256": "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132",
    },
    "humaneval": {
        "file_bytes": 288557,
        "file_sha256": "d915167f554b592be86077d8d66a0bb2416e9616afd7ae6910b13319cc44c294",
        "source_sha256": "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef",
    },
}

EXPECTED_TOTAL = sum(s["count"] for s in SOURCES)


class Unreadable(Exception):
    """Something the gate must read could not be read at all — exit 2."""


# ── shared primitives ─────────────────────────────────────────────────────────


def source_url(src: dict) -> str:
    if src["kind"] == "github-raw":
        return f"https://raw.githubusercontent.com/{src['repo']}/{src['revision']}/{src['path']}"
    if src["kind"] == "huggingface-dataset":
        return f"https://huggingface.co/datasets/{src['repo']}/resolve/{src['revision']}/{src['path']}"
    raise ValueError(f"unknown source kind: {src['kind']}")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_body(messages) -> str:
    """The exact byte-string the content address is taken over."""
    return json.dumps(messages, ensure_ascii=False, separators=(",", ":"))


def body_sha256(messages) -> str:
    """Content address of a prompt body, matching the metrics recorder."""
    return sha256_hex(encode_body(messages).encode("utf-8"))


def select_ids(pool_ids: list, seed, count: int) -> list:
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

    def one(m):
        field, index = m.group(1), m.group(2)
        if field not in record:
            raise KeyError(f"template field {field!r} absent from the upstream record")
        value = record[field]
        if index is not None:
            value = value[int(index)]
        return str(value)

    return PLACEHOLDER.sub(one, template)


def decode_records(raw: bytes, encoding: str, what: str) -> list:
    """Parse an upstream JSONL file. Anything unparseable is Unreadable."""
    try:
        if encoding == "jsonl.gz":
            raw = gzip.decompress(raw)
        elif encoding != "jsonl":
            raise ValueError(f"unknown encoding {encoding!r}")
        return [
            json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()
        ]
    except (OSError, ValueError, UnicodeDecodeError) as e:
        raise Unreadable(f"{what} is unreadable as {encoding}: {e}") from e


def pool_ids_of(records: list, field: str, what: str) -> list:
    try:
        return [str(r[field]) for r in records]
    except (KeyError, TypeError) as e:
        raise Unreadable(f"{what}: a record has no {field!r} field: {e}") from e


def build_sample(dataset_key: str, source_id: str, record: dict, template: str) -> dict:
    messages = [{"role": "user", "content": render_user_message(template, record)}]
    return {
        "id": sample_id(dataset_key, source_id),
        "source_id": source_id,
        "messages": messages,
        "body_sha256": body_sha256(messages),
        "source_record": record,
    }


def sample_id(dataset_key: str, source_id: str) -> str:
    return f"{dataset_key}/{source_id}"


def sample_file_bytes(src: dict, samples: list) -> bytes:
    doc = {
        "name": src["name"],
        "class": "published",
        "dataset": src["key"],
        "notes": src["notes"],
        "samples": samples,
    }
    return (json.dumps(doc, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def source_block(src: dict, raw_sha: str) -> dict:
    return {
        "kind": src["kind"],
        "repo": src["repo"],
        "path": src["path"],
        "revision": src["revision"],
        "url": source_url(src),
        "encoding": src["encoding"],
        "cache_file": src["cache_file"],
        "sha256": raw_sha,
        "license": src["license"],
    }


def sampling_block(src: dict) -> dict:
    if src["mode"] == "all":
        return {"mode": "all"}
    return {"mode": "sample", "seed": SEED}


# ── build ─────────────────────────────────────────────────────────────────────


def read_upstream(src: dict, sources_dir) -> bytes:
    if sources_dir is not None:
        path = sources_dir / src["cache_file"]
        if not path.is_file():
            raise Unreadable(f"upstream file {src['cache_file']} not found under --sources")
        return path.read_bytes()
    with urllib.request.urlopen(source_url(src), timeout=120) as resp:
        return resp.read()


def cmd_build(args) -> int:
    out_dir = pathlib.Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sources_dir = pathlib.Path(args.sources).resolve() if args.sources else None

    datasets = []
    produced = {}
    for src in SOURCES:
        raw = read_upstream(src, sources_dir)
        records = decode_records(raw, src["encoding"], src["cache_file"])
        pool_ids = pool_ids_of(records, src["pool_id"], src["cache_file"])
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

        by_id = dict(zip(pool_ids, records))
        samples = [
            build_sample(src["key"], sid, by_id[sid], src["user_template"])
            for sid in selected
        ]
        blob = sample_file_bytes(src, samples)
        sample_path = out_dir / f"{src['key']}.json"
        sample_path.write_bytes(blob)

        produced[src["key"]] = {
            "file_bytes": len(blob),
            "file_sha256": sha256_hex(blob),
            "source_sha256": sha256_hex(raw),
        }
        datasets.append(
            {
                "key": src["key"],
                "name": src["name"],
                "file": sample_path.name,
                "file_bytes": len(blob),
                "file_sha256": sha256_hex(blob),
                "count": src["count"],
                "source": source_block(src, sha256_hex(raw)),
                "pool_id_field": src["pool_id"],
                "pool_size": len(pool_ids),
                "pool_ids": pool_ids,
                "sampling": sampling_block(src),
                "user_template": src["user_template"],
                "selected_ids": selected,
            }
        )
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

    if produced != PINS:
        print(
            f"\nbuild: what was produced no longer matches PINS in {SCRIPT_NAME}.\n"
            f"Review why the data moved, then replace the PINS block with:\n",
            file=sys.stderr,
        )
        print("PINS = " + json.dumps(produced, indent=4), file=sys.stderr)
        return 1
    return 0


# ── verify ────────────────────────────────────────────────────────────────────


class Report:
    def __init__(self) -> None:
        self.mismatches = 0

    def bad(self, msg: str) -> None:
        print(f"ERROR: {msg}", file=sys.stderr)
        self.mismatches += 1


def check_against_script(src: dict, entry: dict, rep: Report) -> None:
    """Hold the manifest entry to the constants pinned in this script."""
    key = src["key"]
    pin = PINS[key]

    for field, want in (
        ("name", src["name"]),
        ("file", f"{key}.json"),
        ("count", src["count"]),
        ("pool_id_field", src["pool_id"]),
        ("user_template", src["user_template"]),
        ("file_bytes", pin["file_bytes"]),
        ("file_sha256", pin["file_sha256"]),
    ):
        if entry.get(field) != want:
            rep.bad(
                f"{key}: manifest {field} is {entry.get(field)!r} and disagrees with "
                f"{SCRIPT_NAME}, which pins {want!r}"
            )

    want_source = source_block(src, pin["source_sha256"])
    for field, want in want_source.items():
        if entry.get("source", {}).get(field) != want:
            rep.bad(
                f"{key}: manifest source.{field} is "
                f"{entry.get('source', {}).get(field)!r} and disagrees with "
                f"{SCRIPT_NAME}, which pins {want!r}"
            )

    if entry.get("sampling") != sampling_block(src):
        rep.bad(
            f"{key}: manifest sampling is {entry.get('sampling')!r} and disagrees with "
            f"{SCRIPT_NAME}, which pins {sampling_block(src)!r}"
        )


def verify_dataset(src: dict, entry: dict, root, sources_dir, rep: Report) -> int:
    """Check one manifest entry. Returns the number of samples checked."""
    key = src["key"]
    check_against_script(src, entry, rep)

    path = root / f"{key}.json"
    if not path.is_file():
        raise Unreadable(f"{key}: sample file {key}.json not found")
    blob = path.read_bytes()
    pin = PINS[key]

    if len(blob) != pin["file_bytes"]:
        rep.bad(
            f"{key}: {key}.json size changed: {SCRIPT_NAME} pins {pin['file_bytes']} "
            f"bytes, file has {len(blob)}"
        )
    elif sha256_hex(blob) != pin["file_sha256"]:
        rep.bad(
            f"{key}: {key}.json digest mismatch: {SCRIPT_NAME} pins "
            f"{pin['file_sha256']}, file hashes to {sha256_hex(blob)}"
        )

    try:
        doc = json.loads(blob.decode("utf-8"))
        samples = doc["samples"]
    except (ValueError, KeyError, UnicodeDecodeError) as e:
        raise Unreadable(f"{key}: {key}.json is unreadable: {e}") from e
    if not isinstance(samples, list):
        raise Unreadable(f"{key}: {key}.json samples is not a list")
    for i, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise Unreadable(f"{key}: {key}.json sample {i} is not an object")

    pool_ids = entry["pool_ids"]
    if len(pool_ids) != entry["pool_size"]:
        rep.bad(
            f"{key}: pool_ids holds {len(pool_ids)} ids, pool_size records "
            f"{entry['pool_size']}"
        )
    if len(set(pool_ids)) != len(pool_ids):
        rep.bad(f"{key}: pool_ids contains a duplicate id")

    if entry["sampling"]["mode"] == "all":
        want = list(pool_ids)
    else:
        want = select_ids(pool_ids, entry["sampling"].get("seed"), entry["count"])
    if want != entry["selected_ids"]:
        rep.bad(
            f"{key}: selection does not re-derive from the recorded seed — "
            f"{sum(1 for a, b in zip(want, entry['selected_ids']) if a != b)} of "
            f"{entry['count']} positions differ, first divergence at "
            f"{first_divergence(want, entry['selected_ids'])}"
        )

    if len(samples) != entry["count"]:
        rep.bad(
            f"{key}: {key}.json holds {len(samples)} samples, manifest records "
            f"{entry['count']}"
        )

    file_ids = [s.get("source_id") for s in samples]
    if len(set(file_ids)) != len(file_ids):
        rep.bad(f"{key}: {key}.json contains a duplicate sample id")
    for sid in file_ids:
        if sid not in entry["selected_ids"]:
            rep.bad(f"{key}: sample id {sid!r} in {key}.json is not in the selected set")
    for sid in entry["selected_ids"]:
        if sid not in file_ids:
            rep.bad(f"{key}: selected id {sid!r} is absent from {key}.json")

    for sample in samples:
        check_sample(key, entry, sample, rep)

    if sources_dir is not None:
        check_upstream(src, entry, samples, pool_ids, sources_dir, rep)

    return len(samples)


def check_sample(key: str, entry: dict, sample: dict, rep: Report) -> None:
    sid = sample.get("source_id")
    if sample.get("id") != sample_id(key, str(sid)):
        rep.bad(
            f"{key}/{sid}: sample id {sample.get('id')!r} does not derive from its "
            f"source_id — expected {sample_id(key, str(sid))!r}"
        )
    record = sample.get("source_record")
    if not isinstance(record, dict):
        rep.bad(f"{key}/{sid}: source_record is not an object")
        return
    try:
        rendered = render_user_message(entry["user_template"], record)
    except (KeyError, IndexError, TypeError) as e:
        rep.bad(f"{key}/{sid}: cannot render the recorded template: {e}")
        return
    want_messages = [{"role": "user", "content": rendered}]
    got_messages = sample.get("messages")
    # Compare the encoding, not the parsed value: the content address is taken
    # over these exact bytes, so a message re-emitted as {content, role} is a
    # different prompt id even though the two dicts compare equal.
    if encode_body(got_messages) != encode_body(want_messages):
        if got_messages == want_messages:
            rep.bad(
                f"{key}/{sid}: messages carry the right values in the wrong key order "
                f"— the content address is taken over the encoding, so this is a "
                f"different prompt id"
            )
        else:
            rep.bad(
                f"{key}/{sid}: messages do not render from the recorded template and "
                f"the sample's copy of the upstream record"
            )
    got = body_sha256(got_messages)
    if got != sample.get("body_sha256"):
        rep.bad(
            f"{key}/{sid}: body digest mismatch: sample records "
            f"{sample.get('body_sha256')}, messages hash to {got}"
        )


def check_upstream(
    src: dict, entry: dict, samples: list, pool_ids: list, sources_dir, rep: Report
) -> None:
    """Hold the samples to the upstream file at the pinned revision."""
    key = src["key"]
    cache_file = src["cache_file"]
    src_path = sources_dir / cache_file
    if not src_path.is_file():
        raise Unreadable(f"{key}: upstream file {cache_file} not found under --sources")
    raw = src_path.read_bytes()
    got = sha256_hex(raw)
    if got != PINS[key]["source_sha256"]:
        rep.bad(
            f"{key}: upstream {cache_file} differs from the pinned revision "
            f"{src['revision']}: {SCRIPT_NAME} pins "
            f"{PINS[key]['source_sha256']}, file hashes to {got}"
        )
        return

    records = decode_records(raw, src["encoding"], f"{key}: upstream {cache_file}")
    upstream_ids = pool_ids_of(records, src["pool_id"], f"{key}: upstream {cache_file}")
    if upstream_ids != pool_ids:
        rep.bad(f"{key}: pool ids do not match the upstream file at the pinned revision")
        return

    by_id = dict(zip(upstream_ids, records))
    for sample in samples:
        sid = str(sample.get("source_id"))
        if sid not in by_id:
            rep.bad(f"{key}/{sid}: no such record in the upstream file")
            continue
        if sample.get("source_record") != by_id[sid]:
            rep.bad(
                f"{key}/{sid}: source_record differs from the upstream record at the "
                f"pinned revision — the checked-in copy is not what was published"
            )


def first_divergence(a: list, b: list) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"index {i} ({x!r} vs {y!r})"
    return f"index {min(len(a), len(b))} (length {len(a)} vs {len(b)})"


def cmd_verify(args) -> int:
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
        print(
            "ERROR: manifest declares no datasets — nothing would be checked",
            file=sys.stderr,
        )
        return 2

    by_key = {}
    for entry in datasets:
        if not isinstance(entry, dict):
            print("ERROR: a manifest dataset entry is not an object", file=sys.stderr)
            return 2
        by_key[entry.get("key")] = entry
    if set(by_key) != {s["key"] for s in SOURCES}:
        print(
            f"ERROR: manifest datasets {sorted(k for k in by_key if k)} are not the "
            f"sources {SCRIPT_NAME} builds {sorted(s['key'] for s in SOURCES)}",
            file=sys.stderr,
        )
        return 2

    rep = Report()
    checked = 0
    try:
        for src in SOURCES:
            checked += verify_dataset(src, by_key[src["key"]], root, sources_dir, rep)
    except Unreadable as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except (KeyError, TypeError) as e:
        print(f"ERROR: a manifest entry is malformed: {e}", file=sys.stderr)
        return 2

    if checked != EXPECTED_TOTAL:
        rep.bad(
            f"checked {checked} samples, {SCRIPT_NAME} pins {EXPECTED_TOTAL} — a scan "
            f"that reads fewer proves less, whatever else agreed"
        )

    if rep.mismatches:
        print(
            f"check-published-samples: {rep.mismatches} mismatch(es) across "
            f"{len(datasets)} datasets",
            file=sys.stderr,
        )
        return 1

    scope = "offline" if sources_dir is None else "offline + upstream records"
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
        help="also hold the samples to the upstream files here, record for record",
    )
    v.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    try:
        return args.func(args)
    except Unreadable as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
