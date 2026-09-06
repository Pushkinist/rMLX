#!/usr/bin/env python3
"""Render published-protocol runs as the Markdown table a post needs.

Reads one `spec_bench_published.sh` result file per engine mode — plain, and
each speculative arm — and prints Markdown. Writes no DB row, serves nothing,
never launches a model. The only thing it runs is `scripts/perf_ceiling.py`,
itself a static census over the snapshot's `config.json` and safetensors
headers.

EVERY NUMBER CARRIES ITS BOUND.

A measured rate on its own says nothing about how close to the machine the
engine got, and a rate published beside a third party's is read as if it did.
So each measured figure is printed next to the figure it cannot exceed:

  decode rate      the bandwidth-bound autoregressive ceiling for that cell —
                   weights streamed per step plus the KV bytes that cell's
                   codec reads, over the host's memory bandwidth.
  resident memory  the floor: the weights text decode must hold plus the KV the
                   cache holds at that context.
  tokens per round the block the drafter was configured with. `tokens_per_round`
                   is `1 + accept_rate x (block - 1)` while every round drafts
                   the full block, so `block` is its maximum and the quotient is
                   the fraction of the drafted block the verifier kept
                   (docs/SPECULATIVE.md).

`perf_ceiling.py` is the existing instrument and its KV byte model is held to
the engine's own by `make check-kv-byte-model-parity`, so the KV term is not a
second opinion. This script derives no ceiling of its own; it chooses the
contexts to evaluate one at and divides.

WHAT perf_ceiling.py CANNOT EXPRESS, AND WHAT IS PRINTED INSTEAD.

It models one autoregressive forward per token. A speculative round runs one
verifier forward over a whole block and keeps a prefix of it, and the script has
no drafter, no block and no accept rate — there is no speculative ceiling in it
to ask for. So a speculative arm is printed against the SAME autoregressive
ceiling, the column says so, and a value above 100% is expected rather than a
defect: exceeding the autoregressive bound is the point of speculative decoding.
No speculative bound is invented here.

It also reports no prefill ceiling without a measured anchor row in `runs.db`,
by design — a single achieved-GEMM constant is not defensible on this host. The
input-speed bound is then printed as absent, not as a guess.

WHICH CONTEXT THE CEILING IS EVALUATED AT.

A cell's decode window runs from the prompt's length to the prompt plus the
completion, and the ceiling falls across it as the KV grows. One column needs
one context, so it is the middle of that window — and the window's two ends are
evaluated too: if the ceiling moves across the window by more than
`--ceiling-spread-pct`, one number does not describe the cell and this script
refuses rather than printing the midpoint as though it did.

WHICH HOST THE CEILING BELONGS TO.

`perf_ceiling.py`'s bandwidth constant was measured on one machine and names it.
A measurement from another machine divided by that ceiling is a percentage of
nothing, and it renders exactly as plausibly as a right one, so the result's
`hardware_tag` must match — or `--bandwidth-gbs` must state the other host's
bandwidth, which makes the substitution deliberate and visible.

WHAT SEVERAL RESULTS MUST SHARE TO BE ONE TABLE.

Rows from different runs are compared down a column, so every input has to be
the same measurement with one thing changed: the same checkpoint, the same KV
codec, the same context ceiling, the same protocol constants, the same cells and
the same binary. A `x plain` column across two binaries is not a speedup, and
nothing in the rendered table would say so.

Exit codes: 0 — rendered; 1 — an input could not be read, or the inputs do not
describe one comparison; 3 — a ceiling that cannot describe a cell it is asked
to bound.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_LIB = Path(__file__).resolve().parent
sys.path.insert(0, str(_LIB))
sys.path.insert(0, str(_LIB.parent))

from perf_ceiling import HOST_BW_BYTES_PER_S, HOST_BW_HARDWARE_TAG  # noqa: E402
from snapshot_identity import split_model_dir  # noqa: E402

PERF_CEILING = _LIB.parent / "perf_ceiling.py"

# Datasets in the order a reader expects, ahead of any cell a run adds.
DATASET_ORDER = ("mt_bench", "math_500", "humaneval")

# Fields every input must agree on to be rows of one table.
SHARED_FIELDS = ("model_namespace", "model", "weight_quant", "kv_quant", "ctx_max",
                 "hardware_tag", "range_refusal_pct")
SHARED_PROTOCOL = ("passes", "warmups_per_pass", "macro_max_tokens", "thinking",
                   "seed_policy")

# The `<drafter>/depth=` term the round loop puts on its own `done` line when it
# resized its block instead of drafting the configured one (docs/METRICS_DB.md).
# Reading it here is reading the engine's statement, not keeping a second list
# of which drafters are adaptive.
ADAPTIVE_MARKER = "/depth="


class InputError(Exception):
    """An input could not be read, or the inputs are not one comparison."""


class CeilingError(Exception):
    """A ceiling that cannot describe the cell it is asked to bound."""


def arm_label(result: dict) -> str:
    """How this run is named in a column: the engine's own cell key when it has
    one, and the bare arm when it does not."""
    return result.get("decode_config") or result["arm"]


# ── the ceiling ───────────────────────────────────────────────────────────────


def ceilings_at(snapshot: Path, kv_quant: str, max_ctx: int, ctxs: list[int],
                bandwidth_gbs: float, runs_db: str | None) -> dict[int, dict]:
    """`perf_ceiling.py` once, over every context this table needs."""
    cmd = [
        sys.executable, str(PERF_CEILING),
        "--model", str(snapshot),
        "--kv-quant", kv_quant,
        "--max-ctx", str(max_ctx),
        "--bandwidth-gbs", f"{bandwidth_gbs:.6f}",
        "--json",
    ]
    for ctx in ctxs:
        cmd += ["--ctx", str(ctx)]
    cmd += ["--runs-db", runs_db] if runs_db else ["--no-db"]
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise InputError(
            "perf_ceiling.py could not price this checkpoint "
            f"(exit {done.returncode}): {done.stderr.strip() or done.stdout.strip()}"
        )
    try:
        res = json.loads(done.stdout)
    except json.JSONDecodeError as exc:
        raise InputError(f"perf_ceiling.py printed no JSON: {exc}") from exc
    return {row["ctx"]: row for row in res["rows"]}


def window(prompt_tokens: float, completion_tokens: float) -> tuple[int, int, int]:
    """A decode window's start, middle and end, in context tokens."""
    start = max(1, round(prompt_tokens))
    end = max(start, round(prompt_tokens + completion_tokens))
    return start, (start + end) // 2, end


def bound_for(rows: dict[int, dict], where: str, ctxs: tuple[int, int, int],
              spread_pct: float) -> dict:
    """The ceiling at the middle of a decode window, once it describes the whole
    window. A ceiling that moved across the window by more than the band is not
    one number, and printing the midpoint would hide that it is not."""
    start, mid, end = ctxs
    lo, at, hi = (rows[c] for c in (start, mid, end))
    moved = abs(lo["ceiling_tps"] - hi["ceiling_tps"]) / at["ceiling_tps"] * 100.0
    if moved > spread_pct:
        raise CeilingError(
            f"{where}: the ceiling falls from {lo['ceiling_tps']:.2f} tok/s at "
            f"{start} context to {hi['ceiling_tps']:.2f} at {end}, {moved:.1f}% of "
            f"the midpoint and past the {spread_pct:.0f}% band one number is "
            "allowed to describe a window over. Report this cell per context "
            "rather than against a single ceiling."
        )
    return {"ctx": mid, "ceiling_tps": at["ceiling_tps"], "spread_pct": moved,
            "resident_floor_bytes": at["resident_floor_bytes"],
            "resident_weight_bytes": at["resident_weight_bytes"],
            "kv_resident_bytes": at["kv_resident_bytes"],
            "prefill_ceiling_tps": at["prefill_ceiling_tps"]}


# ── rendering helpers ─────────────────────────────────────────────────────────


def ratio_pct(part: float | None, whole: float | None) -> str:
    """`part` as a percent of `whole`. Used both ways round: a rate against the
    ceiling it cannot pass, and a floor against the resident peak it sits under."""
    if part is None or not whole:
        return "—"
    return f"{part / whole * 100:.1f}%"


def gb(value: float | None) -> str:
    return "—" if value is None else f"{value / 1e9:.2f}"


def measured_rate(entry: dict) -> float | None:
    """The mean, or nothing at all when the run-to-run range refused it."""
    return entry["mean"] if entry.get("stable", True) else None


def rate_cell(entry: dict) -> str:
    value = measured_rate(entry)
    return "**UNSTABLE**" if value is None else f"{value:.2f}"


def cell_order(cells: dict) -> list[str]:
    def key(name):
        entry = cells[name]
        dataset = entry["dataset"]
        rank = (DATASET_ORDER.index(dataset) if dataset in DATASET_ORDER
                else len(DATASET_ORDER))
        return (rank, dataset, entry["max_tokens"])
    return sorted(cells, key=key)


# ── document sections ─────────────────────────────────────────────────────────


SYNTHETIC_REASON = (
    "the arms were **synthetic**. A stub answered every request and no model was "
    "ever served, so every rate under those modes is a fixture value and bounds "
    "nothing."
)
UNVERIFIED_REASON = (
    "the samples came from an **unverified root**, not the checked-in "
    "`prompts/published/`, so those runs are not published measurements."
)


def banner(results: list[dict]) -> list[str]:
    """Why these numbers must not be read as a measurement — or nothing."""
    reasons: dict[str, list[str]] = {}
    for result in results:
        label = arm_label(result)
        if result.get("synthetic_arms"):
            reasons.setdefault(SYNTHETIC_REASON, []).append(label)
        if result.get("unverified_samples"):
            reasons.setdefault(UNVERIFIED_REASON, []).append(label)
        taint = (result.get("host") or {}).get("taint") or ""
        if taint:
            reasons.setdefault(f"the run is **tainted** — {taint}.", []).append(label)
    if not reasons:
        return []
    bullets = [
        "> - " + ", ".join(f"`{m}`" for m in modes) + f": {text}"
        for text, modes in reasons.items()
    ]
    return (["> [!WARNING]", "> **THIS TABLE HOLDS NO PUBLISHABLE MEASUREMENT.**", ">"]
            + bullets + [""])


def protocol_block(result: dict) -> list[str]:
    p = result["protocol"]
    sampling = p.get("sampling_resolved")
    sampling_text = (
        ", ".join(f"`{k}`={v}" for k, v in sorted(sampling.items()))
        if sampling else "greedy — the checkpoint resolved no sampler"
    )
    poll = (result.get("fixed_prompt") or {}).get("memory_poll_ms", "—")
    return [
        "## The protocol, and the parts of it we chose",
        "",
        "The published on-device protocol leaves several things unstated. They are",
        "pinned here and printed with the numbers rather than left to a reader to",
        "assume.",
        "",
        "| choice | value |",
        "|---|---|",
        f"| max output tokens | {p['macro_max_tokens']} for every dataset; MATH-500 "
        "also at 4096, as a column beside the headline |",
        f"| thinking tokens | {p['thinking']} |",
        f"| warmup | {p['warmups_per_pass']} untimed request per pass, on a prompt "
        "in no sample set |",
        f"| passes | {p['passes']} consecutive, each score their mean |",
        f"| resident memory | peak `phys_footprint` (`docs/PROFILING.md` §9), "
        f"sampled every {poll} ms, so it is a lower bound on the true peak |",
        "| sampling | the checkpoint's own — the request carries no sampling field. "
        f"Read back from the engine: {sampling_text} |",
        f"| seed | {p['seed_policy']} |",
        "| run-to-run range refusal | a mean whose three passes span more than "
        f"{result['range_refusal_pct']}% of the mean is withheld, not averaged |",
        "",
        "### Three things a reader comparing this to a published figure must know",
        "",
        "1. **MT-Bench questions are two-turn; only the first turn is measured.**",
        "   The second turn is preserved verbatim in",
        "   `prompts/published/mt_bench.json` and is simply not sent, so nothing is",
        "   lost — but a published MT-Bench figure that measures both turns is",
        "   measuring a longer context than this one, and the two are not",
        "   interchangeable.",
        f"2. **The macro average is one cell per dataset**, at the "
        f"{p['macro_max_tokens']}-token budget. MATH-500's 4096-token cell is a",
        "   column beside the headline, not a fourth dataset: folding it in would",
        "   give MATH-500 twice the weight of the other two.",
        "3. **The seed is held fixed across all three passes**, so the run-to-run",
        "   range is a reading of machine stability and not of sampling variance.",
        "   That is checked rather than asserted — `diverged` counts the samples",
        "   that did not generate the same length in all three passes, and for",
        "   those the range carries sampling variance too.",
        "",
    ]


def ceiling_block(result: dict, meta: dict) -> list[str]:
    return [
        "## Where the bounds come from",
        "",
        "Every measured figure is printed beside the figure it cannot exceed. The",
        "bounds are `scripts/perf_ceiling.py` — a static census over the snapshot's",
        "`config.json` and safetensors headers, no GPU and no model — at",
        f"{meta['bandwidth_gbs']:.0f} GB/s on `{meta['hardware_tag']}`, with the KV",
        f"priced at `{result['kv_quant']}` and a ring preallocated to `--max-ctx "
        f"{result['ctx_max']}`. Its KV byte model is held to the engine's own by",
        "`make check-kv-byte-model-parity`, so the KV term is not a second opinion.",
        "",
        "- **decode ceiling** — weights streamed per step plus the KV bytes a step",
        "  reads, over the host's memory bandwidth. Evaluated at the middle of each",
        "  cell's decode window; the `ctx` column names it.",
        "- **resident floor** — the weights text decode must hold plus the KV the",
        "  cache holds at that context.",
        "- **there is no speculative ceiling here.** `perf_ceiling.py` models one",
        "  autoregressive forward per token and has no drafter, no block and no",
        "  accept rate. A speculative arm is printed against the *same*",
        "  autoregressive ceiling, and a value above 100% is the point of",
        "  speculative decoding rather than a defect. None is invented.",
        "- **percent-of-ceiling is not scale-free.** It is",
        "  `1 / (1 + overhead / ideal)`, so the same fixed per-step cost reads worse",
        "  on a small model than on a large one. Compare it down a column, within",
        "  one model — not across models.",
        "",
    ]


def decode_table(results: list[dict], bounds: dict, plain: dict | None) -> list[str]:
    first = results[0]
    cells = first["cells"]
    out = [
        f"## Output speed — `{first['model_namespace']}/{first['model']}` "
        f"({first['weight_quant']}, KV `{first['kv_quant']}`)",
        "",
        f"| cell | mode | samples | max out | t/s (mean of "
        f"{first['protocol']['passes']}) | range % | AR ceiling t/s | % of AR "
        "ceiling | × plain | ctx | worst sample % | diverged |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in cell_order(cells):
        for result in results:
            entry = result["cells"][name]
            bound = bounds[(arm_label(result), name)]
            base = (measured_rate(plain["cells"][name]) if plain is not None else None)
            value = measured_rate(entry)
            speedup = (f"{value / base:.3f}" if base and value else "—")
            out.append(
                f"| `{name}` | `{arm_label(result)}` | {entry['samples']} "
                f"| {entry['max_tokens']} | {rate_cell(entry)} "
                f"| {entry['range_pct']:.2f} | {bound['ceiling_tps']:.2f} "
                f"| {ratio_pct(value, bound['ceiling_tps'])} | {speedup} "
                f"| {bound['ctx']} | {entry['sample_range_pct_max']:.1f} "
                f"| {entry['divergent_samples']} of {entry['samples']} |"
            )
    for result in results:
        macro = result["macro"]
        names = macro["cells"]
        macro_bound = sum(
            bounds[(arm_label(result), n)]["ceiling_tps"] for n in names
        ) / len(names)
        base = (measured_rate(plain["macro"]) if plain is not None else None)
        value = measured_rate(macro)
        speedup = f"{value / base:.3f}" if base and value else "—"
        out.append(
            f"| **MACRO** | `{arm_label(result)}` | {len(names)} cells "
            f"| {macro['max_tokens']} | {rate_cell(macro)} "
            f"| {macro['range_pct']:.2f} | {macro_bound:.2f} "
            f"| {ratio_pct(value, macro_bound)} | {speedup} | — | — | — |"
        )
    out += [
        "",
        f"MACRO is the mean over {', '.join('`' + c + '`' for c in first['macro']['cells'])}",
        "— one cell per dataset. Its ceiling column is the mean of those cells'",
        "ceilings, so both sides of the ratio are averaged the same way.",
        "`range %` is over the three pass means. `worst sample %` is the widest",
        "across-pass range of any one sample, which a pass-mean range cannot see;",
        "it is reported and never refused, because at a sampled temperature one",
        "prompt generates different text of different length on each pass.",
        "",
    ]
    return out


def round_table(results: list[dict]) -> list[str]:
    """How much of each drafted block the verifier kept."""
    speculative = [r for r in results if r.get("block_size")
                   and any("tokens_per_round" in c for c in r["cells"].values())]
    if not speculative:
        return []
    out = [
        "## The round loop",
        "",
        "`tokens_per_round` is `1 + accept_rate × (block − 1)` while every round",
        "drafts the configured block, so `block` is its maximum and `block kept` is",
        "the fraction of the drafted block the verifier kept (`docs/SPECULATIVE.md`).",
        "A loop that resized its block instead says so on its own `done` line, and",
        "for those `block kept` is left empty rather than quoting a fraction of a",
        "block the drafter did not always propose.",
        "",
        "| cell | mode | tokens/round | block | block kept | accepted/step | accept rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in speculative:
        block = result["block_size"]
        adaptive = ADAPTIVE_MARKER in (result.get("decode_config") or "")
        for name in cell_order(result["cells"]):
            entry = result["cells"][name]
            tpr = entry.get("tokens_per_round")
            if tpr is None:
                continue
            aps = entry.get("accepted_per_step")
            rate = entry.get("accept_rate")
            kept = "— (adaptive block)" if adaptive else f"{tpr / block * 100:.1f}%"
            out.append(
                f"| `{name}` | `{arm_label(result)}` | {tpr:.3f} | {block} | {kept} "
                f"| {'—' if aps is None else f'{aps:.3f}'} "
                f"| {'—' if rate is None else f'{rate:.3f}'} |"
            )
    out.append("")
    return out


def fixed_table(result: dict, bound: dict) -> list[str]:
    block = result["fixed_prompt"]
    prefill_bound = bound["prefill_ceiling_tps"]
    footprint = block["phys_footprint_bytes"]["max"]
    out = [
        f"## The fixed-length prompt — {block['prompt_tokens']} tokens, plain decode",
        "",
        f"One prompt of a stated length, {block['max_tokens']} output budget, three",
        "runs. The protocol's second figure is the *autoregressive* one, so this",
        "block is not measured on a speculative arm at all. The body is cut from",
        f"`{block['corpus']}` to hit the token target exactly against this",
        "checkpoint's tokenizer, so it is not checked in — it travels in the result",
        "file with the measurement.",
        "",
        "| figure | measured | bound | % of bound |",
        "|---|---:|---:|---:|",
        f"| output speed (tok/s) | {rate_cell(block['decode_tps'])} "
        f"| {bound['ceiling_tps']:.2f} "
        f"| {ratio_pct(measured_rate(block['decode_tps']), bound['ceiling_tps'])} |",
        f"| input speed (tok/s) | {rate_cell(block['prefill_tps'])} "
        f"| {'—' if not prefill_bound else f'{prefill_bound:.0f}'} "
        f"| {ratio_pct(measured_rate(block['prefill_tps']), prefill_bound)} |",
        f"| peak `phys_footprint` (GB) | {gb(footprint)} "
        f"| {gb(bound['resident_floor_bytes'])} "
        f"| {ratio_pct(bound['resident_floor_bytes'], footprint)} |",
        f"| peak RSS (GB) | {gb(block['rss_bytes']['max'])} "
        f"| {gb(bound['resident_floor_bytes'])} "
        f"| {ratio_pct(bound['resident_floor_bytes'], block['rss_bytes']['max'])} |",
        "",
        "The two resident rows read the other way round — their bound is a",
        "**floor**, so the last column is how much of what the process held is",
        f"accounted for by the {gb(bound['resident_weight_bytes'])} GB of weights",
        f"text decode must hold plus {gb(bound['kv_resident_bytes'])} GB of KV at",
        f"{bound['ctx']} context. The remainder is allocator slack, activations,",
        "the prompt cache and everything else a process holds; it is not waste and",
        "this does not say it is. Both peaks are a sampled gauge, so both are a",
        "lower bound on the true peak.",
    ]
    if not prefill_bound:
        out += [
            "",
            "The input-speed bound is empty on purpose. `perf_ceiling.py` projects",
            "prefill from a measured anchor row in `runs.db` and reports nothing at",
            "all rather than guessing when it has none: a single achieved-GEMM",
            "constant is not defensible on this host, where the recorded rows span a",
            "7× range across models.",
        ]
    out.append("")
    return out


# The columns the protocol omits and this harness set out to publish beside it,
# each keyed on the result field it would arrive in. A section listing what is
# missing is only honest while it is derived from the results rather than
# asserted, so a run that carries one of these drops its line here — and then
# has to grow a section of its own.
EXTRA_COLUMNS = {
    "greedy_match": "greedy token-match rate against plain decode on the same "
                    "checkpoint, per dataset — the lossless proof",
    "longctx": "TTFT and decode rate at 32k and 128k input, plain, from "
               "`prompts/longctx_*.json`",
    "wikitext_ppl": "Wikitext-2 perplexity of the served checkpoint against its "
                    "bf16/mxfp8 sibling",
    "humaneval_pass1": "pass@1 on the HumanEval subset, from these same "
                       "completions",
}


def missing_columns_block(results: list[dict]) -> list[str]:
    """What a reader should not go looking for in the table above."""
    missing = [text for key, text in EXTRA_COLUMNS.items()
               if not any(r.get(key) for r in results)]
    if not missing:
        return []
    return [
        "## Not in this table",
        "",
        "The protocol above omits these and they are what would make the",
        "comparison more than a rate comparison. This harness does not measure",
        "them yet, so they are named here rather than left for a reader to notice",
        "their absence:",
        "",
    ] + [f"- {text}" for text in missing] + [
        "",
        "Thermal state and binary identity, the other two, are under Provenance.",
        "",
    ]


def provenance_block(results: list[dict], snapshot: Path) -> list[str]:
    first = results[0]
    binary = first.get("binary") or {}
    out = [
        "## Provenance",
        "",
        "| field | value |",
        "|---|---|",
        f"| backend | {first.get('backend', '—')} "
        f"{first.get('backend_version', '')} ({first.get('build_profile', '—')}) |",
        f"| binary | `sha256:{binary.get('sha256', '—')[:16]}` |",
        f"| snapshot | `{snapshot.name}` |",
        f"| KV codec | `{first['kv_quant']}` (read back from the engine) |",
        f"| hardware | `{first.get('hardware_tag', '—')}` |",
        "",
        "| mode | run | thermal | host interference |",
        "|---|---|---|---|",
    ]
    for result in results:
        host = result.get("host") or {}
        out.append(
            f"| `{arm_label(result)}` | {result.get('ts_utc', '—')} "
            f"| {'; '.join(dict.fromkeys(host.get('thermal') or [])) or '—'} "
            f"| {'; '.join(dict.fromkeys(host.get('pass_windows') or [])) or '—'} |"
        )
    out += [
        "",
        "Every per-sample row behind these means is recordable into `runs.db` by",
        "`scripts/ingest/published_ingest.py`, which is a separate and explicit",
        "step: a measurement and a record are different acts, and `observations`",
        "is append-only.",
        "",
    ]
    return out


def render(results: list[dict], snapshot: Path, bounds: dict, meta: dict,
           command: str) -> str:
    plain = next((r for r in results if r["arm"] == "plain"), None)
    lines = [
        "<!-- GENERATED FILE — do not edit by hand. -->",
        f"<!-- Regenerate: {command} -->",
        "",
        "# Published-protocol speculative-decoding results",
        "",
    ]
    lines += banner(results)
    lines += [
        "Measured by `scripts/spec_bench_published.sh` under the protocol",
        "third-party on-device speculative-decoding posts report, so these numbers",
        "can sit beside theirs. Bounds by `scripts/perf_ceiling.py`.",
        "",
    ]
    lines += protocol_block(results[0])
    lines += ceiling_block(results[0], meta)
    lines += decode_table(results, bounds, plain)
    lines += round_table(results)
    for result in results:
        if result.get("fixed_prompt"):
            lines += fixed_table(result, bounds[(arm_label(result), "#fixed")])
    lines += missing_columns_block(results)
    lines += provenance_block(results, snapshot)
    return "\n".join(lines).rstrip() + "\n"


# ── driver ────────────────────────────────────────────────────────────────────


def load(paths: list[str]) -> list[dict]:
    results = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            results.append(json.load(handle))
    labels = [arm_label(r) for r in results]
    if len(set(labels)) != len(labels):
        raise InputError(
            f"two of these results are the same engine mode ({sorted(labels)}); "
            "a table with one mode twice compares a run against itself"
        )
    first = results[0]
    for result in results[1:]:
        for field in SHARED_FIELDS:
            if result.get(field) != first.get(field):
                raise InputError(
                    f"`{arm_label(result)}` has {field}={result.get(field)!r} where "
                    f"`{arm_label(first)}` has {first.get(field)!r}; these rows are "
                    "compared down a column and that is two different measurements"
                )
        for field in SHARED_PROTOCOL:
            if result["protocol"].get(field) != first["protocol"].get(field):
                raise InputError(
                    f"`{arm_label(result)}` ran the protocol with "
                    f"{field}={result['protocol'].get(field)!r} where "
                    f"`{arm_label(first)}` used {first['protocol'].get(field)!r}"
                )
        a = (result.get("binary") or {}).get("sha256")
        b = (first.get("binary") or {}).get("sha256")
        if a != b:
            raise InputError(
                f"`{arm_label(result)}` was measured on binary sha256:{str(a)[:16]} "
                f"and `{arm_label(first)}` on sha256:{str(b)[:16]}; a speedup "
                "between two binaries is not a speedup between two engine modes"
            )
        if set(result["cells"]) != set(first["cells"]):
            raise InputError(
                f"`{arm_label(result)}` measured cells {sorted(result['cells'])} "
                f"where `{arm_label(first)}` measured {sorted(first['cells'])}"
            )
        for name, entry in result["cells"].items():
            if entry["samples"] != first["cells"][name]["samples"]:
                raise InputError(
                    f"cell {name}: `{arm_label(result)}` measured {entry['samples']} "
                    f"samples and `{arm_label(first)}` "
                    f"{first['cells'][name]['samples']}"
                )
    return results


def run(args) -> str:
    results = load(args.result)
    first = results[0]

    snapshot = Path(args.model).resolve()
    if not snapshot.is_dir():
        raise InputError(f"the snapshot {snapshot} is not a directory")
    namespace, name = split_model_dir(snapshot.name)
    if (namespace, name) != (first["model_namespace"], first["model"]):
        raise InputError(
            f"the snapshot at {snapshot.name} is {namespace}/{name} where the run "
            f"was measured on {first['model_namespace']}/{first['model']}; a "
            "ceiling from the wrong checkpoint renders exactly as plausibly as a "
            "right one"
        )

    bandwidth = args.bandwidth_gbs
    if bandwidth is None:
        tag = first.get("hardware_tag")
        if tag != HOST_BW_HARDWARE_TAG:
            raise InputError(
                f"this run was measured on {tag!r} and perf_ceiling.py's bandwidth "
                f"constant was measured on {HOST_BW_HARDWARE_TAG!r}; a percentage "
                "of another machine's ceiling is a percentage of nothing. Pass "
                "--bandwidth-gbs for this host."
            )
        bandwidth = HOST_BW_BYTES_PER_S / 1e9

    windows = {}
    for result in results:
        label = arm_label(result)
        for name, entry in result["cells"].items():
            windows[(label, name)] = window(entry["prompt_tokens_mean"],
                                            entry["completion_tokens_mean"])
        block = result.get("fixed_prompt")
        if block:
            windows[(label, "#fixed")] = window(block["prompt_tokens"],
                                                block["completion_tokens"]["mean"])

    wanted = sorted({c for w in windows.values() for c in w})
    rows = ceilings_at(snapshot, first["kv_quant"], first["ctx_max"], wanted,
                       bandwidth, args.runs_db)
    bounds = {
        key: bound_for(rows, f"{key[0]} / {key[1]}", w, args.ceiling_spread_pct)
        for key, w in windows.items()
    }

    meta = {"bandwidth_gbs": bandwidth, "hardware_tag": first.get("hardware_tag", "—")}
    return render(results, snapshot, bounds, meta, args.command)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("result", nargs="+",
                    help="one spec_bench_published.sh result per engine mode")
    ap.add_argument("--model", required=True,
                    help="the verifier snapshot the runs measured")
    ap.add_argument("--out", default=None, help="write here instead of stdout")
    ap.add_argument("--bandwidth-gbs", type=float, default=None,
                    help="this host's memory bandwidth; required off the host "
                         "perf_ceiling.py's constant was measured on")
    ap.add_argument("--ceiling-spread-pct", type=float, default=5.0,
                    help="how far a ceiling may move across a decode window and "
                         "still be printed as one number")
    ap.add_argument("--runs-db", default=None,
                    help="consult this runs.db for the prefill anchor; without it "
                         "the input-speed bound is reported as absent")
    ap.add_argument("--command", default="make published-table",
                    help="the regeneration command printed in the header")
    args = ap.parse_args()
    try:
        text = run(args)
    except InputError as exc:
        print(f"published_table: {exc}", file=sys.stderr)
        return 1
    except CeilingError as exc:
        print(f"published_table: {exc}", file=sys.stderr)
        return 3
    except (OSError, KeyError, ValueError) as exc:
        print(f"published_table: {exc}", file=sys.stderr)
        return 1
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
