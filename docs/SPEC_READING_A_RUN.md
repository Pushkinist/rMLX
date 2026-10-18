# Reading a speculative run

How to read what a speculative run reports, and what a round costs. Split out of
[`SPECULATIVE.md`](SPECULATIVE.md) to keep that doc under the size cap.

Accept rate and decode rate depend on the pair and on the prompt.
`scripts/spec_bench.sh` measures them and files rows in `runs.db`. This doc
records only the figures that explain a result, in "What a round costs".

## The request record

Every entry closes a request with one `done` line at `info`, including a request
whose seed token was a stop token. The event is `<entry>: done`, for example
`mtp_generate: done` or `spec_generate_greedy_cached: done`. A request that is
gone and stops in its prefill or at the head of a round writes no `done` line
([`SERVER_SHUTDOWN.md`](SERVER_SHUTDOWN.md) § "A chat job whose request is
gone").
`crates/rmlx-models/src/speculative/round_stats.rs` holds the counters, the
derivation and the log site.

| Field | Meaning |
|---|---|
| `loop_kind` | which entry ran the request |
| `rounds`, `total_draft`, `total_accept` | rounds run, tokens proposed, proposals accepted |
| `emitted` | tokens the request committed, seed included |
| `seed_emitted` | tokens committed before the first round |
| `emitted_in_rounds` | tokens the rounds committed |
| `accept_rate` | `total_accept / total_draft` |
| `accepted_per_step` | `total_accept / rounds` |
| `tokens_per_round` | `emitted_in_rounds / rounds` |
| `decode_tps` | first emitted token to last, prefill excluded; `Some(x)`, or `None` under two tokens |
| `elapsed_ms` | wall clock of the whole request, prefill included; the prefill has its own `generate: prefill` event (`PROFILING.md` §11) |
| `round_ms`, `draft_ms`, `verifier_ms` | the round loop, and the drafter and verify spans inside it |
| `draft_ms_per_round`, `verify_ms_per_round`, `loop_ms_per_round` | per round; the last is `round_ms` less the other two |
| `conditioned_rows` | rows a projecting drafter's rounds have projected |
| `block_size` | the block the entry resolved |
| `charged` | whether phases were forced (see "Charged and uncharged phases") |
| `decode_config` | the cell the request's rows are filed under, for example `dflash2/block=8` |

`decode_tps` renders through `Debug`, so the field reads `Some(20.98)` or
`None`. `emitted / elapsed_ms` is a different, lower figure, because
`elapsed_ms` includes the prefill.

**`tokens_per_round` is the figure a speculative result is read with.** It
counts what the rounds produced, not the seed. It equals
`1 + accept_rate × (block − 1)` only while every round drafts the full block;
DFlash 1 resizes its block, so it is recorded, not derived.

The engine logs an `error!` when the counts do not add up:

- `seed_emitted + emitted_in_rounds` differs from `emitted`;
- the rounds committed more than `total_accept + rounds`;
- `draft_ms + verifier_ms` exceeds `round_ms`;
- `conditioned_rows` is further than one block from `emitted_in_rounds`.

## The round event

Every round that does not stop on a stop token logs one `speculative round`
event, target `rmlx::spec::phase`, at `debug`, through `round_stats::log_round`:

```
loop_kind round accept num_draft n_committed emitted_total
condition_rows projected_rows
v_offset_before v_target d_offset_before d_target refolded charged
round_ms draft_ms verify_ms walk_ms rollback_ms other_ms
```

- A field a drafter does not have is absent, not zero. `condition_rows` and
  `projected_rows` belong to drafters that carry conditioning;
  `d_offset_before` and `d_target` to drafters that keep a cache.
- `n_committed` is what the round committed, after the request's budget.
- `emitted_total` is every committed token so far, the seed included.
  `scripts/lib/spec_round_log.py` prints an `emitted_total` that sums
  `emitted_in_rounds` and excludes the seed. On a sidecar the two differ by
  `seed_emitted`.
- `v_offset_before` is the verifier offset before or after the verify forward,
  as the drafter declares in `VERIFIER_OFFSET_BASIS`. Both name the position
  the rollback is computed from.
- `refolded` is true only when the verifier's rollback refolded a recurrent
  state. `rollback_round` returns it.
- `rollback_ms` covers the verifier rollback, the drafter's own rollback and the
  next round's conditioning.
- `other_ms` is the time no phase claimed: emission, detokenization and
  bookkeeping. When the phases claim more than the round, the event carries no
  `other_ms` and an `error!` names the phases.

## Charged and uncharged phases

The engine evaluates lazily. A span is charged for the work it issued only when
something inside it blocks, so uncharged phase times drift. A round's rollback
is typically paid inside the next round's `draft_ms`.

At `trace` on `rmlx::spec::phase` (`--log verbose`, or
`RUST_LOG=info,rmlx::spec::phase=trace`), each phase forces its own work before
its span closes. That run is slower than the run it describes, because the
forced evaluations drain the pipeline.

- `phases_charged()` is read once, in the entry, and travels in `RoundCfg`. It
  reaches `rollback_round` as an argument and the record as `charged`.
- Three entries can charge: the MTP sidecar, the assistant and DFlash 2. The
  other four pass `false`, so their phase times are always uncharged.
- On a charged round, `log_round` checks the arrays the round hands to the next
  one. Any that are still unevaluated are named in an `error!` ("these arrays
  are still unevaluated"). After a change to a drafter's forcing, a charged run
  with no such line is the check.

`scripts/spec_bench.sh` refuses to file a charged row, because `observations`
is append-only. `RUST_LOG` takes precedence over `--log`, so unset it before a
bench.

## Benching an arm

`scripts/spec_bench.sh` measures a speculative arm and a no-drafter arm on one
prompt:

- The speculative arm's decode rate is the `decode_tps` of the `done` line,
  read by `scripts/lib/spec_round_log.py`. That reader re-derives every derived
  field from the event's own counters and refuses a mismatch.
- The no-drafter arm's rate is `1000 / step_mean_ms` from `GET /metrics/cache`,
  read by `scripts/lib/server_decode_tps.py`.
- Both use the window from the first emitted token to the last.
  `scripts/lib/sse_decode_window.py` times the same window client-side. A
  disagreement past `CROSS_CHECK_BAND_PCT` (10) stops the run.
- Each measured completion is digested. Without a sampler, equal digests file
  `answer_check=matches_plain`. A speculative answer that differs is judged by
  `rmlx eval spec-answer` over one unstreamed logprob pass per arm, each held
  to its arm's digest. A first divergence at a near tie files
  `answer_check=near_tie@<token>` with `divergence_rank` and
  `divergence_margin`. Any other difference stops the run before any median.
  Whether a sampler ran is read from the engine's own event
  (`scripts/lib/server_sampling.py`). Under a sampler both rows record
  `answer_check=sampled`.

The bench's answer check decides whether to file a row. It does not judge
correctness; that is `SPECULATIVE.md` § "Judging a draft-side change". Its ceiling (0.12) was set on 256-token
prose, so at 128 tokens a broken loop can pass on one prompt. It has no
repetition control. When its digest differs, EAGLE-3 is refused unless its
logprob pass binds, and it is judged with no restricted-vocabulary excuse.

## What a round costs

A round feeds the verifier `f` rows: the carry token and the proposals, so `f`
is the block. The forward reads the weights once, as a plain step does, and
still costs more than a plain step, and every further row adds to it. In plain
steps (one plain step is 1):

| Verifier | Plain step | Verify at `f` = 2 | Per extra row |
|---|---|---|---|
| `Qwen3.8-27B-4bit` | 30.5 ms | 1.04 | 0.095 |
| `gemma-4-e4b-it-mxfp8` | 12.0 ms | 1.04 | 0.113 |

The Qwen3.8 slope is a fit over `f` = 2 to 4. At `f` = 5 the measured verify is
within 4% of the line. The gemma slope is a fit over `f` = 2, 4 and 6: its
segments rise from 0.066 to 0.174 per row.

**The quantised matmul explains the slope.** A microbenchmark of
`quantized_matmul` over each snapshot's layer mix (every distinct trunk and
head shape, at the snapshot's own quantisation) costs 110% of the engine's
slope on both: 3.19 against 2.90 ms per row on Qwen3.8 over 2 to 4 rows, 1.50
against 1.35 ms on gemma over 2, 4 and 6. A bf16 matmul at the same shapes
costs the same at every row count. The weight read is not the cost, and no
other per-row term has room in the slope.

**The slope steepens at 6 rows.** The quantised matmul has a step between 5 and
6 rows on the wide shapes: `N` at or above 6144 on Qwen3.8, and `N` = 10240
and the head on gemma. On Qwen3.8 the summed matmul time rises by 12.8 ms
there (the engine's verify reads 41.8 ms at 5 rows and 54.8 ms at 6). On
gemma it rises by 2.7 ms, 2.3 times the rise per row before it. A Qwen3.8
verify costs about 1.8 plain steps at 6 rows and 2.1 at 8. The 0.095 line
holds to `f` = 5 only.

**A speed-up is `tokens_per_round` over the round's cost in plain steps.** The
round cost covers the verify, the draft and the rollback. A bench row gives it
as `tokens_per_round ÷ speed-up`. `Qwen3.8-27B` with its MTP sidecar at block 2,
on the two prompts of `scripts/spec_bench.sh` (128 tokens, greedy):

| Weights | Prompt | Accept rate | Tokens per round | Speed-up | Round cost |
|---|---|---|---|---|---|
| 4-bit | code | 0.7297 | 1.716 | 1.466 | 1.171 |
| 4-bit | prose | 0.7067 | 1.693 | 1.502 | 1.127 |
| mxfp8 | code | 0.7397 | 1.740 | 1.612 | 1.079 |
| mxfp8 | prose | 0.6410 | 1.628 | 1.504 | 1.082 |

The 4-bit round costs 1.13 to 1.17 plain steps and the mxfp8 round 1.08, on both
prompts. On code the two pairs accept within 1.4% and the speed-ups differ by
9%. The accept rate alone does not predict the speed-up across weight formats;
the round cost does. Above one plain step the round costs about the same in
milliseconds on both pairs: 3.9 ms (prose) to 5.2 ms (code) on the 4-bit pair
(30.5 ms plain step), and 4.2 ms (code) to 4.3 ms (prose) on the mxfp8 pair
(52.7 ms plain step). That is a larger share of the faster 4-bit plain step.
