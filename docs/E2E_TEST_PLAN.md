# rMLX E2E Feature-Proof Test Harness

Status: Modalities + agent surface — Gemma4-e4b image input + Qwen3.6
/ Bonsai tool-calling, all LIVE + green, 3/3. Speculative decoding (DFlash
+ MTP) verified live.

## Purpose

PROVE — end-to-end, against the **real** `rmlx` binary — that every shipped
feature × sub-feature actually works. This is a *correctness* gate, not a perf
gate (perf has its own `scripts/perf_canary.sh` + `docs/PERF_BASELINE.md`).

The harness emits a PASS/FAIL grid artifact (feature × sub-feature) to
`<RMLX_HOME>/e2e/report.{json,md}` so a single glance answers "does feature X
work on model Y at KV preset Z?".

## Form (locked design)

- **Manifest (TOML) + Rust runner driving the real binary.** One declarative
  manifest (`crates/rmlx-cli/tests/e2e/manifest.toml`) lists every case. A
  `#[ignore]`d Rust integration test (`crates/rmlx-cli/tests/e2e_harness.rs`)
  parses it, spawns the **actual** binary — either a CLI subcommand
  (`CARGO_BIN_EXE_rmlx`) or `rmlx serve` + raw HTTP — and asserts on real
  output / metrics / logs. This is true E2E, NOT internal-fn unit tests.
- Reuses existing patterns:
  - golden-token decode — `crates/rmlx-models/tests/common/mod.rs`
  - spawn + HTTP — `crates/rmlx-server/tests/http_smoke.rs`
  - NIAH retrieval — `crates/rmlx-models/tests/niah_long_context.rs`
  - subprocess the binary — `crates/rmlx-cli/tests/cache_type_validate.rs`
- Uses the existing workspace `toml` dep (no new dependency).

## Assertion kinds (hybrid)

| kind | semantics | used for |
|---|---|---|
| `golden` | **REAL** byte-for-byte golden: the per-token chosen-token-id (byte) sequence at temp=0 greedy is captured via the OpenAI logprobs `bytes` field and compared to a recorded golden file under `tests/e2e/golden/<case_id>.json`. Golden ABSENT → `Skip` ("no golden recorded"); `RMLX_E2E_REGEN_GOLDEN=1` or first run → WRITE the golden. Requires the OpenAI non-stream logprobs path. | text core, deterministic OpenAI non-stream paths |
| `contains_coherent` | substring + coherence: deterministic temp=0 output is coherent AND contains the expected substring. The honest kind for fixtures that cannot expose per-token byte identity over their wire shape (SSE stream, anthropic `/v1/messages`). | streaming + anthropic deterministic paths |
| `coherent` | smoke-probe: decoded output is non-empty, non-degenerate (≥2 real words, mean word-len >1, no single-token repeat loop above 60%, no NaN), echoes an expected substring when given | quant V codecs that legitimately shift tokens; short-ctx K-only |
| `niah_retrieval` | needle recovered from an 8k haystack at temp=0 | quant codecs that must preserve long-ctx retrieval |
| `cosine_vs_bf16` | per-position top-k logprob **distribution** cosine vs the `none`/bf16 reference, averaged over positions where the chosen token-ids agree. A chosen-token-id divergence within the first 8 tokens is the real failure signal (reported as such, NOT smuggled into a scalar over mismatched tokens). bf16 ref cached per `(model, fixture)`. | quant codecs — numeric fidelity |
| `exit_code` | process exit code == expected | CLI refusals, surface checks |
| `serve_refused` | **REAL** arch-guard refusal. Spawn `rmlx serve` with an ILLEGAL KV codec for the arch (via `kv_quant` preset or `ctk`/`ctv` compose form) and assert the process EXITS at resolve time with the documented non-zero code (default `expect=78`, `EX_CONFIG`), BEFORE `/health` binds. `resolve_model_flags` loads `config.json` and `std::process::exit(78)` on a `ResolveError` (`QwenMoeKBitsTooLow` for K<8; `QwenMoeTurboKRejected` for a symmetric-K turbo codec). PASS = process exits with the expected code within 30 s; a server that stays alive (guard silently failed to fire) or a wrong code FAILS. Proves the per-arch KV invariant is a working feature, not a doc claim. | per-arch KV invariant refusal |
| `metric_present` | exit 0 + a REQUIRED non-empty `expect` needle present in stdout (an empty needle is a manifest error → `Fail`) | baseline --record, metrics query |
| `dispatch_fired` | **REAL** attention-dispatch log scrape. Serve `--log verbose` (no `RUST_LOG` override) so the per-dispatch `update_and_sdpa` trace span reaches the run jsonl; drive a short generation; scrape `<RMLX_HOME>/logs/*.jsonl` for `update_and_sdpa` spans, tallied by the `path` span field. PASS = ≥1 dispatch span with a resolved `path` AND coherent output. **Honest scope (empirical on Bonsai/Qwen3, head_dim=128):** dispatches on the warm-TTFT flow resolve to `path="legacy"` or `path="flash"` (TurboFlash eligible); the specialised `fused_qk` / `planar_k_fused` kernels stay DORMANT on normal generate. Their dispatch *counters* are process-internal atomics with no HTTP/metrics surface and are covered by the in-crate counter tests, NOT E2E (see coverage note below). | attention dispatch path (instrumentation fires + coherent) |
| `model_lifecycle` | **REAL** multi-model lifecycle. Registry-mode serve (`--registry` JSON with Bonsai + a 2nd model from `GEMMA4_E2B`, `--max-loaded-models`), under single-MLX discipline. Proves: (b) cap=2 → both resident; (a/c) cap=1 eager preload of [A,B] → B resident, A LRU-evicted (`/v1/models/{id}/status` flips); (d) explicit unload B → `loaded:false`, 2nd unload → 404; (e) **claim enforcement** — a 2nd `rmlx serve` on the HELD port exits 11 (`ClaimError::AlreadyHeld`) without binding a competing Metal context. 2nd model absent → 2-model legs SKIP inline, single-model subset (a)+(e) runs. A wrong status transition, a rival that wrongly starts, or an unparseable status body all FAIL. | multi-model lifecycle + claim |
| `byte_identical_restart` | **REAL** SSD cross-restart. Owns two serve phases under single-MLX discipline + a dedicated hermetic `RMLX_HOME`: serve `--kv-ssd-cache-gb 1 --prompt-cache-slots 1`, generate the long prompt (capture per-token chosen bytes), issue a prefix-disjoint prompt to evict→spill, poll the disk for a `.kvb`; kill + claim-settle; restart same HOME; regenerate the long prompt. PASS = phase-2 completion **byte-identical** to phase-1 (rehydrated blocks reproduce decode) AND `/metrics/cache` `ssd_hits ≥ 1` (RAM miss served from `.kvb`, not a silent cold re-prefill). Either leg wrong → FAIL. | SSD KV tier |
| `cache_hit_equivalence` | **REAL** prompt-cache exact-prefix reuse. Single server, cache ON. Issue a ≳600-token (multi-block) prompt twice; Bonsai/Qwen3 is `ExactOnly` so the second identical request is an exact prefix hit. PASS = request-B `content` **byte-identical** to request A (reuse must not change output) AND `/metrics/cache` `block_hits` incremented between A and B (B genuinely hit). Equivalence axis is `content`, not the logprobs `bytes` stream — see finding 4. | prompt cache / APC |
| `image` | **REAL** image input. Serve a vision-capable model; send the `image_color` fixture — a bundled 224×224 solid-red PNG (committed at `tests/e2e/fixtures/vtest_red.png`, referenced by absolute file path so there is no base64 transport) + a one-word colour question. PASS = the answer names the colour (`expect`, default "red", case-insensitive), proving the vision tower → soft-token scatter → generate path end-to-end. FAILS on non-200, empty answer, or wrong/absent colour (text-only fallthrough). Validated on Gemma4-e4b (SigLIP ViT → "Red"). | image input (vision tower) |
| `tool_call` | **REAL** function/tool-calling. Serve a tools-capable model; send the `tool_weather` fixture (a `get_weather` tool + a prompt that should trigger it). PASS = `finish_reason == "tool_calls"` AND a `tool_calls[]` entry whose function name matches `expect` (default `get_weather`), proving request → chat-template → parse → emit end-to-end (not just that `tools` is accepted). Validated on Qwen3.6-MoE (XML `<function=…>` format) and Bonsai (Hermes-JSON format). | tool/function calling |
| `spec_decode` | **REAL** speculative decoding. Serve the verifier (`model`) with a real drafter — `--draft-model` resolved from `case.draft_model` (path/slug/alias, same resolver as `model`), `--draft-kind` from `case.draft_kind` (`mtp`/`dflash`/`eagle3`) — under `--log verbose`; drive one 300-token generation; scrape `<RMLX_HOME>/logs/*.jsonl` for the round-loop summary `<kind>_generate_greedy: done` and read its `fields.accept_rate` / `fields.rounds`. PASS = the round-loop FIRED with `accept_rate > 0` AND the output is coherent (`expect` token, default "Paris", in `content` OR — for a thinking model whose budget stays in the reasoning block — `reasoning_content`). Proves the drafter actually proposes ACCEPTED tokens end-to-end, not merely that it loads. SKIPs (clear reason) when the drafter snapshot is absent. | speculative decoding (drafter fires + coherent) |
| `stop_halts` | same prompt with/without `stop`; stopped completion strictly shorter | stop-sequence feature |

> **`xfail` downgrade is narrow.** A FAIL on an `xfail`-tagged case is
> downgraded to `XFAIL` ONLY when the failure detail matches the case's
> documented failure mode (for `stop_halts`: detail starts with "stop did not
> shorten output"). Infra failures — HTTP error, non-200 status, spawn failure,
> parse error — stay `Fail` and still trip `any_failed()`, so a broken server is
> never masked as a "known gap".

### Verdicts

`PASS` / `FAIL` / `SKIP` (model unresolved) / `PENDING` (Phase 2 stub) /
**`XFAIL`** — a real, documented product gap. A case tagged `xfail` whose
assertion genuinely fails is recorded as `XFAIL` (a finding) and does NOT fail
the suite. A `PASS` on an `xfail` case means the gap was fixed → drop the tag.

### Known findings (run-1, Bonsai)

1. **stop sequence not truncating** (`xfail`): `stop:["charlie"]` returns
   `finish_reason:"stop"` but the returned `content` still contains the full
   sequence past the stop string. Stop is detected but output is not truncated.
   Confirmed via manual probe (also reproduces with `stop:["delta"]`).
2. **RotK Display form not accepted by `--kv-quant`**: `rot_k_v4g64` (the
   `KvQuant::RotK` `Display`) is rejected by `--kv-quant` FromStr (the valid
   list omits it). RotK is reachable only via the compose form
   `--ctk rot_k --ctv q4_g64`, which works correctly. Display ↔ FromStr
   asymmetry.
3. **rot_k_tq4v degraded on Bonsai-2bit**: drops the answer token at temp=0
   ("The capital of France." — no "Paris") and is incoherent at 8k NIAH.
   Reclassified to short-coherence-only (`degraded` tag), like the K-only
   family.
4. **Prompt-cache hit omitted first token from logprobs — RESOLVED**: previously,
   on the exact cache-hit path the cached `first_id` token was replayed without
   live logits (`qwen3.rs` Path A), so it was OMITTED from the OpenAI
   `logprobs.content` array — a hit response carried N-1 logprob entries vs the
   miss's N, while the detokenized `content` stayed byte-identical. The fix
   stores the prefill-token logprobs alongside `first_id` (captured at the OpenAI
   `top_logprobs` ceiling, 20) and replays them on hit, truncated to the
   request's `top_logprobs_k`. The hit now emits exactly one `logprobs.content`
   entry per token (N == N) and the first token's `token_logprob` is byte-equal
   to the miss path's true value. The `cache_hit_equivalence` proof continues to
   assert `content` equality; with this fix the logprob streams are length-equal
   too (no more skew to note).

Rationale: temp=0 deterministic paths (text core, tool calls) get exact
golden-token match. Quant V codecs that legitimately shift the argmax get the
softer coherence + NIAH + cosine triple. State features get equivalence.

## Run-1 scope

**Bonsai only** (`prism-ml__Ternary-Bonsai-8B-mlx-2bit`, `Qwen3ForCausalLM`,
head_dim=128, dense — NOT MoE, so the Qwen-MoE K<8 PPL guards do NOT apply, and
nearly every KV preset is legal), every legal KV preset. Correctness-first;
perf incidental.

Bonsai facts that drive legality:
- `head_dim = 128` → divisible by 4 (Iso quaternion) and 32 (Planar) — all
  rotation codecs are shape-legal.
- dense `Qwen3ForCausalLM` (not `Qwen3_5MoeForConditionalGeneration`) →
  `k_below_8bit()` codecs (`*Sym`, `IsoKOnly*`, `RotorKOnly*`,
  `RotorK*Asym`, `PlanarK`) are all LEGAL here.
- `max_position_embeddings = 65536` via YARN → 8k NIAH is well within band.

### K-only codec caveat

The K-only family (`IsoKOnly3/4`, `RotorKOnly3/4`) and the `*Sym` variants
re-quantise K every decode step and are **incoherent at long context on
Bonsai-2bit**. For these the harness asserts **short-ctx coherence only**
(NOT NIAH, NOT cosine), and the manifest row is tagged `konly` + carries a
note. This is a documented limitation, not a harness gap.

## Feature → sub-feature matrix (Phase 1)

| Feature | Sub-features | Assert |
|---|---|---|
| CLI surface | `info --probe-smoke`, `info --list-cache-types`, `baseline --record`, `metrics query`, `healthcheck --full` | exit 0 + stdout/db shape |
| Text core | chat non-stream, chat stream, anthropic `/v1/messages`, multi-turn, `stop`, `seed` determinism | golden / coherent at temp=0 |
| Quant matrix | every legal Bonsai KV preset (see below) | coherent + NIAH@8k + cosine≥0.99 (full codecs); short coherence only (K-only/Sym) |
| Thinking | `enable_thinking=true` + `thinking_budget` | `reasoning_content` populated + budget enforced + answer correct |

### Quant presets exercised (Bonsai, dense Qwen3)

Full codecs (coherent + NIAH@8k + cosine≥0.99):
`k8v8`, `k8v4`, `k8vturbo3`, `k8vturbo2`, `k8vturbo3tcq`, `k8vturbo2tcq`,
`planar`, `planar3`, `iso3`, `iso4`, `rotor3`, `rotor4`,
`mixed_k8g64_v4g64`, `rot_k_v4g64` (RotK), `rot_k_tq4v`, `planar_k`.

K-only / symmetric (short-ctx coherence only, see K-only codec caveat above):
`k_iso3`, `k_iso4`, `k_rotor3`, `k_rotor4`, `iso3_sym`, `iso4_sym`,
`rotor3_sym`, `rotor4_sym`, `tsym3`, `tsym4`.

Preset → flag mapping is derived from `KvQuant` `Display`/`FromStr`
(`crates/rmlx-kv-quant/src/quant.rs`) + `docs/CLI.md`. The harness passes the
preset string straight to `--kv-quant` (server) which round-trips through
`KvQuant::from_str`.

## How to run

```bash
# point the harness at the Bonsai snapshot (absolute path)
export RMLX_E2E_MODEL_BONSAI=/abs/path/to/prism-ml__Ternary-Bonsai-8B-mlx-2bit
# or rely on the existing RMLX_TEST_MODEL_BONSAI / RMLX_O_MODELS_ROOT fallback

make e2e                 # builds, runs --ignored --test-threads=1, prints grid path
# or directly:
cargo test -p rmlx-cli --test e2e_harness -- --ignored --test-threads=1 --nocapture
```

### Model resolution (data-driven — adding a model needs no code edit)

A manifest row's `model` field is a **spec**, resolved by `resolve_model()`
(`runner.rs`). A spec is one of:

1. **Path** — contains `/` and exists → used verbatim (absolute or cwd-relative).
2. **Slug** — a snapshot directory name (e.g.
   `mlx-community__gemma-4-31b-it-mxfp8`) → joined under `RMLX_O_MODELS_ROOT`
   (default `./models`).
3. **Alias** — a short canonical shorthand (`BONSAI`, `GEMMA4_E4B`,
   `GEMMA4_E2B`, `QWEN36`) → mapped to its slug. The alias table is
   **frozen** to the canonical targets; new models come in as a slug or
   path, never a new match arm.

Resolution order for any spec (first hit wins):
1. `RMLX_E2E_MODEL_<SPEC>` (spec upper-cased, non-alphanumerics → `_`) — runtime
   redirect, beats everything. Works for a model with **no manifest row** too.
2. `RMLX_TEST_MODEL_<SPEC>`.
3. Spec-as-path (when it contains `/` and exists).
4. `RMLX_O_MODELS_ROOT/<alias-slug-or-literal-spec>`.

So a new model is added by either: a manifest `[[case]]` row carrying its slug
(`model = "mlx-community__…"`), or — for a one-off run — purely
`RMLX_E2E_MODEL_<SLUG>=/abs/path` at the CLI with no file change. The
resolver code is never touched to add coverage.

When nothing resolves, every model-gated case **skips** (records `SKIP` in the
grid) and the suite is green — the harness is safe on machines without
snapshots.

### Single-MLX claim discipline

The harness holds the single-MLX invariant (hard rule 8). Before each
model-touching case it runs the preflight (`pkill -f "rmlx serve"`; remove the
claim file) and tears down spawned `rmlx serve` children after. `--test-threads=1`
is mandatory — only one MLX process per Mac.

## Phasing

- **Phase 1 (run-1):** Bonsai, the four features above. Executed.
- **Phase 2a (run-2a, LIVE):** Bonsai state features — SSD KV cross-restart
  (`byte_identical_restart`) and prompt-cache exact-prefix reuse
  (`cache_hit_equivalence`). Both rows are tagged `run2a` (no longer `phase2`),
  execute against the real binary, and PASS on Bonsai (see Run-2a results
  below). A new `RMLX_E2E_ONLY=id1,id2` env selector restricts a run to named
  case ids for targeted reruns (empty/unset → full manifest).
- **Phase 2b (run-2b, LIVE):** multi-model lifecycle (`model_lifecycle`) and
  attention dispatch (`dispatch_fired`). Both rows are tagged `run2b` (no longer
  `phase2`), execute against the real binary, and PASS on Bonsai + Gemma4-e2b
  (see Run-2b results below).
- **Phase 2c (run-2c, LIVE):** broader-model text-core proof for the other two
  test-target families — **Gemma4-e4b** (`mlx-community__gemma-4-e4b-it-mxfp8`,
  mxfp8, SWA, head_dim=256, NOT a thinking model) and **Qwen3.6-35B-A3B**
  (`mlx-community__Qwen3.6-35B-A3B-8bit`, 8-bit affine, sparse MoE, 7:1 GQA,
  head_dim=128, K≥8 enforced). Rows are tagged `run2c` (no longer `phase2`) and
  execute against the real binary. CORRECTNESS-first with a **representative**
  legal-KV subset per arch (NOT the full 18-preset matrix — extensible by
  adding rows): golden text-core + stream/anthropic coherence, niah@8k +
  cosine≥0.99 per legal cell, a SWA window-crossing case (Gemma4), the Qwen-MoE
  arch-guard refusals (exit 78), and a Qwen3.6 thinking case. See Run-2c results
  below. Resolver keys `GEMMA4_E4B` / `QWEN36` were already present; absent
  models SKIP with a clear reason.
- **Phase 2d (run-2d, LIVE):** big-Gemma4 text-core proof — **gemma-4-26b-a4b**
  (`mlx-community__gemma-4-26b-a4b-it-mxfp8`, MoE, 128 experts, 30 layers) and
  **gemma-4-31b** (`mlx-community__gemma-4-31b-it-mxfp8`, dense, 60 layers,
  kv_heads=16). Both `Gemma4ForConditionalGeneration`, head_dim=256, SWA
  window=1024 → same KV legality as Gemma4-e4b (SWA layers stay bf16,
  full-attn layers quantized, K8V4 needs TurboFlash). Rows tagged `run2d`.
  Representative per-arch cells: golden text-core + stream coherence, k8v8 /
  k8v4 (`--turbo-flash on`) / planar cosine≥0.99, k8v8 + auto-KV SWA
  window-crossing niah@8k. These rows carry the snapshot **slug** directly in
  `model` (e.g. `model = "mlx-community__gemma-4-31b-it-mxfp8"`) — the resolver
  is data-driven, so adding a model needs NO code edit (see §Model resolution).
  **Result: 14/14 PASS, zero defects** — both big models load, serve coherent
  (golden byte-identical replay), recover the niah needle across the SWA
  boundary, and hold top-k cosine 1.0000 on every legal KV cell. Confirms the
  Gemma4 path (SWA snapshot/restore, TurboFlash head_dim=256, MoE vs dense)
  is size-invariant from e4b up to 31B.
- **Phase 2e (run-2e, LIVE):** speculative decoding for Qwen3.6-MoE with REAL
  drafters — **DFlash** (`z-lab__Qwen3.6-35B-A3B-DFlash`) and the **MTP sidecar**
  (`mlx-community__Qwen3.6-35B-A3B-MTP-5bit`), both against the
  `Qwen3.6-35B-A3B-8bit` verifier. The `spec_decode` assert serves with
  `--draft-model`/`--draft-kind` under verbose logging, drives a 300-token
  generation, and scrapes the `<kind>_generate_greedy: done` round-loop summary
  for `accept_rate`. **Result: 2/2 PASS** — DFlash accept_rate≈0.75 (52 rounds),
  MTP accept_rate≈0.89 (108 rounds), both coherent ("Paris" in the thinking
  block). Unlike the earlier recommendation, accept-rate IS externally
  observable via the round-loop's `tracing::info!` summary in the run jsonl — no
  HTTP surface needed, same log-scrape mechanism as `dispatch_fired`. This
  supersedes the old `p2_speculative_decode` Bonsai/MTP `phase2` stub (Bonsai is
  dense Qwen3 with no MTP drafter; the real coverage is the Qwen3.6 rows).
- **Phase 2f (run-2f, LIVE):** modalities + agent surface — the two heavy-use
  paths. **Image input** (`p2f_gemma4_image`): Gemma4-e4b's SigLIP vision tower
  reads a bundled solid-red PNG and answers the dominant colour ("Red"). **Tool
  calling** (`p2f_qwen36_tool_call`, `p2f_bonsai_tool_call`): Qwen3.6-MoE (XML
  `<function=…>`) and Bonsai (Hermes-JSON) each emit a real `get_weather`
  `tool_call` with `finish_reason=tool_calls` and correct args. **3/3 PASS.**
  Supersedes the `p2_image_input` `phase2` stub. The tool-calling subsystem was
  fully wired but had no generation test (only request-schema acceptance); the
  vision real-weights test was `#[ignore]` — both are now proven live.
- **Phase 2 remaining (stubbed in manifest, `tags=["phase2"]`, NOT executed):**
  audio input (`p2_audio_input`) — pending an audio request fixture + the
  whisper path (a `whisper-large-v3-mlx` snapshot is on disk for it).

### Attention-kernel dispatch — coverage note

The *specialised* attention-kernel dispatch (TurboFlash `flash`, generalised
`fused_qk`, `planar_k_fused`, two-phase `sparse_attn`) is **counted via
process-internal atomics** (`fused_qk_total_dispatch_count()`,
`sparse_attn_total_dispatch_count()`, the per-codec `*_dispatch_count()`
getters) with **no HTTP/metrics surface** — there is no clean externally-
observable signal when driving the real binary. Empirically (verbose-log probe,
Bonsai/Qwen3 head_dim=128), every codec routes through `path=legacy` on the
normal generate flow because the warm-TTFT bf16-K seed shortcuts the
specialised kernels. That dimension is therefore covered by the **in-crate
dispatch-counter tests**, not E2E:

- `crates/rmlx-models/tests/sparse_attn_dispatch.rs` (`RMLX_SPARSE_ATTN_STRICT=1`)
- `crates/rmlx-kv-quant/tests/rotor_fused_qk_dispatch.rs` (rotor decode routing contract)

The E2E `dispatch_fired` row asserts the one thing that **is** externally
observable: the attention-dispatch instrumentation fires (`update_and_sdpa`
span with a resolved `path`, count ≥ 1) and the output stays coherent.

### Run-2a results (Bonsai)

| Feature | Sub-feature | Verdict | Evidence |
|---|---|---|---|
| SSD KV tier | byte-identical restart (spill→restart→hydrate) | **PASS** | 24-token completion byte-identical across the process restart; `ssd_hits=1`, 1 `.kvb` spilled |
| prompt cache | exact-prefix reuse (ExactOnly) | **PASS** | request-B `content` byte-identical to A; `block_hits` 0→1; logprobs stream length-equal A↔B (finding 4 resolved — see Known findings) |

Run targeted: `RMLX_E2E_ONLY=ssd_byte_identical_restart,prompt_cache_prefix_reuse make e2e`.

### Run-2b results (Bonsai + Gemma4-e2b)

| Feature | Sub-feature | Verdict | Evidence |
|---|---|---|---|
| multi_model | load / LRU-evict / unload / claim-enforcement | **PASS** | cap=2 → Bonsai + Gemma4-e2b both resident; cap=1 eager preload → Gemma4-e2b resident, Bonsai LRU-evicted (status flipped); unload → `loaded:false`, 2nd unload → 404; rival `rmlx serve` on the held port rejected (exit 11, claim error) |
| attention | `update_and_sdpa` dispatch fired (verbose-log scrape, k8v4) | **PASS** | 360 `update_and_sdpa` dispatches scraped from the verbose jsonl (`path=legacy` or `path=flash`); output coherent (`"The capital of France is Paris."`). Specialised fused-QK / planar kernels dormant on the warm-TTFT flow — counter coverage is in-crate (see note above) |

Run targeted: `RMLX_E2E_ONLY=p2_multi_model_lifecycle,p2_attn_dispatch_fired RMLX_O_MODELS_ROOT=/path/to/open-models make e2e`.

2nd-model resolution: `model_lifecycle` resolves `GEMMA4_E2B`
(`mlx-community__gemma-4-e2b-it-mxfp8`) for model B. When absent it runs the
single-model subset (legs a + e) and marks the 2-model legs SKIPPED.

### Run-2c results (Gemma4-e4b + Qwen3.6-35B-A3B-MoE)

All 24 Phase-2c cells GREEN (0 FAIL). Cosine cells compare each codec's
per-token top-k logprob distribution against the model's own bf16 reference
(spawned under single-MLX discipline after the quant server is torn down).

**Gemma4-e4b** (mxfp8, SWA, head_dim=256, not a thinking model):

| Sub-feature | Verdict | Evidence |
|---|---|---|
| text-core golden (temp=0) | **PASS** | 8 tokens recorded then replayed byte-identical (`golden/p2c_gemma4_golden.json`) |
| chat stream coherent | **PASS** | `"The capital of France is Paris."` |
| anthropic `/v1/messages` coherent | **PASS** | `"The capital of France is Paris."` |
| k8v8 niah@8k / cosine | **PASS** / **PASS** | needle recovered; cosine 1.0000 over 8 tokens |
| k8v4 niah@8k / cosine (`--turbo-flash on`) | **PASS** / **PASS** | needle recovered; cosine 1.0000 (TurboFlash kernel forced ON for head_dim=256) |
| k8vturbo3 niah@8k / cosine | **PASS** / **PASS** | needle recovered; cosine 1.0000 |
| planar niah@8k / cosine | **PASS** / **PASS** | needle recovered; cosine 1.0000 |
| SWA window-crossing niah@8k (auto KV) | **PASS** | needle recovered across the sliding window (ring snapshot/restore correct) |

**Qwen3.6-35B-A3B** (8-bit affine, sparse MoE, 7:1 GQA, head_dim=128, K≥8):

| Sub-feature | Verdict | Evidence |
|---|---|---|
| text-core golden (temp=0) | **PASS** | 8 tokens recorded then replayed byte-identical (`golden/p2c_qwen36_golden.json`) |
| chat stream coherent | **PASS** | `"The capital of France is Paris."` |
| k8v8 niah@8k / cosine | **PASS** / **PASS** | needle recovered; cosine 1.0000 |
| k8v4 niah@8k / cosine | **PASS** / **PASS** | needle recovered; cosine 1.0000 |
| planar niah@8k / cosine | **PASS** / **PASS** | needle recovered; cosine 1.0000 |
| iso3 niah@8k / cosine | **PASS** / **PASS** | needle recovered; cosine 1.0000 (rotation KV, K stays 8-bit) |
| **arch-guard**: K<8 (`--cache-type-k q4_g128`) REJECTED | **PASS** | serve exits **78** at resolve time (`QwenMoeKBitsTooLow`) before `/health` binds |
| **arch-guard**: symmetric-K (`--kv-quant tsym3`) REJECTED | **PASS** | serve exits **78** at resolve time (`QwenMoeTurboKRejected`) |
| enable_thinking (reasoning + answer) | **PASS** | `reasoning_content` populated (204 chars); answer 68 found |

No real feature bugs surfaced — every codec produced coherent output, both
arch guards fired with the documented exit 78, and the thinking path populated
`reasoning_content` with the correct answer.

Run targeted, per family (single-MLX, heavy 35B → split into bounded batches):
```bash
export RMLX_O_MODELS_ROOT=/path/to/open-models
RMLX_E2E_ONLY=p2c_gemma4_golden,p2c_gemma4_k8v8_niah,… make e2e
RMLX_E2E_ONLY=p2c_qwen36_guard_kbits,p2c_qwen36_guard_symk make e2e   # fast (no weights)
```

#### Extensibility (full matrix)

Phase 2c/2d are REPRESENTATIVE subsets, not the full per-arch matrix. To
extend: add more `[[case]]` rows with `model = "GEMMA4_E4B"` / `"GEMMA4_26B"` /
`"GEMMA4_31B"` / `"QWEN36"` and the remaining legal `kv_quant` presets (Gemma4:
`k8vturbo2`, `planar3`, `iso3/4`, `rotor3/4`, …; Qwen3.6: `iso4`, `rotor3/4`,
`mixed_k8g64_v4g64`, …, all K≥8).
Each new golden auto-records on first run (`RMLX_E2E_REGEN_GOLDEN=1` to refresh).
The Qwen-MoE illegal-codec surface (every `k_below_8bit()` variant) can be
swept with additional `serve_refused` rows.

## Files

- `crates/rmlx-cli/tests/e2e/manifest.toml` — the single declarative case list.
- `crates/rmlx-cli/tests/e2e/runner.rs` — parse + spawn + assert.
- `crates/rmlx-cli/tests/e2e/report.rs` — grid writer (`report.{json,md}`).
- `crates/rmlx-cli/tests/e2e/golden/<case_id>.json` — recorded byte-for-byte
  golden token sequences for `golden`-kind cases (regen: `RMLX_E2E_REGEN_GOLDEN=1`).
- `crates/rmlx-cli/tests/e2e_harness.rs` — `#[ignore]` entry point.
- `docs/E2E_TEST_PLAN.md` — this document.
