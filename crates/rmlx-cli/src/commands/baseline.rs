//! `rmlx baseline` command implementation.

#![allow(
    clippy::case_sensitive_file_extension_comparisons,
    clippy::cognitive_complexity,
    clippy::needless_pass_by_value,
    clippy::too_many_lines,
    clippy::used_underscore_binding
)]
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rmlx_metrics::events::EventRecorder;
use rmlx_metrics::identity::RunIdentity;
use rmlx_mlx::Device;
use rmlx_models::arch;
use tracing::{info, info_span, warn};

/// Decide how to handle a tokenized prompt longer than the prompt cap.
///
/// `max_prompt_tokens` is the explicit `--max-prompt-tokens`; when it is
/// `None` the cap is `ceiling` — the context ceiling
/// `rmlx_models::context::resolve_context` produced for this run, so the
/// default cap is exactly what the engine will actually serve rather than a
/// constant unrelated to the model.
///
/// Returns the effective prompt length to use (`Ok`), or an error when
/// silently truncating would misrepresent the measurement.
///
/// - Prompt fits under the cap: no-op, returns the prompt length unchanged.
/// - `--device cpu`: always silently truncates (with a `warn!`) -- CPU
///   forward is genuinely O(N^2), so the cap is a real sanity guard.
/// - `--device gpu` with an *explicit* `--max-prompt-tokens` or
///   `--allow-truncate`: the caller opted in, so truncate with a `warn!`.
/// - `--device gpu` on the default cap with no opt-in: a truncated run would
///   silently record a shorter measurement that looks like a full-length one,
///   so this is a hard error naming the cap that bit and the flag that lifts
///   it.
pub(crate) fn resolve_prompt_truncation(
    prompt_len: usize,
    max_prompt_tokens: Option<usize>,
    ceiling: usize,
    device: Device,
    allow_truncate: bool,
) -> anyhow::Result<usize> {
    let cap = max_prompt_tokens.unwrap_or(ceiling);
    if prompt_len <= cap {
        return Ok(prompt_len);
    }

    if device == Device::Cpu || max_prompt_tokens.is_some() || allow_truncate {
        warn!(
            original = prompt_len,
            cap, "baseline: prompt truncated to cap"
        );
        return Ok(cap);
    }

    Err(anyhow::anyhow!(
        "prompt has {prompt_len} tokens, exceeding the resolved context ceiling of {cap} on \
         --device gpu. Silently truncating would record a shorter run that looks like a \
         full-length measurement. Raise --max-ctx to {prompt_len} (or higher, up to the \
         model's positional capacity) to measure the full prompt, or pass --allow-truncate \
         to opt into truncation."
    ))
}

/// Escape a field value for RFC 4180 CSV: wrap in double-quotes if the value
/// contains a comma, double-quote, or newline, escaping interior double-quotes
/// by doubling them. Returns the original string if no special characters.
pub(crate) fn baseline_csv_escape(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        let inner = s.replace('"', "\"\"");
        format!("\"{inner}\"")
    } else {
        s.to_owned()
    }
}

/// Read peak RSS in MiB for the current process.
///
/// On macOS, `ru_maxrss` is in bytes (unlike Linux where it is kilobytes).
/// We use `ps -o rss= -p <pid>` which returns KiB on all platforms, avoiding
/// the need for the `libc` crate (not in the workspace).
///
/// Returns 0.0 on any error (best-effort, non-fatal).
pub(crate) fn peak_rss_mb() -> f64 {
    let pid = std::process::id();
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let s = String::from_utf8_lossy(&out.stdout);
            let trimmed = s.trim();
            // ps -o rss= returns KiB.
            match trimmed.parse::<f64>() {
                Ok(kib) => kib / 1024.0,
                Err(_) => 0.0,
            }
        }
        _ => 0.0,
    }
}

/// Render an optional measurement, substituting `missing` when a run measured
/// nothing.
///
/// A printed `0.0` reads as a measured zero — to a human, to the scripts that
/// grep the summary line, and to anything that averages the CSV column. The
/// substitute is `n/a` on the console and an empty field in CSV, both of which
/// fail a numeric parse instead of quietly weighting a mean.
fn fmt_measurement(value: Option<f64>, decimals: usize, missing: &str) -> String {
    value.map_or_else(|| missing.to_string(), |v| format!("{v:.decimals$}"))
}

/// Per-phase timing computed from the per-token `step_fn` callback wall-clocks.
///
/// `generate_greedy` invokes `step_fn` once per produced token. The FIRST call
/// fires for the prefill-produced token (step 0); subsequent calls fire per
/// steady-state decode step. So:
/// - `ttft_ms` = wall-clock from generate-start to the first callback
///   (prefill + first token).
/// - `decode_tps` = `(n_generated - 1) / (last_cb - first_cb)` — steady-state
///   decode of tokens 2..N, prefill/first-token cost EXCLUDED.
/// - `overall_tps` = `n_generated / total_elapsed` — the historical combined
///   number (prefill + decode), kept for the `overall_tps` metric.
/// - `prefill_tps` = `prompt_tokens / ttft_s` — prefill throughput.
///
/// Every field is `Option`: a run that produced no token measured no rate, and
/// `0.0` in that slot is a fabricated measurement — it prints, averages and
/// (once recorded) wins a `bests` cell exactly like a real throughput of zero.
/// `None` is the only honest answer, and the recorder writes no row for it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PhaseTiming {
    pub ttft_ms: Option<f64>,
    pub decode_tps: Option<f64>,
    pub overall_tps: Option<f64>,
    pub prefill_tps: Option<f64>,
}

/// Compute phase timings from raw inputs.
///
/// `first_cb_s` / `last_cb_s` are the elapsed-seconds (relative to
/// generate-start) of the first and last `step_fn` callbacks. `total_s` is the
/// full generate wall-clock. `n_generated` is the token count; `prompt_tokens`
/// the prefill length.
///
/// Invariant (asserted by the caller): with `n_generated >= 2` and a non-zero
/// decode window, `decode_tps >= overall_tps` — removing the fixed prefill cost
/// from the denominator can only raise TPS.
pub(crate) fn compute_phase_timing(
    first_cb_s: f64,
    last_cb_s: f64,
    total_s: f64,
    n_generated: usize,
    prompt_tokens: usize,
) -> PhaseTiming {
    // No callback ever fired, or it landed at t=0: nothing was timed, so
    // nothing is reported. Same condition as `prefill_tps` below, which is
    // derived from this same first-callback timestamp — a state one field
    // calls unmeasured cannot be a measurement in the other.
    let ttft_ms = (n_generated > 0 && first_cb_s > 0.0).then_some(first_cb_s * 1000.0);

    // Decode window: tokens 2..N over (last_cb - first_cb). Needs >= 2 tokens
    // and a positive window; otherwise fall back to the combined number.
    let decode_window_s = last_cb_s - first_cb_s;
    let overall_tps = (total_s > 0.0 && n_generated > 0).then(|| n_generated as f64 / total_s);
    let decode_tps = if n_generated >= 2 && decode_window_s > 0.0 {
        Some((n_generated as f64 - 1.0) / decode_window_s)
    } else {
        overall_tps
    };
    let prefill_tps =
        (first_cb_s > 0.0 && prompt_tokens > 0).then(|| prompt_tokens as f64 / first_cb_s);

    PhaseTiming {
        ttft_ms,
        decode_tps,
        overall_tps,
        prefill_tps,
    }
}

/// One message in a chat-JSON prompt fixture (`prompts/longctx_<N>k.json`):
/// `{"messages": [{"role": "...", "content": "..."}, ...], ...}`.
#[derive(Debug, serde::Deserialize)]
struct ChatFixtureMessage {
    role: String,
    content: String,
}

/// `true` when `value` is a JSON object carrying a non-empty `messages`
/// array -- the single "is this a chat-JSON fixture" predicate shared by
/// `parse_chat_fixture` (tokenization, below) and the CLI's `--prompt-tokens`
/// record-body embedding (`main.rs`), so the two call sites cannot disagree
/// on edge cases such as `messages: []`.
pub(crate) fn is_chat_fixture(value: &serde_json::Value) -> bool {
    value
        .get("messages")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|arr| !arr.is_empty())
}

/// Parse `prompt_text` as a chat-JSON fixture.
///
/// Returns `Ok(None)` when `prompt_text` is not a chat-JSON fixture at all --
/// not valid JSON, not a JSON object, no `messages` key, `messages` is not an
/// array, or the array is empty (per [`is_chat_fixture`]) -- the caller then
/// falls back to tokenizing `prompt_text` verbatim as raw text (the
/// plain-`.txt` fixture path).
///
/// Returns `Err` when `prompt_text` unambiguously IS a chat-JSON fixture (a
/// non-empty `messages` array) but one or more elements do not deserialize
/// into the `{"role": "<string>", "content": "<string>"}` shape this baseline
/// harness supports -- e.g. an OpenAI parts-array `content`, `content:
/// null`, or a message missing `role`. Silently falling back to
/// raw-envelope tokenization in that case would record a wrong measurement
/// with no error, so this matches the existing loud-failure convention for a
/// missing `chat_template.jinja`.
/// Tokenize a prompt file the way the bench harnesses measure it.
///
/// Returns `(prompt_ids, template_used)`. A chat-JSON fixture is rendered
/// through the model's `chat_template.jinja` first; anything else is tokenized
/// as raw text with `add_special_tokens=true`. Shared by `rmlx baseline` and
/// `rmlx bench` so the two cannot disagree on what "N prompt tokens" means.
pub(crate) fn tokenize_prompt_text(
    model_path: &Path,
    tokenizer: &tokenizers::Tokenizer,
    prompt_text: &str,
) -> anyhow::Result<(Vec<u32>, bool)> {
    if let Some(messages) = parse_chat_fixture(prompt_text)? {
        return Ok((
            tokenize_chat_fixture(model_path, tokenizer, &messages)?,
            true,
        ));
    }
    let encoding = tokenizer
        .encode(prompt_text, true)
        .map_err(|e| anyhow::anyhow!("tokenize prompt: {e}"))?;
    Ok((encoding.get_ids().to_vec(), false))
}

fn parse_chat_fixture(prompt_text: &str) -> anyhow::Result<Option<Vec<ChatFixtureMessage>>> {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(prompt_text) else {
        return Ok(None);
    };
    if !is_chat_fixture(&value) {
        return Ok(None);
    }
    // `is_chat_fixture` already proved `messages` is present and a non-empty
    // array; `.get()` (not indexing) keeps this panic-free regardless.
    let Some(messages) = value.get("messages") else {
        return Ok(None);
    };
    serde_json::from_value(messages.clone())
        .map(Some)
        .map_err(|e| {
            anyhow::anyhow!(
                "prompt is a chat-JSON fixture (non-empty \"messages\" array) but its elements \
                 do not match the expected {{\"role\": \"<string>\", \"content\": \"<string>\"}} \
                 shape: {e}"
            )
        })
}

/// Render `messages` through `<model_path>/chat_template.jinja` and tokenize
/// the result -- the same render-then-tokenize path the HTTP
/// chat-completions route uses (`crates/rmlx-server/src/openai/chat.rs`), so
/// `--prompt-tokens N` measures N *content* tokens rather than a chat-JSON
/// fixture's raw envelope + syntax tokens.
fn tokenize_chat_fixture(
    model_path: &Path,
    tokenizer: &tokenizers::Tokenizer,
    messages: &[ChatFixtureMessage],
) -> anyhow::Result<Vec<u32>> {
    let template_src =
        rmlx_server::chat_template::load_template_source(model_path).map_err(|e| {
            anyhow::anyhow!(
                "prompt is a chat-JSON fixture but {} has no usable chat_template.jinja: {e}",
                model_path.display()
            )
        })?;
    let template = rmlx_server::chat_template::ChatTemplate::new(template_src)
        .map_err(|e| anyhow::anyhow!("compile chat_template.jinja: {e}"))?;
    let cfg = rmlx_server::tokenizer_io::load_tokenizer_config(model_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer_config.json: {e}"))?;
    let bos_token = cfg.bos_token.unwrap_or_default();
    let eos_token = cfg.eos_token.unwrap_or_default();

    let tpl_messages: Vec<rmlx_server::chat_template::ChatMessageTpl<'_>> = messages
        .iter()
        .map(|m| rmlx_server::chat_template::ChatMessageTpl {
            role: m.role.as_str(),
            content: m.content.as_str(),
            ..Default::default()
        })
        .collect();
    let opts = rmlx_server::chat_template::RenderOpts {
        bos_token: &bos_token,
        eos_token: &eos_token,
        add_generation_prompt: true,
        tools: &[],
        enable_thinking: None,
    };
    let rendered = template
        .render(&tpl_messages, &opts)
        .map_err(|e| anyhow::anyhow!("render chat_template.jinja: {e}"))?;

    let ids = rmlx_server::tokenizer_io::encode(tokenizer, &rendered.text)
        .map_err(|e| anyhow::anyhow!("tokenize rendered chat prompt: {e}"))?;
    if ids.is_empty() {
        return Err(anyhow::anyhow!(
            "chat-JSON fixture rendered through chat_template.jinja encoded to zero tokens"
        ));
    }
    Ok(ids)
}

/// Record a performance baseline for the given model snapshot.
///
/// Steps:
/// 1. Parse device, read prompt file, tokenize. A chat-JSON fixture
///    (`{"messages": [...], ...}`, e.g. `prompts/longctx_<N>k.json`) is
///    rendered through the model's real `chat_template.jinja` first, so the
///    token count reflects the message content rather than the fixture's JSON
///    envelope + syntax.
/// 2. `arch::load_model` -- capture `load_ms`.
/// 3. Resolve the context ceiling from the loaded checkpoint, then apply the
///    prompt cap (`--max-prompt-tokens`, defaulting to that ceiling) via
///    `resolve_prompt_truncation` -- truncates on `--device cpu` or an
///    explicit opt-in, errors loudly on `--device gpu` with the default cap.
/// 4. `arch.generate_greedy` -- per-token `step_fn` callback captures wall-clock
/// so prefill (TTFT) and steady-state decode are timed SEPARATELY.
/// 5. Compute decode-only TPS, measured TTFT, first-50-token preview, peak RSS,
///    and the Metal-allocator peak bracketed across prefill+decode.
/// 6. Emit EventRecorder records and 1 baseline.csv row.
/// 7. Print one-line summary to stdout, plus the exact generated token-id
///    sequence when `emit_token_ids` is set.
#[tracing::instrument(skip_all, fields(
    model_dir = %model_path.display(),
    device = device_str,
    max_tokens,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_baseline(
    model_path: &Path,
    prompt_path: &Path,
    device_str: &str,
    max_tokens: u32,
    run_id: &str,
    prompt_label: &str,
    kv_quant_override: Option<rmlx_kv_quant::KvQuant>,
    max_ctx_override: Option<i32>,
    max_prompt_tokens: Option<usize>,
    allow_truncate: bool,
    yarn_override: Option<rmlx_models::qwen3::YarnOverride>,
    emit_token_ids: bool,
    sink: &EventRecorder,
    record_args: Option<BaselineRecordArgs<'_>>,
) -> anyhow::Result<()> {
    // -- Validate device -------------------------------------------------------
    let device = match device_str {
        "cpu" => Device::Cpu,
        "gpu" => Device::Gpu,
        other => {
            return Err(anyhow::anyhow!(
                "--device must be 'cpu' or 'gpu', got '{other}'"
            ));
        }
    };

    // -- Read prompt file -------------------------------------------------------
    let prompt_text = std::fs::read_to_string(prompt_path)
        .map_err(|e| anyhow::anyhow!("cannot read prompt file {}: {e}", prompt_path.display()))?;

    // -- Load tokenizer --------------------------------------------------------
    let tk_path = model_path.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tk_path)
        .map_err(|e| anyhow::anyhow!("cannot load tokenizer.json: {e}"))?;

    // A chat-JSON fixture (`{"messages": [...], ...}`) is rendered through the
    // model's real chat_template.jinja and only the resulting content tokens
    // are counted -- matching the HTTP chat-completions path. Anything else
    // (e.g. the default plain-text fixture) is tokenized as raw text with
    // add_special_tokens=true so BOS is prepended naturally.
    let (mut prompt_ids, template_used) =
        tokenize_prompt_text(model_path, &tokenizer, &prompt_text)?;

    // -- Load model, capture load_ms -------------------------------------------
    let ts_load_start = Instant::now();
    let model = arch::load_model(
        model_path,
        device,
        &arch::LoadOpts {
            yarn: yarn_override,
        },
    )
    .map_err(|e| anyhow::anyhow!("arch::load_model: {e}"))?;
    let load_ms = ts_load_start.elapsed().as_millis() as f64;

    info!(load_ms, "baseline: model loaded");

    // Truncate to the prompt cap when it allows it; on GPU with the default
    // (non-explicit) cap this is a hard error instead -- see
    // `resolve_prompt_truncation`. The default cap is the run's resolved
    // context ceiling, which is why this runs after the load.
    let resolved_ctx =
        rmlx_models::context::resolve_context(&model.context_limits(), max_ctx_override)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    let effective_len = resolve_prompt_truncation(
        prompt_ids.len(),
        max_prompt_tokens,
        resolved_ctx.ceiling.max(0) as usize,
        device,
        allow_truncate,
    )?;
    prompt_ids.truncate(effective_len);
    let prompt_token_count = prompt_ids.len();

    info!(
        model = %model_path.display(),
        device = device_str,
        prompt_tokens = prompt_token_count,
        max_ctx = resolved_ctx.ceiling,
        max_tokens,
        template_used,
        "baseline: starting"
    );

    // -- Generate, capture per-phase timing ------------------------------------
    // `generate_greedy` calls `step_fn` once per produced token. The FIRST call
    // fires for the prefill-produced token (TTFT); each later call fires per
    // steady-state decode step. We record the elapsed-seconds of the first and
    // last callbacks so prefill cost is timed SEPARATELY from decode, without
    // changing `generate_greedy`'s signature (serve/chat/info paths untouched).

    // A7.2: baseline bench is always greedy (temperature 0.0) — keeps the
    // untouched GPU argmax path. Fresh per-call sampler config + rng.
    let baseline_sampler_cfg = rmlx_models::SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: None,
        top_logprobs_k: 0,
    };
    let mut baseline_rng = rmlx_models::Pcg32::new(baseline_sampler_cfg.seed_or_default());
    let baseline_penalty_cfg = rmlx_models::PenaltyConfig::default();
    let mut baseline_token_history: Vec<u32> = Vec::new();

    // Task 8: decode_loop top-level info span. Carries prompt_tokens, gen_tokens,
    // arch_class, kv_quant as structured fields. Span duration = prefill + decode
    // wall-clock (the arch_class and kv_quant are fixed per run so structuring
    // them here — before the call — is accurate). gen_tokens is updated via the
    // returned `n_generated` but the span is built pre-call; the elapsed_ms
    // emitted by the JSONL subscriber reflects the true wall-clock.
    let arch_class = model.arch_class();
    let kv_quant_str = kv_quant_override.map_or_else(|| "auto".to_owned(), |q| q.to_string());
    let decode_span = info_span!(
        "decode_loop",
        prompt_tokens = prompt_token_count,
        gen_tokens = max_tokens,
        arch_class,
        kv_quant = %kv_quant_str,
    );
    let _decode_span_guard = decode_span.enter();
    // Scope the Metal allocator high-water mark to prefill+decode. A bare
    // reading is process-lifetime and so carries the model load with it, which
    // makes it useless for comparing two runs of the same model. Opened before
    // the timer starts and closed after it stops so the reset/read cost cannot
    // land inside a measured interval.
    let peak_bracket = rmlx_mlx::PeakBracket::open();
    let ts_generate_start = Instant::now();
    // Per-token callback wall-clocks (elapsed seconds since generate-start).
    let mut first_cb_s: Option<f64> = None;
    let mut last_cb_s: f64 = 0.0;
    let mut step_timing = |_step: &rmlx_models::ProbeStep| -> Option<u32> {
        let elapsed = ts_generate_start.elapsed().as_secs_f64();
        if first_cb_s.is_none() {
            first_cb_s = Some(elapsed);
        }
        last_cb_s = elapsed;
        None
    };
    let kv_before = model.kv_cache_bytes_sample();
    let steps = model
        .generate_greedy(
            &tokenizer,
            &prompt_ids,
            max_tokens as usize,
            device,
            kv_quant_override,
            max_ctx_override,
            1,   // baseline bench: single-slot cache
            &[], // baseline bench: no EOS-stop, force full max_tokens steps
            &mut step_timing,
            None, // A6.2: baseline never uses sampler constraints.
            &baseline_sampler_cfg,
            &mut baseline_rng,
            &baseline_penalty_cfg,
            &mut baseline_token_history,
        )
        .map_err(|e| anyhow::anyhow!("generate_greedy: {e}"))?;
    let generate_elapsed = ts_generate_start.elapsed();
    let peak_reading = peak_bracket.close();
    // Read actual on-device KV-cache bytes from the per-instance counter that
    // `generate_greedy` writes via `store_kv_cache_bytes`.
    //
    // `Unreported` is a refusal, not a warning. Baseline runs exactly one
    // generation per process with EOS-stop disabled, and every architecture
    // stores through the same counter, so the sequence failing to advance means
    // the generation left before its store — it ended early on a path that did
    // not report the failure. Downgrading that to `warn!` + `0` produced a
    // summary line and, under `--record`, a permanent row in the append-only
    // store, from a run that never generated: the whole measurement is void, not
    // just its KV column. `rmlx bench` already refuses the identical verdict.
    //
    // `ReportedZero` stays a warning: there the generation ran to completion and
    // only its KV column is unusable, so omitting that one column — which the
    // downstream `> 0` gates already do — keeps an otherwise valid timing
    // measurement.
    //
    // Note this DIVERGES from `rmlx bench`, which refuses both verdicts. The two
    // commands have different products: bench records `kv_cache_bytes` as a
    // first-class result, so a broken byte count voids its output, while
    // baseline's product is a timing row that survives a missing KV column
    // intact. They agree on `Unreported` because that one is not a broken
    // column — it means the generation never finished.
    let kv_cache_bytes: u64 =
        match rmlx_models::classify_kv_bytes(kv_before, model.kv_cache_bytes_sample()) {
            rmlx_models::KvBytesVerdict::Reported(n) => n,
            rmlx_models::KvBytesVerdict::Unreported => {
                return Err(anyhow::anyhow!(
                    "generation on arch {} reported no KV-cache byte count (the store sequence \
                     did not advance), so it returned before its post-decode store. A run that \
                     did not complete a decode loop is not a measurement — refusing to report or \
                     record it",
                    model.arch_class()
                ));
            }
            rmlx_models::KvBytesVerdict::ReportedZero => {
                tracing::warn!(
                    arch = model.arch_class(),
                    "generation reported a KV cache of 0 bytes after a real prefill — the byte \
                 accounting is wrong, not the cache; omitting kv_cache_bytes from this run"
                );
                0
            }
        };
    // Drop the span guard explicitly so the span closes (and its elapsed_ms is
    // emitted to the JSONL log) right here — before the summary println below.
    drop(_decode_span_guard);

    let n_generated = steps.len();

    // Decode-only timing: prefill (TTFT) excluded from the decode denominator.
    let timing = compute_phase_timing(
        first_cb_s.unwrap_or(0.0),
        last_cb_s,
        generate_elapsed.as_secs_f64(),
        n_generated,
        prompt_token_count,
    );
    let ttft_ms = timing.ttft_ms;
    let decode_tps = timing.decode_tps;
    let overall_tps = timing.overall_tps;
    let prefill_tps = timing.prefill_tps;

    // Invariant: removing the fixed prefill cost from the denominator can only
    // raise TPS. decode_tps must be >= the combined overall_tps for the run.
    debug_assert!(
        n_generated < 2
            || match (decode_tps, overall_tps) {
                (Some(d), Some(o)) => d + 1e-9 >= o,
                _ => true,
            },
        "decode_tps ({decode_tps:?}) must be >= overall_tps ({overall_tps:?})"
    );

    // `tps` retains the decode-only number for downstream summary + the
    // `decode_tps_warm` metric column (the corrected, prefill-excluded value).
    let tps: Option<f64> = decode_tps;

    // -- Peak RSS (after model + generation memory is committed) ---------------
    // `peak_rss_mb` returns 0.0 when the platform probe fails. A live process
    // always occupies memory, so that zero is a failed read, not a reading.
    let rss_mb = peak_rss_mb();
    let rss_mb_measured = (rss_mb > 0.0).then_some(rss_mb);

    // -- First-50-token preview ------------------------------------------------
    let preview: String = steps
        .iter()
        .take(50)
        .map(|s| s.piece.as_ref())
        .collect::<Vec<_>>()
        .join("");

    // If fewer than 50 tokens were generated, note it.
    let preview_full = if n_generated < 50 {
        format!("{preview}<EOS>")
    } else {
        preview
    };
    // Truncate to 200 chars for notes field.
    let notes_truncated: String = preview_full.chars().take(200).collect();

    // Emit properly decoded text for the smoke runner's
    // `extract_decoded_from_trace`. Raw `id_to_token()` pieces use tokenizer-
    // specific space markers (SentencePiece `▁`, GPT-2 `Ġ`) that the runner's
    // validate_regex cannot match. `tokenizer.decode()` converts the token ids
    // back to human-readable text with proper spaces. Falls back to the raw
    // piece preview on decode failure (best-effort, non-fatal).
    // Emit ALL generated tokens so the smoke runner's validate_regex can
    // match anywhere in the output (not just the first 50 tokens). Thinking
    // models (Bonsai / Qwen3.6) spend tokens on <think> blocks before the
    // actual answer; with a 50-token cap those models would fail instruction
    // prompts whose answer begins after token 50. The full decode is written
    // as a single tracing field; the runner's extract_decoded_from_trace
    // strips ANSI and captures the quoted value.
    let token_ids: Vec<u32> = steps.iter().map(|s| s.token_id).collect();
    let decoded_text: String = tokenizer
        .decode(&token_ids, true)
        .unwrap_or_else(|_| preview_full.clone());
    info!(decoded = ?decoded_text, "baseline: decoded preview");

    // -- Model basename --------------------------------------------------------
    let model_basename = model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("(unknown)");

    // -- Stdout summary -------------------------------------------------------
    // `decode_tps` is steady-state (prefill excluded); `overall_tps` is the
    // combined prefill+decode number kept for cross-reference.
    //
    // `metal_peak_mb` is the most Metal-allocator bytes live at once during
    // prefill+decode; `metal_gen_alloc_mb` is that figure minus what was
    // already live when generation began, i.e. what the generation itself
    // needed on top of the resident weights. The second number is the one that
    // compares across two runs of the same model — the first still carries the
    // weights. Both read 0 on a build with no Metal allocator.
    //
    // When the bracket could not be scoped — no Metal allocator, or the peak
    // reset failed while the live reading succeeded — both fields report 0
    // rather than the process-lifetime peak. Printing the unscoped figure under
    // a name that promises a scoped one is how a consumer ends up diffing two
    // numbers that describe the process instead of the generation.
    let (metal_peak_mb, metal_gen_alloc_mb) = if peak_reading.measurable() {
        (
            peak_reading.peak_bytes as f64 / 1_048_576.0,
            peak_reading.headroom_bytes() as f64 / 1_048_576.0,
        )
    } else {
        warn!(
            "Metal peak bracket unavailable (allocator absent, or the peak reset failed); \
             reporting metal_peak_mb=0 rather than an unscoped process-lifetime figure"
        );
        (0.0, 0.0)
    };
    println!(
        "baseline: model={model_basename}  load={load_ms:.0}ms  ttft_ms={ttft}  \
         decode_tps={decode}  overall_tps={overall}  prefill_tps={prefill}  \
         prompt_tokens={prompt_token_count}  peak_rss={rss}MB  \
         metal_peak_mb={metal_peak_mb:.1}  metal_gen_alloc_mb={metal_gen_alloc_mb:.1}  \
         kv_cache_bytes={kv_bytes}",
        ttft = fmt_measurement(ttft_ms, 0, "n/a"),
        decode = fmt_measurement(decode_tps, 3, "n/a"),
        overall = fmt_measurement(overall_tps, 3, "n/a"),
        prefill = fmt_measurement(prefill_tps, 1, "n/a"),
        rss = fmt_measurement(rss_mb_measured, 1, "n/a"),
        // `ReportedZero` above already decided this column is unusable for this
        // run; printing the literal `0` would read as "this codec's cache is
        // empty" and average into a residency comparison as a real reading.
        // `n/a` is the same refusal the rate columns use.
        kv_bytes = if kv_cache_bytes > 0 {
            kv_cache_bytes.to_string()
        } else {
            "n/a".to_string()
        },
    );

    // Exact generated token-id sequence, one line, opt-in.
    //
    // An A/B harness needs to prove two arms produced the *same* tokens, and
    // the decoded text cannot do that: two different id sequences can decode to
    // the same string. This is the same `Vec<u32>` identity the golden-token
    // tests assert, emitted from the run that also measured the speed, so a
    // change that is fast and wrong fails the invocation that made it look fast.
    if emit_token_ids {
        let ids_csv = token_ids
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        println!("baseline: token_ids={ids_csv}");
    }

    // -- EventRecorder records --------------------------------------------------
    let abs_path = model_path
        .canonicalize()
        .unwrap_or_else(|_| model_path.to_path_buf());
    let abs_path_str = abs_path.to_string_lossy();

    let cfg = rmlx_loader::load_config(model_path).ok();
    let quant_mode = cfg
        .as_ref()
        .and_then(|c| c.quantization.as_ref())
        .map_or_else(
            || "none".to_owned(),
            |q| format!("{} g{}", q.mode_or_default(), q.group_size),
        );

    // New CSV columns: quantization_type (mode g<gs> b<bits>), context_size.
    let quantization_type = cfg
        .as_ref()
        .and_then(|c| c.quantization.as_ref())
        .map(|q| format!("{} g{} b{}", q.mode_or_default(), q.group_size, q.bits))
        .unwrap_or_default();
    let context_size: u32 = cfg
        .as_ref()
        .and_then(|c| c.text_config.as_ref())
        .and_then(|tc| tc.max_position_embeddings)
        .unwrap_or(0);

    // A phase this run did not measure records nothing — an event row reading
    // `0` tok/s is indistinguishable from a measured stall.
    let metrics_data: &[(&str, &str, Option<f64>)] = &[
        ("baseline_load_ms", "ms", Some(load_ms)),
        ("baseline_ttft_ms", "ms", ttft_ms),
        ("baseline_tps", "tok/s", decode_tps),
        ("baseline_overall_tps", "tok/s", overall_tps),
        ("baseline_prefill_tps", "tok/s", prefill_tps),
        (
            "baseline_prompt_tokens",
            "count",
            Some(prompt_token_count as f64),
        ),
        ("baseline_peak_rss_mb", "MB", rss_mb_measured),
    ];

    for (op, unit, measured) in metrics_data {
        let Some(value) = *measured else { continue };
        sink.record(&rmlx_metrics::events::Measurement {
            model_path: &abs_path_str,
            quant_mode: &quant_mode,
            stage: "baseline",
            op,
            value_unit: unit,
            value,
            notes: &notes_truncated,
        })
        .map_err(|e| anyhow::anyhow!("metrics record {op}: {e}"))?;
    }

    // Emit kv_cache_bytes only when non-zero (un-wired archs return 0; omitting
    // them matches the `build_run_record` gate and prevents a spurious
    // "measured 0-byte KV" row in the events table).
    if kv_cache_bytes > 0 {
        sink.record(&rmlx_metrics::events::Measurement {
            model_path: &abs_path_str,
            quant_mode: &quant_mode,
            stage: "baseline",
            op: "baseline_kv_cache_bytes",
            value_unit: "bytes",
            value: kv_cache_bytes as f64,
            notes: &notes_truncated,
        })
        .map_err(|e| anyhow::anyhow!("metrics record baseline_kv_cache_bytes: {e}"))?;
    }

    // -- Append to metrics/baseline.csv ---------------------------------------
    // Gated on the same kill switch as the DB. `--metrics off` means this run
    // records nothing, and baseline.csv is a metrics surface under
    // <RMLX_HOME>/metrics/ exactly like runs.db is — an append here is as
    // permanent as a row there. Left ungated, a run with the DB shut off (a GPU
    // capture, say, whose timings are meaningless by construction) still left a
    // row behind.
    if rmlx_metrics::mode::current() == rmlx_metrics::mode::MetricsMode::Off {
        info!("baseline: --metrics off, no baseline.csv row written");
    } else {
        let ts_utc = chrono::Utc::now()
            .format("%Y-%m-%dT%H:%M:%S%.3fZ")
            .to_string();

        // Truncate first-50 to 200 chars then CSV-escape.
        let output_cell = baseline_csv_escape(&notes_truncated);

        let csv_path = rmlx_core::paths::metrics_dir().join("baseline.csv");

        // Header written only when the file is absent or zero-length.
        let write_header =
            !csv_path.exists() || std::fs::metadata(&csv_path).map_or(true, |m| m.len() == 0);

        let mut csv_file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&csv_path)
            .map_err(|e| anyhow::anyhow!("open {}: {e}", csv_path.display()))?;

        if write_header {
            csv_file.write_all(
                b"run_id,timestamp_utc,backend,model_basename,quantization_type,context_size,\
                  prompt,device,prompt_tokens,load_ms,ttft_ms,tps,peak_rss_mb,output_first_50\n",
            )?;
        }

        let row = format!(
            "{},{},rMLX,{},{},{},{},{},{},{:.0},{},{},{},{}\n",
            baseline_csv_escape(run_id),
            ts_utc,
            baseline_csv_escape(model_basename),
            baseline_csv_escape(&quantization_type),
            context_size,
            baseline_csv_escape(prompt_label),
            baseline_csv_escape(device_str),
            prompt_token_count,
            load_ms,
            fmt_measurement(ttft_ms, 0, ""),
            fmt_measurement(tps, 3, ""),
            fmt_measurement(rss_mb_measured, 1, ""),
            output_cell,
        );
        csv_file
            .write_all(row.as_bytes())
            .map_err(|e| anyhow::anyhow!("write baseline.csv row: {e}"))?;
        csv_file
            .flush()
            .map_err(|e| anyhow::anyhow!("flush baseline.csv: {e}"))?;
    }

    info!(
        load_ms,
        ttft_ms,
        tps,
        prompt_tokens = prompt_token_count,
        n_generated,
        rss_mb,
        "baseline: complete"
    );

    // -- §8.5 universal record (Phase-8 bench harness) ------------------------
    if let Some(record_args) = record_args.as_ref() {
        // Checked before building anything: `--metrics off` means a no-op at
        // the producer, not "build the record, then throw it away".
        if !rmlx_metrics::mode::observations_enabled() {
            info!("baseline: observations disabled, no record written");
            return Ok(());
        }

        let weight_quant_str = cfg
            .as_ref()
            .and_then(|c| c.quantization.as_ref())
            .map_or_else(
                || "bf16".to_string(),
                |q| match q.mode_or_default() {
                    // Existing whitelist values: "mxfp8", "mxfp4", "nvfp4", "8bit",
                    // "6bit", "4bit", "3bit", "2bit". Pass through the quant mode
                    // if it already matches; otherwise fall back to a bit-only tag.
                    "mxfp8" | "mxfp4" | "nvfp4" => q.mode_or_default().to_string(),
                    _ => format!("{}bit", q.bits),
                },
            );

        // First-64-char preview for the `output_first_64` column.
        let preview_64: String = preview_full.chars().take(64).collect();

        let record = build_run_record(
            run_id,
            model_path,
            record_args,
            prompt_label,
            &prompt_text,
            &weight_quant_str,
            prompt_token_count as i64,
            i64::from(max_tokens),
            load_ms,
            ttft_ms,
            decode_tps,
            overall_tps,
            prefill_tps,
            rss_mb_measured,
            n_generated,
            &preview_64,
            kv_cache_bytes,
        )?;

        let path = write_buffer_record(&record)?;
        info!(path = %path.display(), "baseline: wrote §8.5 ingest record");

        // Inline ingest so the row is visible to `rmlx metrics champions` /
        // `bests` right away. Drop the buffer file on success; leave it for
        // `rmlx metrics record --replay-pending` to pick up on failure.
        let db_path = rmlx_core::paths::metrics_db_path();
        match rmlx_metrics::schema::open(&db_path) {
            Ok(mut conn) => {
                let inserted_by = RunIdentity::get().inserted_by("rmlx-cli");
                let mut rec_inst = rmlx_metrics::recorder::Recorder::new(&mut conn, inserted_by);
                let run: rmlx_metrics::ingest::RunRecord = serde_json::from_value(record)
                    .map_err(|e| anyhow::anyhow!("deserialize RunRecord: {e}"))?;
                match rec_inst.record_run(&run) {
                    Ok(outcome) => {
                        info!(
                            run_id = %outcome.run_id,
                            inserted = outcome.observation_ids.len(),
                            "baseline: ingested record into runs.db"
                        );
                        // Drop the buffer file: row is durable in the DB.
                        let _ = std::fs::remove_file(&path);
                    }
                    Err(e) => {
                        warn!(
                            error = %e,
                            buffer = %path.display(),
                            "baseline: inline ingest failed; leaving buffer file for replay"
                        );
                    }
                }
            }
            Err(e) => {
                warn!(
                    error = %e,
                    buffer = %path.display(),
                    "baseline: cannot open metrics DB; leaving buffer file for replay"
                );
            }
        }
    }

    Ok(())
}

/// Arguments needed to emit one §8.5 universal `RunRecord` to the metrics
/// ingest buffer at the end of a baseline run. Populated by `main.rs` when
/// the caller passes `--record`.
pub(crate) struct BaselineRecordArgs<'a> {
    /// Free-form `--label` value. Goes into the `notes` column.
    pub label: Option<&'a str>,
    /// Resolved canonical prompt id (e.g. `longctx_4k`) or `None` when the
    /// caller supplied `--prompt` / `--prompt-file`.
    pub prompt_id: Option<&'a str>,
    /// Full prompt JSON body to embed by-value. When `None`, falls back to a
    /// sha256-only `PromptRef` derived from the file text.
    pub prompt_body: Option<serde_json::Value>,
    /// Final resolved `KvQuant` actually used by the run.
    pub kv_quant: rmlx_kv_quant::KvQuant,
    /// Final resolved `ctx_max`.
    pub ctx_max: i64,
    /// Caller-supplied `--git-sha` value, or `None`. Provenance only — the
    /// binary never derives this itself (see `RunIdentity`'s doc).
    pub git_sha: Option<&'a str>,
}

/// Resolve a `--prompt-tokens N` flag to the canonical `prompts/longctx_<N/1024>k.json`
/// path inside the workspace. Returns an `Err` whose message lists the
/// available files when the size is not on disk.
pub(crate) fn resolve_prompt_tokens_file(
    prompts_dir: &Path,
    n: u32,
) -> anyhow::Result<(PathBuf, String)> {
    if n == 0 || !n.is_multiple_of(1024) {
        return Err(anyhow::anyhow!(
            "--prompt-tokens must be a positive multiple of 1024 (e.g. 4096, 8192), got {n}"
        ));
    }
    let k = n / 1024;
    let name = format!("longctx_{k}k");
    let path = prompts_dir.join(format!("{name}.json"));
    if !path.exists() {
        let listing = std::fs::read_dir(prompts_dir)
            .ok()
            .map(|it| {
                let mut names: Vec<String> = it
                    .filter_map(Result::ok)
                    .filter_map(|e| e.file_name().into_string().ok())
                    .filter(|n| n.starts_with("longctx_") && n.ends_with(".json"))
                    .collect();
                names.sort();
                names.join(", ")
            })
            .unwrap_or_default();
        return Err(anyhow::anyhow!(
            "--prompt-tokens {n}: no such canonical prompt at {} \
             (available longctx files in {}: {})",
            path.display(),
            prompts_dir.display(),
            if listing.is_empty() {
                "<empty>"
            } else {
                listing.as_str()
            },
        ));
    }
    Ok((path, name))
}

// `kv_quant_whitelist_str` + `kv_quant_describe` were removed.
// The canonical KvQuant string is `<KvQuant as Display>` and the metrics
// whitelist accepts the long-form `mixed_k<kb>g<kg>_v<vb>g<vg>` directly.

/// Build a §8.5 universal `RunRecord` JSON value from a completed baseline run.
#[allow(clippy::too_many_arguments)]
fn build_run_record(
    run_id: &str,
    model_path: &Path,
    args: &BaselineRecordArgs<'_>,
    prompt_label_fallback: &str,
    prompt_text: &str,
    weight_quant: &str,
    prompt_tokens: i64,
    max_tokens: i64,
    load_ms: f64,
    ttft_ms: Option<f64>,
    decode_tps: Option<f64>,
    overall_tps: Option<f64>,
    prefill_tps: Option<f64>,
    rss_mb: Option<f64>,
    n_generated: usize,
    preview_first_64: &str,
    kv_cache_bytes: u64,
) -> anyhow::Result<serde_json::Value> {
    // Identity: namespace + model name from the snapshot directory basename.
    let snapshot_str = model_path
        .canonicalize()
        .unwrap_or_else(|_| model_path.to_path_buf())
        .to_string_lossy()
        .to_string();
    let (ns, model) = rmlx_metrics::identity::split_model_path(&snapshot_str)
        .map_err(|e| anyhow::anyhow!("split_model_path({snapshot_str}): {e}"))?;

    let ts_utc = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();

    // Prompt: prefer the canonical longctx body when known, else embed the
    // raw text under the supplied label.
    let (prompt_name, prompt_body) = match (args.prompt_id, args.prompt_body.clone()) {
        (Some(id), Some(body)) => (id.to_string(), body),
        _ => (
            prompt_label_fallback.to_string(),
            serde_json::Value::String(prompt_text.to_string()),
        ),
    };

    // Canonical KvQuant string lives in the `kv_quant` column (Display
    // form, incl. long-form `mixed_*`). `notes` carries only operator metadata.
    let mut notes = match args.label {
        Some(l) if !l.is_empty() => format!("label={l}"),
        _ => String::new(),
    };
    if n_generated < max_tokens as usize {
        if !notes.is_empty() {
            notes.push_str("; ");
        }
        // write!(String) is infallible — let _ discards the unit Ok.
        let _ = write!(notes, "early_stop=true n_generated={n_generated}");
    }

    let kv_quant_str = args.kv_quant.to_string();

    // `decode_tps_warm` now carries the prefill-EXCLUDED steady-state number;
    // `overall_tps` keeps the combined prefill+decode value; `prefill_tps` is
    // prompt throughput.  An unmeasured phase serializes to `null` and the
    // recorder writes no row for it — the §8.5 way of saying "not measured".
    // `kv_cache_bytes` is omitted when 0 (arch not yet wired) so the RunRecord
    // schema stays backward-compatible.
    let mut metrics = vec![
        serde_json::json!({ "name": "decode_tps_warm", "value": decode_tps }),
        serde_json::json!({ "name": "overall_tps",     "value": overall_tps }),
        serde_json::json!({ "name": "prefill_tps",     "value": prefill_tps }),
        serde_json::json!({ "name": "ttft_warm_ms",    "value": ttft_ms }),
        serde_json::json!({ "name": "model_load_ms",   "value": load_ms }),
        serde_json::json!({ "name": "peak_rss_mb",     "value": rss_mb }),
    ];
    if kv_cache_bytes > 0 {
        metrics.push(serde_json::json!({ "name": "kv_cache_bytes", "value": kv_cache_bytes }));
    }
    let metrics = serde_json::Value::Array(metrics);

    let mut record = serde_json::json!({
        "schema_version": rmlx_metrics::ingest::RECORD_SCHEMA_VERSION,
        "model_namespace": ns,
        "model": model,
        "weight_quant": weight_quant,
        "kv_quant": kv_quant_str,
        "ctx_max": args.ctx_max,
        "prompt": {
            "name": prompt_name,
            "body": prompt_body,
        },
        "ts_utc": ts_utc,
        "prompt_tokens": prompt_tokens,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "n_warmups": 0,
        "n_measure": 1,
        "output_first_64": preview_first_64,
        "notes": notes,
        "description": format!("baseline {run_id}"),
        "metrics": metrics,
    });

    // Identity (backend, backend_version, build_profile, hardware_tag) comes
    // from the single Rust source — baseline does not assemble it.
    // `stamp_json` deliberately does not touch `git_sha`: that field is
    // caller-supplied provenance (see `RunIdentity`'s doc), not something
    // this binary derives. `--git-sha` is the only source for it here.
    RunIdentity::get()
        .stamp_json(&mut record)
        .map_err(|e| anyhow::anyhow!("stamp run identity: {e}"))?;
    // Blank-string `--git-sha ""` is not provenance either — normalize it to
    // the same `None` a caller who omitted the flag gets.
    let git_sha = args.git_sha.filter(|s| !s.trim().is_empty());
    record
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("record is not a JSON object"))?
        .insert("git_sha".to_string(), serde_json::Value::from(git_sha));

    Ok(record)
}

/// Write a `RunRecord` JSON to `<RMLX_HOME>/metrics/buffer/pending/<ts>-<uniq>.json`.
fn write_buffer_record(rec: &serde_json::Value) -> anyhow::Result<PathBuf> {
    let dir = rmlx_core::paths::ingest_buffer_dir();
    std::fs::create_dir_all(&dir).map_err(|e| anyhow::anyhow!("create {}: {e}", dir.display()))?;
    let ts = chrono::Utc::now().format("%Y%m%dT%H%M%S%.3f");
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    let path = dir.join(format!("{ts}-rmlx-{pid}-{nanos}.json"));
    std::fs::write(&path, serde_json::to_string_pretty(rec)?)
        .map_err(|e| anyhow::anyhow!("write {}: {e}", path.display()))?;
    Ok(path)
}

// -- Unit tests for baseline helpers ----------------------------------------
#[cfg(test)]
#[path = "baseline_tests.rs"]
mod tests;
