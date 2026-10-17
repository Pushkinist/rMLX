//! Logprobs through the server's speculative generator.
//!
//! A request that asks for logprobs must get one record per emitted token from
//! every drafter path, and a request that asks for none must get none. The
//! cells drive `SpeculativeGenerator::generate`, so the request's
//! `top_logprobs` has to reach the round loop through the server's own call
//! site.
//!
//! The observable: the `logprobs` of each non-terminal `GenerationToken`.
//! What cannot move:
//!
//! - one record per emitted token, each with `top_logprobs` entries;
//! - the record's token is the surface of the step's `token_id`;
//! - at temperature 0 the emitted token is the top-ranked entry of its own
//!   record, with the same logprob to the bit: the verifier's argmax is the
//!   top of the row the record was built from;
//! - a request that asks for none carries no record.
//!
//! What this does not see: whether a record belongs to the right position when
//! the token also happens to be the top of a neighbouring row. A record built
//! from a row one position off ranks a different token first, so it fails the
//! top-entry check except where two adjacent rows share their argmax. The
//! plain decode path is not compared here: a block verify and a one-token
//! decode score one computation in two chunkings, so their logprobs differ by
//! bf16 noise.
//!
//! Run: `make gpu-test CRATE=rmlx-server FILTER=logprobs`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr
)]

mod thread_boundary;

use std::sync::Arc;

use futures::StreamExt;
use rmlx_mlx::Device;
use rmlx_server::{GenerationRequest, GenerationToken, Generator};
use thread_boundary::{
    one_thread_pool, release_cached_buffers, request, snapshot, speculative_generator, Load,
    GEMMA4_E2B, GEMMA4_E2B_ASSISTANT, GEMMA4_E4B, QWEN38_4BIT, QWEN38_MTP_4BIT, STEP_LIMIT,
};

/// Alternatives asked for per token.
const TOP_K: u32 = 2;
/// Tokens per request: several rounds at the default block of five.
const STEPS: u32 = 48;

fn logprob_request(dir: &std::path::Path, top_k: u32, temperature: f32) -> GenerationRequest {
    let mut req = request(dir, Vec::new(), Vec::new());
    req.max_tokens = STEPS;
    req.sampling.temperature = temperature;
    req.sampling.top_logprobs_k = top_k;
    req
}

/// Load the generator on the pool's one thread and serve `requests` there, in
/// order, keeping every item of every stream.
fn serve_tokens(load: &Load, requests: Vec<GenerationRequest>) -> Vec<Vec<GenerationToken>> {
    release_cached_buffers();
    let pool = one_thread_pool();
    let load = Arc::clone(load);
    let generator = pool
        .block_on(pool.spawn_blocking(move || load()))
        .expect("load task panicked");
    requests
        .into_iter()
        .map(|req| {
            pool.block_on(async {
                tokio::time::timeout(STEP_LIMIT, collect(&generator, req))
                    .await
                    .expect("the request outran its time limit")
            })
        })
        .collect()
}

async fn collect(generator: &Arc<dyn Generator>, req: GenerationRequest) -> Vec<GenerationToken> {
    let mut stream = generator.generate(req);
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(item.expect("the request failed"));
    }
    items
}

fn steps(items: &[GenerationToken]) -> Vec<&GenerationToken> {
    items.iter().filter(|t| !t.done).collect()
}

/// Every emitted token carries a record of `TOP_K` alternatives, for the token
/// it was emitted as. At `greedy`, that token is also the top-ranked one.
fn assert_records(
    test: &str,
    tokenizer: &tokenizers::Tokenizer,
    items: &[GenerationToken],
    greedy: bool,
) {
    let steps = steps(items);
    assert!(!steps.is_empty(), "{test}: the request emitted no token");
    for (i, step) in steps.iter().enumerate() {
        let lp = step
            .logprobs
            .as_ref()
            .unwrap_or_else(|| panic!("{test}: step {i} carries no logprob record"));
        assert_eq!(
            Some(lp.token.clone()),
            tokenizer.id_to_token(step.token_id),
            "{test}: step {i}: the record is for another token than the one emitted"
        );
        assert_eq!(
            lp.top_logprobs.len(),
            TOP_K as usize,
            "{test}: step {i}: the record holds the wrong number of alternatives"
        );
        assert!(
            lp.logprob.is_finite() && lp.logprob <= 0.0,
            "{test}: step {i}: logprob {} is not a log-probability",
            lp.logprob
        );
        assert!(
            lp.top_logprobs[0].logprob >= lp.top_logprobs[1].logprob,
            "{test}: step {i}: the alternatives are not in descending order"
        );
        if greedy {
            let top = &lp.top_logprobs[0];
            assert_eq!(
                top.token, lp.token,
                "{test}: step {i}: at temperature 0 the emitted token is the top-ranked one"
            );
            assert_eq!(
                top.logprob.to_bits(),
                lp.logprob.to_bits(),
                "{test}: step {i}: the top entry and the token's own logprob are one number"
            );
        }
    }
}

fn assert_none_asked(test: &str, items: &[GenerationToken]) {
    let steps = steps(items);
    assert!(!steps.is_empty(), "{test}: the request emitted no token");
    assert!(
        steps.iter().all(|t| t.logprobs.is_none()),
        "{test}: a request that asked for no logprobs got a record"
    );
}

fn assert_drafter(test: &str, verifier: (&str, &str), drafter: (&str, &str), also_sampled: bool) {
    let Some(v) = snapshot(test, verifier.0, verifier.1) else {
        return;
    };
    let Some(d) = snapshot(test, drafter.0, drafter.1) else {
        return;
    };
    let tokenizer = tokenizers::Tokenizer::from_file(v.join("tokenizer.json")).expect("tokenizer");
    let load = speculative_generator(v.clone(), d, Device::Gpu);

    let mut requests = vec![logprob_request(&v, TOP_K, 0.0), logprob_request(&v, 0, 0.0)];
    if also_sampled {
        requests.push(logprob_request(&v, TOP_K, 0.7));
    }
    let out = serve_tokens(&load, requests);

    assert_records(test, &tokenizer, &out[0], true);
    assert_none_asked(test, &out[1]);
    if also_sampled {
        assert_records(test, &tokenizer, &out[2], false);
    }
}

#[test]
#[ignore = "requires Metal GPU and the gemma-4-e2b + assistant snapshots; run with `make gpu-test`"]
fn the_assistant_pair_reports_logprobs_for_every_token() {
    assert_drafter(
        "the_assistant_pair_reports_logprobs_for_every_token",
        GEMMA4_E2B,
        GEMMA4_E2B_ASSISTANT,
        false,
    );
}

#[test]
#[ignore = "requires Metal GPU and the gemma-4-e4b + e2b snapshots; run with `make gpu-test`"]
fn the_two_model_pair_reports_logprobs_for_every_token() {
    assert_drafter(
        "the_two_model_pair_reports_logprobs_for_every_token",
        GEMMA4_E4B,
        GEMMA4_E2B,
        true,
    );
}

#[test]
#[ignore = "requires Metal GPU and the Qwen3.8-27B 4-bit + MTP snapshots; run with `make gpu-test`"]
fn the_qwen_mtp_pair_reports_logprobs_for_every_token() {
    assert_drafter(
        "the_qwen_mtp_pair_reports_logprobs_for_every_token",
        QWEN38_4BIT,
        QWEN38_MTP_4BIT,
        false,
    );
}
