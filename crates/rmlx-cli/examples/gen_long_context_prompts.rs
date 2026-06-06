// Long-context calibration-prompt generator.
//
// This is a dev-time example binary; it prints to stdout/stderr by design and
// has no library surface, so the lint suite is relaxed accordingly.

#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::indexing_slicing,
    missing_docs,
    reason = "dev-time example: user-facing CLI output, fixed-size lookup tables, no library API"
)]

//
// Synthesises 15 prompts targeting 4k–8k tokens each on Bonsai-8B-2bit's
// tokenizer (Qwen3 family) and writes them to the calibration-prompt JSON
// schema consumed by `rmlx kv-calibrate --recipe softmax_mass`.
//
// Composition:
//   * 5 × NIAH (needle-in-haystack) at ~4k tokens, varied needle depths.
//   * 5 × NIAH at ~8k tokens.
//   * 5 × long-form narrative/technical at ~4k tokens (no needle).
//
// Usage:
//   cargo run --example gen_long_context_prompts -p rmlx-cli -- \
//     <bonsai_tokenizer.json> <output.json>
//
// The example prints per-prompt token counts and an assertion that every
// prompt is >= 4096 tokens. Non-zero exit on failure.
//
// Determinism: each prompt is constructed from a fixed filler corpus and a
// fixed needle string. Re-running the generator yields byte-identical output
// (no RNG, no time-of-day).

use std::path::PathBuf;

use serde_json::json;
use tokenizers::Tokenizer;

// Target byte counts. Bonsai tokenizer averages ~4.8 bytes/token on this
// corpus, so we multiply by 5 (used downstream) and the target_bytes input
// to the builder is target_token * 5. Both 4k and 8k variants overshoot to
// guarantee >= 4096 (HARD_FLOOR) after tokenization.
const TARGET_4K: usize = 5200; // -> ~4400-4700 tokens after tokenize.
const TARGET_8K: usize = 9000; // -> ~7500-7900 tokens after tokenize.
const HARD_FLOOR: usize = 4096;

const NEEDLE_PREFIX: &str = "Important note: the secret code is";
const NEEDLE_VALUES: &[&str] = &[
    "AX7-PURPLE-FOX-9421",
    "BK2-CRIMSON-WHALE-5731",
    "QZ9-EMERALD-PHOENIX-3082",
    "JR4-AZURE-LYNX-6195",
    "MV5-AMBER-FALCON-7314",
    "NS8-VIOLET-OTTER-4628",
    "TY1-SAPPHIRE-WOLF-2197",
    "DH6-SCARLET-HERON-8053",
    "WP3-INDIGO-FOX-1746",
    "GU0-CORAL-EAGLE-9572",
];

// Distinct filler paragraphs ensure that haystacks for different prompts are
// not byte-identical, so the calibration sees a variety of K distributions.
const FILLERS: &[&str] = &[
    "The grass is green and the sun is yellow. Mountains rise tall above the silent valley below. \
     Rivers flow steadily toward the open sea. Quiet birds nest among the cedar boughs each spring, \
     and the rolling hills cast long shadows when the late afternoon sun begins its descent. ",
    "Engineers tracing a distributed bug reach for traces, metrics, and logs in that order, knowing \
     that a well-instrumented system surfaces its faults along orthogonal axes. The trick is to \
     correlate by causal id, not by clock. Wall clocks drift; happens-before edges do not. ",
    "In the workshop the apprentice studies dovetail joints. The pins and tails must mate flush, no \
     gaps, no glue squeeze-out. Old joiners cut by eye; new joiners cut by jig. Either way the wood \
     forgives nothing — once the saw bites past the line, the joint is scrap. ",
    "Cartographers of the early eighteenth century relied on latitude from celestial observations \
     and longitude from dead-reckoning, until John Harrison's marine chronometer finally let them \
     fix east-west position to within a few miles after a transatlantic voyage. ",
    "The kitchen filled with the smell of slow-cooked onions, browning at the edges. A bay leaf \
     drifted in the simmering broth, releasing oils only heat could draw out. The cook tasted, \
     adjusted salt, then left the pot to its own slow chemistry for another half hour. ",
    "A reviewer reads the diff first, then the test names, then the test bodies, then the file \
     header, then maybe the rest of the source. A reviewer who reads top-to-bottom ends up \
     reviewing what the author wrote, not what the change actually does. ",
    "Mercury, the smallest planet, races around the sun once every 88 days. Its day is twice as \
     long as its year if you measure sunrise to sunrise, a consequence of its slow rotation and \
     fast orbit. The surface is heavily cratered, scoured by solar wind. ",
    "Compilers fuse loops when the safety analysis says no observable side effect changes. \
     Allocators bunch small allocations to amortise per-call overhead. Networks coalesce packets \
     to amortise the kernel-to-userspace boundary. Bandwidth is rented in bulk. ",
    "The cellist drew the bow slowly, letting the string sing for almost two full seconds before \
     the next note overtook it. The phrase climbed gradually toward the upper register, then \
     resolved on the open A as the conductor lifted her hand. ",
    "Indexing strategies vary by access pattern: a covering index turns a lookup into a scan of \
     the index alone; a partial index serves a hot predicate without paying storage on the cold \
     tail; a bloom filter rejects negative lookups before any disk I/O is incurred at all. ",
    "On the high steppe the wind moves in long unbroken sweeps from west to east, lifting fine \
     loess into the air and depositing it again in drifts wherever the grass thickens. The herds \
     graze upwind in summer and downwind in winter, shifting weekly with the prevailing currents. ",
    "Speculative execution wins when the misprediction cost is small and the success rate is high. \
     It loses when the recovery is expensive and the alternatives are nearly free. Tuning is a \
     matter of measuring both edges of the trade. ",
];

fn build_haystack_target_bytes(needle: &str, depth_frac: f32, target_bytes: usize) -> String {
    let needle_sentence = format!("{NEEDLE_PREFIX} {needle}. Remember this code. ");
    let mut buf = String::with_capacity(target_bytes + needle_sentence.len() + 256);
    let needle_at = (target_bytes as f32 * depth_frac).clamp(0.0, target_bytes as f32) as usize;
    let mut planted = false;
    let mut filler_idx: usize = 0;
    while buf.len() < target_bytes {
        if !planted && buf.len() >= needle_at {
            buf.push_str(&needle_sentence);
            planted = true;
        }
        buf.push_str(FILLERS[filler_idx % FILLERS.len()]);
        filler_idx += 1;
    }
    if !planted {
        buf.push_str(&needle_sentence);
    }
    buf
}

fn build_niah_prompt(needle: &str, depth_frac: f32, target_bytes: usize) -> String {
    let haystack = build_haystack_target_bytes(needle, depth_frac, target_bytes);
    format!(
        "You are given a long document. Read it carefully and find the secret code embedded \
         somewhere in it. The secret code is a unique alphanumeric token. When you find it, \
         repeat it exactly.\n\nDocument:\n{haystack}\n\nQuestion: What is the secret code from \
         the document above?\nAnswer: The secret code is "
    )
}

fn build_narrative_prompt(seed_idx: usize, target_bytes: usize) -> String {
    // No needle; long-form technical/narrative concat from FILLERS rotated by
    // a seed offset so each prompt has a different fingerprint. Includes an
    // instruction wrapper so the model treats it as a comprehension task.
    let mut buf = String::with_capacity(target_bytes + 512);
    let preambles = [
        "Summarise the following extended passage in three sentences without losing key details.",
        "Read the passage and identify the three most important themes.",
        "Read the passage and list any factual claims that depend on outside knowledge.",
        "Identify any structural pattern in the passage's sequence of paragraphs.",
        "Note any contradictions between successive paragraphs in the passage.",
    ];
    buf.push_str(preambles[seed_idx % preambles.len()]);
    buf.push_str("\n\nPassage:\n");
    let start = (seed_idx * 7) % FILLERS.len();
    let mut i = 0;
    while buf.len() < target_bytes {
        buf.push_str(FILLERS[(start + i) % FILLERS.len()]);
        i += 1;
    }
    buf.push_str("\n\nResponse:\n");
    buf
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: gen_long_context_prompts <bonsai_tokenizer.json> <output.json>");
        std::process::exit(2);
    }
    let tk_path = PathBuf::from(&args[1]);
    let out_path = PathBuf::from(&args[2]);

    let tokenizer = Tokenizer::from_file(&tk_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", tk_path.display()))?;

    let mut prompts: Vec<String> = Vec::with_capacity(15);

    // 5 NIAH at ~4k.
    let niah_4k_depths = [0.20_f32, 0.40, 0.55, 0.70, 0.85];
    for (i, depth) in niah_4k_depths.iter().enumerate() {
        prompts.push(build_niah_prompt(NEEDLE_VALUES[i], *depth, TARGET_4K * 4));
    }

    // 5 NIAH at ~8k.
    let niah_8k_depths = [0.15_f32, 0.35, 0.50, 0.65, 0.80];
    for (i, depth) in niah_8k_depths.iter().enumerate() {
        prompts.push(build_niah_prompt(
            NEEDLE_VALUES[5 + i],
            *depth,
            TARGET_8K * 4,
        ));
    }

    // 5 narrative at ~4k.
    for i in 0..5 {
        prompts.push(build_narrative_prompt(i, TARGET_4K * 4));
    }

    // Verify token counts.
    let mut token_counts: Vec<usize> = Vec::with_capacity(prompts.len());
    for (i, p) in prompts.iter().enumerate() {
        let enc = tokenizer
            .encode(p.as_str(), true)
            .map_err(|e| anyhow::anyhow!("tokenize prompt {i}: {e}"))?;
        let n = enc.get_ids().len();
        token_counts.push(n);
        println!("prompt {i:>2}: {n:>5} tokens ({} bytes)", p.len());
    }
    let min_tok = *token_counts.iter().min().unwrap_or(&0);
    let max_tok = *token_counts.iter().max().unwrap_or(&0);
    let mean_tok = token_counts.iter().sum::<usize>() / token_counts.len();
    println!(
        "summary: min={min_tok} max={max_tok} mean={mean_tok} (n={})",
        token_counts.len()
    );
    if min_tok < HARD_FLOOR {
        anyhow::bail!(
            "long-context prompts: min token count {min_tok} < {HARD_FLOOR}; regenerate with larger TARGET_*"
        );
    }

    let description = format!(
        "Long-context calibration prompt set. 15 prompts: 5 NIAH-style \
         needle-in-haystack at ~4k tokens (varied needle depths), 5 NIAH at ~7-8k tokens \
         (varied depths), 5 long-form narrative/technical at ~4k tokens. Measured token \
         counts on Bonsai-8B-2bit (Qwen3 family) tokenizer: min={min_tok}, max={max_tok}, \
         mean={mean_tok}. Replaces the original short-context corpus (200-600 tokens)."
    );

    let value = json!({
        "version": 2,
        "description": description,
        "prompts": prompts,
    });
    let pretty = serde_json::to_string_pretty(&value)?;
    std::fs::write(&out_path, pretty)
        .map_err(|e| anyhow::anyhow!("write {}: {e}", out_path.display()))?;
    println!(
        "wrote {} ({} bytes)",
        out_path.display(),
        std::fs::metadata(&out_path)?.len()
    );
    Ok(())
}
