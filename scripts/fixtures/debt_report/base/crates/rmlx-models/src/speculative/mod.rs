pub fn spec_generate_greedy(prompt_ids: &[u32], n_tokens: usize) -> Vec<u32> {
    let mut out = Vec::new();
    for _ in 0..n_tokens {
        let next = prompt_ids.last().copied().unwrap_or(0) + 1;
        out.push(next);
    }
    out
}
