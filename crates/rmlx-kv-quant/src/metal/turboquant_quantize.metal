
// Each threadgroup handles one group of 32 elements.
// threadgroup_position_in_grid.x = group index (0 .. N_groups-1).
// thread_position_in_threadgroup.x = element within group (0 .. 31).
uint group_id = threadgroup_position_in_grid.x;
uint elem     = thread_position_in_threadgroup.x;

// ── Step 1: load group into threadgroup shared memory ──────────────────
threadgroup float shared[32];
shared[elem] = inp[group_id * 32u + elem];
threadgroup_barrier(mem_flags::mem_threadgroup);

// ── Step 2: thread 0 finds max(|x|) and writes scale ──────────────────
// Sequential scan by thread 0 mirrors the CPU path exactly.
// scale = max(|x_i|) / CB_MAX where CB_MAX = 2.7176671 (Lloyd-Max N(0,1) 4-bit).
threadgroup float group_scale[1];
if (elem == 0u) {
    float abs_max = 0.0f;
    for (uint i = 0u; i < 32u; i++) {
        float a = abs(shared[i]);
        if (a > abs_max) {
            abs_max = a;
        }
    }
    // Lloyd-Max N(0,1) 4-bit max centroid 2.7176671 (replaces prior 2.7326 quantile centroid).
    const float cb_max = 2.7176671f;
    group_scale[0]     = (abs_max > 0.0f) ? (abs_max / cb_max) : 0.0f;
    scales[group_id]   = group_scale[0];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float scale = group_scale[0];

// ── Step 3: each thread finds nearest centroid index ───────────────────
float inv_scale  = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
float normalized = shared[elem] * inv_scale;

uint idx = 0u;
for (uint b = 0u; b < 15u; b++) {
    if (normalized > BOUNDARIES[b]) {
        idx++;
    }
}

// ── Step 4: pack 8 indices per uint32 word ─────────────────────────────
// word 0 holds elements 0..7, word 1 holds elements 8..15, etc.
// Within each word: element e occupies bits [e_in_word*4 .. e_in_word*4+3].
threadgroup uint idx_shared[32];
idx_shared[elem] = idx;
threadgroup_barrier(mem_flags::mem_threadgroup);

// Threads 0, 8, 16, 24 each pack one 32-bit word (8 elements × 4 bits).
if (elem % 8u == 0u) {
    uint word_idx = group_id * 4u + (elem / 8u);
    uint word     = 0u;
    for (uint i = 0u; i < 8u; i++) {
        word |= (idx_shared[elem + i] & 0xFu) << (i * 4u);
    }
    codes[word_idx] = word;
}
