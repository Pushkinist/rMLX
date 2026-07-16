
// Each threadgroup handles one group of 32 elements.
uint group_id = threadgroup_position_in_grid.x;
uint elem     = thread_position_in_threadgroup.x;

// ── Step 0: cache the 16-entry codebook in threadgroup memory ──────────
// First 16 threads load one centroid each; the rest no-op the load.
threadgroup float cb_shared[16];
if (elem < 16u) {
    cb_shared[elem] = cb[elem];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// ── Step 1: load group into threadgroup shared memory ──────────────────
threadgroup float shared[32];
shared[elem] = inp[group_id * 32u + elem];
threadgroup_barrier(mem_flags::mem_threadgroup);

// ── Step 2: thread 0 finds cb_max + group scale ────────────────────────
// cb_max = max(|cb[i]|) over the 16 entries; scale = max(|x|) / cb_max.
threadgroup float group_scale[1];
if (elem == 0u) {
    float cb_max = 0.0f;
    for (uint i = 0u; i < 16u; i++) {
        float a = abs(cb_shared[i]);
        if (a > cb_max) {
            cb_max = a;
        }
    }
    float abs_max = 0.0f;
    for (uint i = 0u; i < 32u; i++) {
        float a = abs(shared[i]);
        if (a > abs_max) {
            abs_max = a;
        }
    }
    group_scale[0]   = (abs_max > 0.0f && cb_max > 0.0f) ? (abs_max / cb_max) : 0.0f;
    scales[group_id] = group_scale[0];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float scale = group_scale[0];

// ── Step 3: each thread finds nearest centroid via 15 runtime boundaries
float inv_scale  = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
float normalized = shared[elem] * inv_scale;

uint idx = 0u;
for (uint b = 0u; b < 15u; b++) {
    float boundary = (cb_shared[b] + cb_shared[b + 1u]) * 0.5f;
    if (normalized > boundary) {
        idx++;
    }
}

// ── Step 4: pack 8 indices per uint32 word (same layout as hardwired) ──
threadgroup uint idx_shared[32];
idx_shared[elem] = idx;
threadgroup_barrier(mem_flags::mem_threadgroup);

if (elem % 8u == 0u) {
    uint word_idx = group_id * 4u + (elem / 8u);
    uint word     = 0u;
    for (uint i = 0u; i < 8u; i++) {
        word |= (idx_shared[elem + i] & 0xFu) << (i * 4u);
    }
    codes[word_idx] = word;
}
