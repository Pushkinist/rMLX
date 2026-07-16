
// ── Thread / threadgroup coordinates ──────────────────────────────────
uint s_kv         = threadgroup_position_in_grid.x;
uint bh           = threadgroup_position_in_grid.y; // b * n_q_heads + hq
uint tid          = thread_position_in_threadgroup.x;
uint head_dim     = dims[0];
uint kv_seq       = dims[1];
uint kv_h         = dims[2];
uint heads_per_kv = dims[3];

// (b, hq) → (b, kv_h_idx)
uint n_q_heads = kv_h * heads_per_kv;
uint b         = bh / n_q_heads;
uint hq        = bh % n_q_heads;
uint kv_h_idx  = hq / heads_per_kv;

// Token-base offsets into K's per-(b, kv_h_idx, s_kv) record.
// K packing is SEQUENCE-major (`[B, S, kv_h, D]` element order): per token
// all heads are contiguous. This matches QuantPlanarK's storage, whose
// chunk appends land at a per-token (all-heads) offset; a head-major base
// would scramble heads↔seq after a multi-token append at prev_seq > 0.
uint kv_tok = (b * kv_seq + s_kv) * kv_h + kv_h_idx;
// Codes layout (axis-agnostic across 3-bit / 4-bit): every group of 32
// elements occupies exactly 4 u32 words (4-bit packs 8 vals/word;
// 3-bit packs 10 vals/word with 2 wasted bits — but still 4 words/group).
uint codes_words_per_tok  = (head_dim / 32u) * 4u;
uint scales_pairs_per_tok = head_dim / 2u;
uint rot_words_per_tok    = head_dim / 16u;
uint codes_tok_off        = kv_tok * codes_words_per_tok;
uint scales_tok_off       = kv_tok * scales_pairs_per_tok;
uint rot_tok_off          = kv_tok * rot_words_per_tok;

// Group bookkeeping (32 elements per group; 16 pairs per group).
uint group_id_in_head = tid / 32u;
uint elem_in_group    = tid % 32u;
uint pair_in_group    = elem_in_group / 2u;
uint elem_in_pair     = elem_in_group & 1u;

// ── Load Q into threadgroup memory (so all threads can dot with K[d]) ─
// Q layout: [B, n_q_heads, head_dim], flat.
threadgroup float q_shared[1024]; // upper bound — head_dim ≤ 1024 in supported archs.
q_shared[tid] = query[bh * head_dim + tid];
threadgroup_barrier(mem_flags::mem_threadgroup);

// ── Decode K[tid] for this (b, kv_h_idx, s_kv) ────────────────────────
// Codes (bits = 3): word_in_group = elem_in_group / vals_per_word.
uint code_word_in_group = elem_in_group / 10u;
uint code_word_abs      = codes_tok_off + group_id_in_head * 4u + code_word_in_group;
uint cb_idx             = (codes[code_word_abs] >> ((elem_in_group % 10u) * 3u)) & 0x7u;

// Per-pair scale.
uint pair_global = group_id_in_head * 16u + pair_in_group; // pair index within head
float scale_pair = scales[scales_tok_off + pair_global];

// Per-pair Givens rotation index (4-bit, 8 pairs per u32 word, 2 words per group).
uint rot_word_in_group = pair_in_group / 8u;
uint rot_word_abs      = rot_tok_off + group_id_in_head * 2u + rot_word_in_group;
uint rot_shift         = (pair_in_group & 7u) * 4u;
uint rot_idx           = (rot32[rot_word_abs] >> rot_shift) & 0xFu;

// The thread for the partner element (other half of the pair) needs to
// share its raw centroid+scale, then both threads multiply by the right
// row of the rotation transpose (the inverse Givens rotation).
//
// We materialise *both* y_a and y_b for the pair via threadgroup memory
// (one slot per element of the head, reused after each kernel-internal
// phase).
threadgroup float k_pre_rot[1024];
k_pre_rot[tid] = QK_CB[cb_idx] * scale_pair;
threadgroup_barrier(mem_flags::mem_threadgroup);

// Apply R^T to (y_a, y_b) where entry = [c, -s, s, c]:
//   R^T = [[c, s], [-s, c]]
//   a = c*y_a + s*y_b
//   b = -s*y_a + c*y_b
// The pair starts at tid - elem_in_pair (= even thread of the pair).
uint pair_base = tid - elem_in_pair;
float ya       = k_pre_rot[pair_base];
float yb       = k_pre_rot[pair_base + 1u];

float c     = QK_ROT_CB[rot_idx][0];
float neg_s = QK_ROT_CB[rot_idx][1];
float sv    = QK_ROT_CB[rot_idx][2];
float c2    = QK_ROT_CB[rot_idx][3];

// Pre-rotation element value (in registers — never written back to HBM).
float k_val = (elem_in_pair == 0u)
                  ? (c * ya + sv * yb)      // even slot: row 0 of R^T
                  : (neg_s * ya + c2 * yb); // odd slot:  row 1 of R^T

// ── Per-thread partial product + tree reduction ───────────────────────
threadgroup float dot_shared[1024];
dot_shared[tid] = q_shared[tid] * k_val;
threadgroup_barrier(mem_flags::mem_threadgroup);

// Tree reduce: REQUIRES head_dim to be a power of two.  The Rust
// dispatcher (`planar_fused_qk`) returns Err for non-pow-2 head_dim
// (e.g. head_dim=80), so the SDPA caller falls back to the legacy
// dequant+SDPA path.  Do not relax this without revising the reduction.
for (uint stride = head_dim >> 1; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
        dot_shared[tid] += dot_shared[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (tid == 0u) {
    out[bh * kv_seq + s_kv] = dot_shared[0] * scale_arr[0];
}
