
uint s_kv         = threadgroup_position_in_grid.x;
uint bh           = threadgroup_position_in_grid.y;
uint tid          = thread_position_in_threadgroup.x;
uint head_dim     = dims[0];
uint kv_seq       = dims[1];
uint kv_h         = dims[2];
uint heads_per_kv = dims[3];
uint has_mask     = dims[4];

uint n_q_heads = kv_h * heads_per_kv;
uint b         = bh / n_q_heads;
uint hq        = bh % n_q_heads;
uint kv_h_idx  = hq / heads_per_kv;

// Per-token layout: D elements per K row =>
//   groups_per_tok = D / 32
//   words_per_tok  = D * 3 / 32     (== groups_per_tok * 3)
//   scales_per_tok = D / 32         (== groups_per_tok)
uint kv_tok         = (b * kv_h + kv_h_idx) * kv_seq + s_kv;
uint groups_per_tok = head_dim / 32u;
uint words_per_tok  = groups_per_tok * 3u;
uint scales_per_tok = groups_per_tok;
uint codes_tok_off  = kv_tok * words_per_tok;
uint scales_tok_off = kv_tok * scales_per_tok;

// tid handles one head_dim element. Which group and which lane in group?
uint group_id_in_head = tid / 32u;
uint elem_in_group    = tid % 32u;

// Each thread loads one Q element into shared memory.  256-slot SMEM
// covers head_dim in {128, 256}; trailing slots are unused for 128.
threadgroup float q_shared[256];
q_shared[tid] = query[bh * head_dim + tid];
threadgroup_barrier(mem_flags::mem_threadgroup);

// 3-bit unpack: locate this thread's index inside the 96-bit group
// stream.  bits [elem_in_group*3, elem_in_group*3+3) of the concatenated
// LE stream of (codes[group*3], codes[group*3+1], codes[group*3+2]).
uint group_codes_base = codes_tok_off + group_id_in_head * 3u;
uint bit_off          = elem_in_group * 3u;
uint word0_id         = bit_off / 32u; // 0, 1 or 2
uint shift0           = bit_off - word0_id * 32u;
ulong window          = (ulong)codes[group_codes_base + word0_id];
if (word0_id + 1u < 3u) {
    window |= ((ulong)codes[group_codes_base + word0_id + 1u]) << 32;
}
uint idx = (uint)((window >> shift0) & 0x7ul);

// Codebook lookup x per-group scale.  No WHT / Hadamard — turbo3 K
// is plain codebook * scale (matches V3_DEQUANTIZE_SOURCE).
float k_scale = scales[scales_tok_off + group_id_in_head];
float k_val   = CB3[idx] * k_scale;

// QK partial product + threadgroup tree reduction (in-place).
threadgroup float dot_shared[256];
dot_shared[tid] = q_shared[tid] * k_val;
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint stride = head_dim >> 1; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
        dot_shared[tid] += dot_shared[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (tid == 0u) {
    float result = dot_shared[0] * scale_arr[0];
    if (has_mask != 0u) {
        result += mask[bh * kv_seq + s_kv];
    }
    out[bh * kv_seq + s_kv] = result;
}
