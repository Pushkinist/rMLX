// ParoQuant pairwise Givens rotation — body of `paro_rotate_gpu`.
//
// Ported from `z-lab/paroquant/paroquant/kernels/metal/rotation.metal`.
// `cos_theta` / `sin_theta` arrive pre-computed (upstream's `_cache_rotation`
// pattern) so the kernel does no transcendental math; `params` is the flat
// I32 array `[batch, hidden, krot, group_size]`, same as upstream.
//
// `ROWS_PER_TILE`, `MAX_KROT`, `MAX_GROUP_SIZE` and `InT` are MLX template
// arguments supplied at dispatch (`set_template_int` / `set_template_dtype`).
// They must be compile-time constants: two of them size arrays.
const int batch_size  = params[0];
const int hidden_size = params[1];
const int krot        = params[2];
const int group_size  = params[3];

const int half_gs     = group_size / 2;
const int half_hidden = hidden_size / 2;

const int tile_idx  = threadgroup_position_in_grid.x;
const int group_idx = threadgroup_position_in_grid.y;
const int tid       = thread_index_in_threadgroup;

if (tid >= half_gs)
    return;

float cos_vals[MAX_KROT], sin_vals[MAX_KROT];
int pair_vals[MAX_KROT];

for (int k = 0; k < krot; k++) {
    int idx      = k * half_hidden + group_idx * half_gs + tid;
    cos_vals[k]  = float(cos_theta[idx]);
    sin_vals[k]  = float(sin_theta[idx]);
    pair_vals[k] = int(packed_pairs[idx]);
}

threadgroup float tile[MAX_GROUP_SIZE * ROWS_PER_TILE];

const int ch_lo = group_idx * group_size + tid;
const int ch_hi = ch_lo + half_gs;
float scale_lo  = float(channel_scales[ch_lo]);
float scale_hi  = float(channel_scales[ch_hi]);

for (int r = 0; r < ROWS_PER_TILE; r++) {
    int row = tile_idx * ROWS_PER_TILE + r;
    if (row < batch_size) {
        tile[tid * ROWS_PER_TILE + r]             = float(x[row * hidden_size + ch_lo]) * scale_lo;
        tile[(tid + half_gs) * ROWS_PER_TILE + r] = float(x[row * hidden_size + ch_hi]) * scale_hi;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (int k = 0; k < krot; k++) {
    int i_local = pair_vals[k] & 0xFFFF;
    int j_local = pair_vals[k] >> 16;
    float c = cos_vals[k], s = sin_vals[k];

    for (int m = 0; m < ROWS_PER_TILE; m++) {
        float a                           = tile[i_local * ROWS_PER_TILE + m];
        float b                           = tile[j_local * ROWS_PER_TILE + m];
        tile[i_local * ROWS_PER_TILE + m] = a * c + b * s;
        tile[j_local * ROWS_PER_TILE + m] = b * c - a * s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (int r = 0; r < ROWS_PER_TILE; r++) {
    int row = tile_idx * ROWS_PER_TILE + r;
    if (row < batch_size) {
        out[row * hidden_size + ch_lo] = InT(tile[tid * ROWS_PER_TILE + r]);
        out[row * hidden_size + ch_hi] = InT(tile[(tid + half_gs) * ROWS_PER_TILE + r]);
    }
}
