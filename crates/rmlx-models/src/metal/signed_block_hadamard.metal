// Signed block Walsh-Hadamard transform — body of `signed_block_hadamard_gpu`.
//
// One threadgroup per block of N values along the last axis. Forward
// (INVERSE == 0) stores H (s * x); inverse stores s * (H x). The sign
// multiply and the butterfly run in f32 in registers and threadgroup memory;
// the store rounds once to T.
//
// Ported from `hadamard_n` in MLX's `mlx/backend/metal/kernels/hadamard.h`
// (ml-explore/mlx v0.32.3, MIT), its `stride == 1` branch, with MLX's host
// choice of radix 16 and read width 4 (`hadamard_mn_contiguous`). Changes:
// the buffer is float for every T, the signs are applied at the load or the
// store, and the block index comes from the grid.
//
// T, N, WIDTH and INVERSE are MLX template arguments; scale is a 0-D f32
// input, which MLX declares as `constant float&`.
constexpr int MAX_RADIX   = N < 16 ? N : 16;
constexpr int READ_WIDTH  = MAX_RADIX < 4 ? MAX_RADIX : 4;
constexpr int NUM_THREADS = N / MAX_RADIX;
constexpr int LOG_N       = __builtin_ctz(N);
constexpr int LOG_R       = __builtin_ctz(MAX_RADIX);
constexpr int NUM_STEPS   = LOG_N / LOG_R;
constexpr int LOG_FINAL   = LOG_N % LOG_R;
constexpr int FINAL_RADIX = 1 << LOG_FINAL;

const int i          = int(thread_position_in_threadgroup.x);
const uint block_idx = threadgroup_position_in_grid.y;
const uint base      = block_idx * N;
const uint sign_base = (block_idx % (WIDTH / N)) * N;

threadgroup float buf[N];

#pragma clang loop unroll(full)
for (int j = 0; j < MAX_RADIX / READ_WIDTH; j++) {
    const int index = j * READ_WIDTH * NUM_THREADS + i * READ_WIDTH;
#pragma clang loop unroll(full)
    for (int r = 0; r < READ_WIDTH; r++) {
        const float v  = float(x[base + index + r]);
        buf[index + r] = INVERSE ? v : v * signs[sign_base + index + r];
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float reg[MAX_RADIX];
int h = 1;
#pragma clang loop unroll(full)
for (int s = 0; s < NUM_STEPS; s++) {
    const int k = i & (h - 1);
    const int j = ((i - k) << LOG_R) + k;
#pragma clang loop unroll(full)
    for (int r = 0; r < MAX_RADIX; r++) {
        reg[r] = buf[j + h * r];
    }
    rmlx_hadamard_radix<MAX_RADIX>(reg);
#pragma clang loop unroll(full)
    for (int r = 0; r < MAX_RADIX; r++) {
        buf[j + h * r] = reg[r];
    }
    h <<= LOG_R;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

if (FINAL_RADIX > 1) {
#pragma clang loop unroll(full)
    for (int t = 0; t < MAX_RADIX / FINAL_RADIX; t++) {
        const int index = i + t * NUM_THREADS;
        const int k     = index & (h - 1);
        const int j     = ((index - k) << LOG_FINAL) + k;
#pragma clang loop unroll(full)
        for (int r = 0; r < FINAL_RADIX; r++) {
            reg[r] = buf[j + h * r];
        }
        rmlx_hadamard_radix<FINAL_RADIX>(reg);
#pragma clang loop unroll(full)
        for (int r = 0; r < FINAL_RADIX; r++) {
            buf[j + h * r] = reg[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

#pragma clang loop unroll(full)
for (int j = 0; j < MAX_RADIX / READ_WIDTH; j++) {
    const int index = j * READ_WIDTH * NUM_THREADS + i * READ_WIDTH;
#pragma clang loop unroll(full)
    for (int r = 0; r < READ_WIDTH; r++) {
        const float v         = buf[index + r] * scale;
        out[base + index + r] = T(INVERSE ? v * signs[sign_base + index + r] : v);
    }
}
