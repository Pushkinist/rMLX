// Thread-local Walsh-Hadamard butterfly of size R over registers, in f32.
// Header of `signed_block_hadamard.metal`.
//
// Ported from `radix_func` in MLX's `mlx/backend/metal/kernels/hadamard.h`
// (ml-explore/mlx v0.32.3, MIT). The stages run at strides 1, 2, ..., R / 2.
template <short R>
inline void rmlx_hadamard_radix(thread float *x) {
    constexpr short log_r = __builtin_ctz(R);
    short h               = 1;
#pragma clang loop unroll(full)
    for (short s = 0; s < log_r; s++) {
#pragma clang loop unroll(full)
        for (short i = 0; i < R / 2; i++) {
            short k  = i & (h - 1);
            short j  = ((i - k) << 1) + k;
            float a  = x[j];
            float b  = x[j + h];
            x[j]     = a + b;
            x[j + h] = a - b;
        }
        h <<= 1;
    }
}
