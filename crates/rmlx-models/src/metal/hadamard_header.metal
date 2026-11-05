// Thread-local Walsh-Hadamard butterfly of size R over registers, in f32.
// Header of `signed_block_hadamard.metal`.
//
// Ported from `radix_func` in MLX's `mlx/backend/metal/kernels/hadamard.h`
// (ml-explore/mlx v0.32.3, MIT). The stages run at strides 1, 2, ..., R / 2.
//
// MLX's licence (ml-explore/mlx v0.32.3, `LICENSE`), which covers the ported code:
//
// Copyright © 2023 Apple Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
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
