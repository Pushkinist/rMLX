// MLX-JIT language-version probe. Not a production kernel: it reports what
// language version MLX compiled it at, and whether the cooperative-tensor
// include path survived MLX's source wrapping.
//
// out[0] __METAL_VERSION__ as MLX's JIT saw it (400 = Metal 4.0)
// out[1] 1 if __HAVE_TENSOR__ reached the body, else 0
// out[2] .m of a constexpr matmul2d_descriptor, else -1
// out[3] 0x5A5A — liveness sentinel. It has to be a value the buffer cannot
//        hold by accident: MLX hands out pooled buffers and this dispatch sets
//        no init value, so a slot the kernel never wrote can read as anything,
//        and 0 in particular is what a fresh allocation reads as.
out[0] = __METAL_VERSION__;
#if __HAVE_TENSOR__
out[1] = 1;
constexpr matmul2d_descriptor desc(8, 32, 128, false, true, false,
                                   matmul2d_descriptor::mode::multiply);
out[2] = (int)desc.m;
#else
out[1] = 0;
out[2] = -1;
#endif
out[3] = 0x5A5A;
