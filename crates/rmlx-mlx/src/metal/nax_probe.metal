// MLX-JIT language-version probe. Not a production kernel: it reports what
// language version MLX compiled it at, and whether the cooperative-tensor
// include path survived MLX's source wrapping.
//
// out[0] __METAL_VERSION__ as MLX's JIT saw it (400 = Metal 4.0)
// out[1] 1 if __HAVE_TENSOR__ reached the body, else 0
// out[2] .m of a constexpr matmul2d_descriptor, else -1
// out[3] 0 — liveness marker, so "never written" is distinguishable from
//        "written by a pre-4.0 compile"
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
out[3] = 0;
