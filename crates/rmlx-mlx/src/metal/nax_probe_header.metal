// Header for the MLX-JIT language-version probe. Not a production kernel.
//
// The Metal 4 cooperative-tensor surface, guarded so the kernel still registers
// on a toolchain that compiles below 4.0. `__HAVE_TENSOR__` is defined from
// -std=metal4.0 onwards; below that these includes do not exist and `mpp` is an
// undeclared identifier.
//
// This is a header, not a body, because the includes must land at namespace
// scope — MLX splices the body into the generated kernel function.
#if __HAVE_TENSOR__
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
#endif
