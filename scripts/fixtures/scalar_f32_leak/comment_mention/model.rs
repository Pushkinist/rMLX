// Synthetic fixture for scripts/check/kernel_dispatch.sh scalar-f32 - not compiled.
/// Builds the factor; never call scalar_f32(x) bare here.
// scalar_f32(x) without a cast promotes the residual stream.
fn f(x: Array) -> Array { x }
