// NOT a codec kernel. This stores OUT OF BOUNDS on purpose.
//
// It is the positive control for the shader-validation gate: an out-of-bounds
// device store is dropped silently by Metal, so the only way to know the gate's
// detector still matches this toolchain's report format is to emit one on
// demand and check that it is caught. Built only under the
// `shader-validation-canary` feature, and its test refuses to dispatch unless
// MTL_SHADER_VALIDATION is on, so the write is always instrumented (and, under
// the gate's FAIL_MODE=zerofill, discarded) rather than corrupting memory.
//
// Keep the store far past the end so no rounding of the allocation can make it
// accidentally land in bounds.
uint i            = thread_position_in_grid.x;
out[i]            = inp[i];
out[i + 1000000u] = 1.0f;
