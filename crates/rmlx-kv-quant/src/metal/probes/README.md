# MSL compile probes

Support files for `make check-metal-compiles` (`scripts/check_metal_compiles.sh`).
Nothing here is compiled into the binary or read at runtime — production kernels
come from `../*.metal` via `include_str!`, and production headers are built in
Rust at dispatch.

## Why this exists

`../*.metal` files are kernel **bodies**. MLX generates the function signature
and buffer declarations when the kernel is registered, so a body by itself is a
run of statements at file scope and cannot be compiled standalone. To
syntax-check one, the gate assembles:

```
#include <metal_stdlib> + using namespace metal;
<codec header>
kernel void rmlx_msl_compile_probe(...) {
    <buffer aliases: the names this body expects, at their dispatch dtype>
    <body>
}
```

Buffers are injected as local aliases rather than kernel parameters, so no probe
needs a per-kernel signature or buffer-index bookkeeping.

## Files

| File | Role |
|---|---|
| `kernels.manifest` | Per body: which header to prepend and which buffers to declare. Mirrors the `MetalKernel::new(..)` call sites. |
| `*.hdr.metal` | Snapshots of headers that Rust generates at dispatch (codebooks, rotation constants, quaternions). |

Codecs whose header is already a static file (`../turboquant_header.metal`,
`../turbo_flash_header.metal`, …) are referenced directly from the manifest with
a `../` prefix and need no snapshot here.

## Refreshing a `*.hdr.metal` snapshot

These are captured output, not hand-written. They only need refreshing when a
Rust header builder changes what it emits. A stale snapshot surfaces as a probe
compile failure naming the codec (typically an undeclared constant), so the gate
tells you when it happened.

To refresh, dump the builder's output for the codec in question. Add a temporary
test to that module's sibling `*_tests.rs` (which can reach the private builder):

```rust
#[test]
fn dump_header_tmp() {
    std::fs::write("<path>/probes/<codec>.hdr.metal", build_<codec>_header(<bits>)).unwrap();
}
```

run `cargo test -p rmlx-kv-quant dump_header_tmp`, then remove the temporary
test. The `bits` / parameter value to pass is whichever variant the manifest
line names (e.g. `iso_fused_qk_b3.hdr.metal` ← `build_iso_fused_qk_header(3)`).

Snapshots are representative: the probe checks that the kernel text parses and
resolves, not that the constants are numerically current. Numerical correctness
is covered by the KV parity tests and the real-model smoke.
