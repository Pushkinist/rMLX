# MSL compile probes

Support files for `make check-metal-compiles` (`scripts/check_metal_compiles.sh`).
Nothing here is compiled into the binary or read at runtime — production kernels
come from `../*.metal` via `include_str!`, and production headers are built in
Rust at dispatch.

This is the reference copy of the convention. `crates/rmlx-models/src/metal/probes/`
and `crates/rmlx-mlx/src/metal/probes/` carry their own manifests in the same
format and point back here.

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
    <#defines: the dispatch-time values that are neither buffers nor header constants>
    <body>
}
```

Buffers are injected as local aliases rather than kernel parameters, so no probe
needs a per-kernel signature or buffer-index bookkeeping. The `#define`s sit
immediately ahead of the body so a common name (`T`) cannot collide with the
header or the probe's own signature.

Each body is compiled twice, at `-std=metal3.0` and `-std=metal4.0`. The second
pass is what makes a `#if __HAVE_TENSOR__` body checkable: below Metal 4.0 that
macro is undefined and such a body compiles to an empty translation unit. On a
toolchain that cannot do the second pass, those bodies are reported as `SKIP` and
counted rather than compiled without the guard; CI runs `--strict`, which refuses
the reduced gate outright.

Buffer types are `u` (uint), `i` (int) and `f` (float) — use the one the dispatch
site declares.

## Files

| File | Role |
|---|---|
| `kernels.manifest` | Per body: which header to prepend, which buffers to declare, and (optional 4th field) which `#define`s to emit. Mirrors the `MetalKernel::new(..)` / `set_template_*` call sites. |
| `*.hdr.metal` | Snapshots of headers that Rust generates at dispatch (codebooks, rotation constants, quaternions). |

Codecs whose header is already a static file (`../turboquant_header.metal`,
`../turbo_flash_header.metal`, …) are referenced directly from the manifest with
a `../` prefix and need no snapshot here.

Every `.metal` file in `../` must be named by the manifest, as a body or as a
`../`-prefixed header. The gate hard-fails otherwise: a body nothing lists is a
body nothing compiles.

## Snapshots are pinned to their builders

These are captured output, not hand-written, so they can drift from the Rust
builder that produced them. Every one is pinned by an equality test in the
owning module's sibling `*_tests.rs` (`hdr_probe_snapshot*_match*_builder*`):

```rust
assert_eq!(kernel_header(), include_str!("metal/probes/rot_k.hdr.metal"));
```

A builder that changes what it emits then fails `cargo test -p rmlx-kv-quant`
with `stale snapshot: refresh <file>`. That test is what keeps a snapshot
honest — the compile gate alone would only notice a constant being **added or
renamed**. A changed *value*, or a *removed* constant, still compiles, so
without the equality test the gate would keep passing while validating text the
builders no longer emit.

For the same reason the whitespace pre-commit hooks skip `*.hdr.metal`
(see `.pre-commit-config.yaml`): trimming a captured trailing space would edit
the capture out from under its test.

## Refreshing a snapshot

When a builder legitimately changes, refresh the capture by temporarily turning
its guard into a writer, in the module's sibling `*_tests.rs`:

```rust
#[allow(clippy::unwrap_used, reason = "temporary snapshot refresh; removed before commit")]
#[test]
fn refresh_snapshot_tmp() {
    std::fs::write(
        concat!(env!("CARGO_MANIFEST_DIR"), "/src/metal/probes/rot_k.hdr.metal"),
        kernel_header(),
    )
    .unwrap();
}
```

Run `cargo test -p rmlx-kv-quant refresh_snapshot_tmp`, remove the temporary
test, then re-run the guard to confirm it passes. `unwrap_used` is denied
workspace-wide (tests included), hence the `#[allow]`. The parameter to pass is
whichever variant the manifest line names — e.g. `rotor_fused_qk_b3.hdr.metal`
← `build_rotor_fused_qk_header(3)`.

Snapshots are representative: the probe checks that the kernel text parses and
resolves, not that the constants are numerically current. Numerical correctness
is covered by the KV parity tests and the real-model smoke.
