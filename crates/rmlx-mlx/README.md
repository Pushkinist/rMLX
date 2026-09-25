# rmlx-mlx

FFI bindings to the Homebrew-built `mlx-c` C ABI and the safe Rust layer over
them: `Array`, `Device`, `Dtype`, the ops, the Metal kernel and capture
wrappers, and the MLX pin and NAX checks. MLX is linked, never built; its
kernels ship precompiled in `mlx.metallib`.

## Prerequisites

```sh
brew install mlx-c   # pulls mlx
```

Apple Silicon only. `build.rs` fails when either dylib is missing. The
validated MLX / mlx-c pair is declared in [`mlx-pin.txt`](mlx-pin.txt). The
pin test, `rmlx healthcheck` and `make mlx-preflight` check the loaded pair
against it; `build.rs` does not. See
[`docs/FFI.md`](../../docs/FFI.md#pinned-mlx--mlx-c-pair).

## Environment variables

Both are optional. Unset, `build.rs` resolves each prefix with
`brew --prefix <formula>`, then falls back to `/opt/homebrew/opt/<formula>`.
That is the `opt` symlink the dylibs' install names point at.

| Variable | Description |
|---|---|
| `MLX_C_PREFIX` | Root of the mlx-c install (`lib/libmlxc.dylib`, `include/`) |
| `MLX_PREFIX` | Root of the mlx install (`lib/libmlx.dylib`, `include/`) |
