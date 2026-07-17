# rmlx-mlx

FFI bindings to the brew-prebuilt `mlx-c` C ABI, plus a minimal safe Rust layer
(`Array`, `Device`, `Dtype`, `add`). No vendored source build — requires only
`brew install mlx mlx-c` (kernels ship precompiled in `mlx.metallib`).

## Prerequisites

```sh
brew install mlx mlx-c
```

Apple Silicon only. The validated MLX / mlx-c pair is declared in
[`mlx-pin.txt`](mlx-pin.txt) — that file is the single source of truth, so the
versions are deliberately not repeated here. `build.rs` warns (never fails) when
the resolved stack differs from it, or when the resolved `mlx.metallib` is
missing the fast GEMM kernels. Rationale and the un-pin procedure:
[`docs/FFI.md`](../../docs/FFI.md#pinned-mlx--mlx-c-pair).

## Environment variables

Both are optional overrides. Unset, `build.rs` resolves each prefix via
`brew --prefix <formula>`, falling back to `/opt/homebrew/opt/<formula>` — the
`opt` symlink the dylibs' install names already point at, which is what keeps
the library compiled against and the library loaded on the same file.

| Variable | Description |
|---|---|
| `MLX_C_PREFIX` | Root of the mlx-c install (contains `lib/libmlxc.dylib`, `include/`) |
| `MLX_PREFIX` | Root of the mlx install (contains `lib/libmlx.dylib`, `include/`) |

Set these only for a non-Homebrew layout or a non-default Homebrew prefix.
