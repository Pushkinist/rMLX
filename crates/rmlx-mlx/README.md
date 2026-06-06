# rmlx-mlx

FFI bindings to the brew-prebuilt `mlx-c` C ABI, plus a minimal safe Rust layer
(`Array`, `Device`, `Dtype`, `add`). No vendored source build — requires only
`brew install mlx mlx-c` (kernels ship precompiled in `mlx.metallib`).

## Prerequisites

```sh
brew install mlx mlx-c
```

Tested against `mlx-c 0.6.0_2` and `mlx 0.31.2`. Apple Silicon only.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MLX_C_PREFIX` | `/opt/homebrew/Cellar/mlx-c/0.6.0_2` | Root of the mlx-c cellar entry |
| `MLX_PREFIX` | `/opt/homebrew/Cellar/mlx/0.31.2` | Root of the mlx cellar entry |

Set these if you have a non-default Homebrew prefix or a different version pinned.
