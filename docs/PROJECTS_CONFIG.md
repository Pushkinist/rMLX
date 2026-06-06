# projects.toml — per-project cap defaults

Place `projects.toml` inside `<RMLX_HOME>` (resolved as `$RMLX_HOME`,
`<workspace>/.rmlx/`, or `$HOME/.rmlx/` — same as every other rMLX runtime
file) to set stable per-project SSD and RAM budgets without long CLI lines.
The file is **optional** (absent = silent no-op) and **rMLX-read-only**
(operator or coding agent edits it; rMLX never writes it). Changes take
effect on the next `rmlx serve` restart — there is no live reload.

## File shape

```toml
[global]
ssd_pool_gb = 200.0          # default --kv-ssd-global-gb (cross-namespace ceiling)
ram_prompt_cache_gb = 2.0    # default --prompt-cache-ram-gb

[project.alpha]
ssd_cap_gb = 50.0            # per-namespace SSD cap for --project alpha

[project.beta]
ssd_cap_gb = 30.0
```

## Precedence

```
CLI flag  >  [project.<name>]  >  [global]  >  built-in default
```

Unknown `--project` names silently fall back to `[global]` (no section is
auto-created). Passing `--kv-ssd-cache-gb` or `--kv-ssd-global-gb` on the
CLI always beats the file values. A malformed file is a startup error (exit 2).
