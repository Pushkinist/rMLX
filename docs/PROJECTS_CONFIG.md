# projects.toml — per-project cap defaults

`rmlx serve` reads `<RMLX_HOME>/projects.toml` at startup to fill in its SSD
and RAM cache caps. `<RMLX_HOME>` resolves as for every other runtime file.
The file is optional: a missing or empty file changes nothing. rMLX never
writes it. An edit takes effect at the next `rmlx serve` start.

## File shape

```toml
[global]
ssd_pool_gb = 200.0          # default --kv-ssd-global-gb
ram_prompt_cache_gb = 2.0    # default --prompt-cache-ram-gb

[project.alpha]
ssd_cap_gb = 50.0            # default --kv-ssd-cache-gb for --project alpha

[project.beta]
ssd_cap_gb = 30.0
```

## Resolution

| Cap | Order | Built-in default |
|---|---|---|
| `--kv-ssd-global-gb` | flag, `[global].ssd_pool_gb`, default | `0.0` (no global ceiling) |
| `--kv-ssd-cache-gb` | flag, `[project.<name>].ssd_cap_gb`, default | `0.0` (SSD tier off) |
| `--prompt-cache-ram-gb` | flag, `[global].ram_prompt_cache_gb`, default | 2 GiB |

- `[global]` has no `ssd_cap_gb`. Without a matching project section, the SSD
  tier stays off unless the flag turns it on.
- A `--project` name with no section uses no project values. No section is
  created.
- The two SSD flags default to `0.0`, so passing `0` counts as not passed and
  the file value applies.
- Unknown keys are ignored.
- A file that is not valid TOML, or has a wrong value type, stops
  `rmlx serve` at startup with `projects.toml: <error>` and exit code 1.

The code is `crates/rmlx-core/src/projects_config.rs`; `serve` calls it from
`crates/rmlx-cli/src/commands/serve.rs`.
