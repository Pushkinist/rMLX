use super::*;

// ── unit test #1: round-trip TOML deserialization ─────────────────

const FULL_TOML: &str = r"
[global]
ssd_pool_gb = 200.0
ram_prompt_cache_gb = 2.0

[project.alpha]
ssd_cap_gb = 50.0

[project.beta]
ssd_cap_gb = 30.0
";

#[test]
fn deserialize_full_toml() {
    let cfg: ProjectsConfig = toml::from_str(FULL_TOML).expect("parse");
    assert_eq!(cfg.global.ssd_pool_gb, Some(200.0));
    assert_eq!(cfg.global.ram_prompt_cache_gb, Some(2.0));
    let alpha = cfg.project.get("alpha").expect("alpha");
    assert_eq!(alpha.ssd_cap_gb, Some(50.0));
    let beta = cfg.project.get("beta").expect("beta");
    assert_eq!(beta.ssd_cap_gb, Some(30.0));
}

#[test]
fn deserialize_missing_optional_fields() {
    // Only [global] present; no [project.*] sections.
    let toml = "[global]\nssd_pool_gb = 10.0\n";
    let cfg: ProjectsConfig = toml::from_str(toml).expect("parse");
    assert_eq!(cfg.global.ssd_pool_gb, Some(10.0));
    assert_eq!(cfg.global.ram_prompt_cache_gb, None); // optional, absent
    assert!(cfg.project.is_empty());
}

#[test]
fn deserialize_empty_string_is_default() {
    // Simulates the load() path: empty content → Default.
    let cfg: ProjectsConfig = toml::from_str("").expect("parse empty");
    assert_eq!(cfg.global.ssd_pool_gb, None);
    assert_eq!(cfg.global.ram_prompt_cache_gb, None);
    assert!(cfg.project.is_empty());
}

#[test]
fn deserialize_global_only_no_projects() {
    let toml = "[global]\nram_prompt_cache_gb = 4.0\n";
    let cfg: ProjectsConfig = toml::from_str(toml).expect("parse");
    assert_eq!(cfg.global.ram_prompt_cache_gb, Some(4.0));
    assert_eq!(cfg.global.ssd_pool_gb, None);
    assert!(cfg.project.is_empty());
}

#[test]
fn deserialize_project_section_only() {
    let toml = "[project.gamma]\nssd_cap_gb = 75.0\n";
    let cfg: ProjectsConfig = toml::from_str(toml).expect("parse");
    assert_eq!(cfg.global.ssd_pool_gb, None);
    let gamma = cfg.project.get("gamma").expect("gamma");
    assert_eq!(gamma.ssd_cap_gb, Some(75.0));
}

// ── unit test #2: precedence resolution — 6 cases ─────────────────

fn default_config() -> ProjectsConfig {
    toml::from_str(FULL_TOML).unwrap()
}

// Case 1: CLI flag wins over project + global.
#[test]
fn cli_wins_over_project_and_global_ssd_cap() {
    let cfg = default_config();
    let cli = CliCaps {
        ssd_pool_gb: None,
        ssd_cap_gb: Some(99.0),
        ram_prompt_cache_gb: None,
    };
    let resolved = resolve_caps(&cli, &cfg, Some("alpha"));
    assert_eq!(resolved.ssd_cap_gb, 99.0, "CLI must override project");
}

// Case 2: project section wins over global when CLI not set.
#[test]
fn project_wins_over_global_when_cli_absent() {
    let cfg = default_config();
    let cli = CliCaps {
        ssd_pool_gb: None,
        ssd_cap_gb: None, // not set by CLI
        ram_prompt_cache_gb: None,
    };
    let resolved = resolve_caps(&cli, &cfg, Some("alpha"));
    assert_eq!(
        resolved.ssd_cap_gb, 50.0,
        "project.alpha.ssd_cap_gb must apply"
    );
}

// Case 3: global ssd_pool_gb used when CLI not set.
#[test]
fn global_ssd_pool_gb_applies_when_cli_absent() {
    let cfg = default_config();
    let cli = CliCaps {
        ssd_pool_gb: None,
        ssd_cap_gb: None,
        ram_prompt_cache_gb: None,
    };
    let resolved = resolve_caps(&cli, &cfg, Some("alpha"));
    assert_eq!(resolved.ssd_pool_gb, 200.0, "global.ssd_pool_gb must apply");
}

// Case 4: CLI ssd_pool_gb overrides global.
#[test]
fn cli_ssd_pool_gb_overrides_global() {
    let cfg = default_config();
    let cli = CliCaps {
        ssd_pool_gb: Some(500.0),
        ssd_cap_gb: None,
        ram_prompt_cache_gb: None,
    };
    let resolved = resolve_caps(&cli, &cfg, None);
    assert_eq!(resolved.ssd_pool_gb, 500.0);
}

// Case 5: unknown project name falls back to built-in (0.0 ssd_cap).
#[test]
fn unknown_project_uses_builtin_default() {
    let cfg = default_config();
    let cli = CliCaps {
        ssd_pool_gb: None,
        ssd_cap_gb: None,
        ram_prompt_cache_gb: None,
    };
    let resolved = resolve_caps(&cli, &cfg, Some("nonexistent"));
    assert_eq!(
        resolved.ssd_cap_gb, 0.0,
        "unknown project must produce built-in default 0.0"
    );
    // global.ssd_pool_gb still applies
    assert_eq!(resolved.ssd_pool_gb, 200.0);
}

// Case 6: ram_prompt_cache_gb: CLI > global > None.
#[test]
fn ram_cap_precedence() {
    let cfg = default_config(); // global.ram_prompt_cache_gb = Some(2.0)

    // CLI wins
    let cli_set = CliCaps {
        ssd_pool_gb: None,
        ssd_cap_gb: None,
        ram_prompt_cache_gb: Some(8.0),
    };
    let r = resolve_caps(&cli_set, &cfg, None);
    assert_eq!(r.ram_prompt_cache_gb, Some(8.0));

    // No CLI → global
    let cli_absent = CliCaps {
        ssd_pool_gb: None,
        ssd_cap_gb: None,
        ram_prompt_cache_gb: None,
    };
    let r2 = resolve_caps(&cli_absent, &cfg, None);
    assert_eq!(r2.ram_prompt_cache_gb, Some(2.0));

    // No CLI, no global → None
    let empty_cfg = ProjectsConfig::default();
    let r3 = resolve_caps(&cli_absent, &empty_cfg, None);
    assert_eq!(r3.ram_prompt_cache_gb, None);
}

// ── unit test #3: malformed TOML returns Err ──────────────────────

#[test]
fn malformed_toml_returns_err() {
    let bad = "this is not valid toml = = =";
    let result: Result<ProjectsConfig, _> = toml::from_str(bad).map_err(ProjectsConfigError::from);
    assert!(result.is_err(), "malformed TOML must return Err");
}

// ── unit test #4: load_from_path() with missing file returns Default ─

#[test]
fn load_missing_file_returns_default() {
    // Use a temp dir that has no projects.toml to exercise the
    // file-not-found branch of the public load() contract via load_from_path.
    let tmp = tempfile::TempDir::new().unwrap();
    let nonexistent = tmp.path().join("projects.toml");
    let cfg = load_from_path(&nonexistent).expect("missing file must return Ok(default)");
    assert_eq!(cfg.global.ssd_pool_gb, None);
    assert!(cfg.project.is_empty());
}
