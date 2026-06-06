//! E2E grid report writer.
//!
//! Emits the PASS/FAIL grid (feature × sub-feature) to
//! `<RMLX_HOME>/e2e/report.json` and `report.md`. The JSON is the
//! machine-readable source of truth; the Markdown is the human grid.
//!
//! Kept dependency-free beyond `serde_json` (already a workspace dep) — the
//! report is assembled by hand so the harness adds no new crate.

// Items are used per-binary; allow the standard tests/common lints.
#![allow(dead_code, unreachable_pub, clippy::format_push_string)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Outcome of one case.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verdict {
    Pass,
    Fail,
    /// Known/expected failure — a real, documented product gap (e.g. a stop
    /// sequence that does not truncate). Recorded in the grid as a finding but
    /// does NOT trip `any_failed()`, so the suite stays green for known issues.
    XFail,
    Skip,
    Pending,
}

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Pass => "PASS",
            Verdict::Fail => "FAIL",
            Verdict::XFail => "XFAIL",
            Verdict::Skip => "SKIP",
            Verdict::Pending => "PENDING",
        }
    }

    /// Grid glyph for the Markdown table.
    fn glyph(self) -> &'static str {
        match self {
            Verdict::Pass => "PASS",
            Verdict::Fail => "**FAIL**",
            Verdict::XFail => "_xfail (finding)_",
            Verdict::Skip => "skip",
            Verdict::Pending => "_pending_",
        }
    }
}

/// One recorded case result.
#[derive(Clone, Debug)]
pub struct CaseResult {
    pub id: String,
    pub feature: String,
    pub subfeature: String,
    pub verdict: Verdict,
    /// One-line detail (cosine value, golden match, why-skip, why-fail).
    pub detail: String,
    pub tags: Vec<String>,
}

/// Tallies by verdict.
#[derive(Default, Clone, Copy)]
struct Counts {
    pass: usize,
    fail: usize,
    xfail: usize,
    skip: usize,
    pending: usize,
}

/// Accumulates case results and writes the grid.
#[derive(Default)]
pub struct Report {
    results: Vec<CaseResult>,
}

impl Report {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, r: CaseResult) {
        self.results.push(r);
    }

    pub fn results(&self) -> &[CaseResult] {
        &self.results
    }

    fn counts(&self) -> Counts {
        let mut c = Counts::default();
        for r in &self.results {
            match r.verdict {
                Verdict::Pass => c.pass += 1,
                Verdict::Fail => c.fail += 1,
                Verdict::XFail => c.xfail += 1,
                Verdict::Skip => c.skip += 1,
                Verdict::Pending => c.pending += 1,
            }
        }
        c
    }

    /// `true` when at least one case FAILed — the entry point fails the test.
    pub fn any_failed(&self) -> bool {
        self.results.iter().any(|r| r.verdict == Verdict::Fail)
    }

    /// Write `report.json` + `report.md` under `<dir>/`. Returns the dir.
    pub fn write(&self, dir: &Path) -> std::io::Result<PathBuf> {
        std::fs::create_dir_all(dir)?;
        let json_path = dir.join("report.json");
        let md_path = dir.join("report.md");
        std::fs::write(&json_path, self.to_json())?;
        std::fs::write(&md_path, self.to_markdown())?;
        Ok(dir.to_path_buf())
    }

    fn to_json(&self) -> String {
        let c = self.counts();
        let cases: Vec<serde_json::Value> = self
            .results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.id,
                    "feature": r.feature,
                    "subfeature": r.subfeature,
                    "verdict": r.verdict.as_str(),
                    "detail": r.detail,
                    "tags": r.tags,
                })
            })
            .collect();
        let doc = serde_json::json!({
            "schema": "rmlx-e2e-report-v1",
            "summary": { "pass": c.pass, "fail": c.fail, "xfail": c.xfail,
                         "skip": c.skip, "pending": c.pending,
                         "total": self.results.len() },
            "cases": cases,
        });
        serde_json::to_string_pretty(&doc).unwrap_or_else(|_| "{}".to_owned())
    }

    fn to_markdown(&self) -> String {
        let c = self.counts();
        let mut out = String::new();
        out.push_str("# rMLX E2E Feature-Proof Grid\n\n");
        out.push_str(&format!(
            "Summary: **{} PASS**, **{} FAIL**, {} xfail (findings), {} skip, {} pending \
             (total {}).\n\n",
            c.pass,
            c.fail,
            c.xfail,
            c.skip,
            c.pending,
            self.results.len()
        ));

        // Group by feature, preserving manifest order within a feature.
        let mut by_feature: BTreeMap<&str, Vec<&CaseResult>> = BTreeMap::new();
        for r in &self.results {
            by_feature.entry(&r.feature).or_default().push(r);
        }

        out.push_str("| Feature | Sub-feature | Verdict | Detail |\n");
        out.push_str("|---|---|---|---|\n");
        for (feature, rows) in &by_feature {
            for r in rows {
                let detail = r.detail.replace('|', "\\|").replace('\n', " ");
                out.push_str(&format!(
                    "| {feature} | {} | {} | {detail} |\n",
                    r.subfeature,
                    r.verdict.glyph(),
                ));
            }
        }
        out.push('\n');
        out
    }
}
