//! A6.4/A6.5 JSON-Schema constraint engine — submodules.
//!
//! Exported surface (unchanged from the former `schema.rs`):
//! - `SchemaError`, `EngagePolicy`, `SchemaNode` — types
//! - `SchemaGrammar`, `is_only_fence_or_whitespace` — grammar
//! - `SchemaConstraint` — `ConstraintEngine` impl

pub(super) mod compiler;
pub(super) mod constraint;
pub(super) mod grammar;
pub(super) mod types;

#[cfg(test)]
mod tests;

pub use constraint::SchemaConstraint;
pub(crate) use grammar::is_only_fence_or_whitespace;
#[cfg(test)]
pub(crate) use grammar::SchemaGrammar;
pub use types::{EngagePolicy, SchemaError, SchemaNode};

// ── Test-visible re-exports ───────────────────────────────────────────────────

#[cfg(test)]
pub(crate) use super::{TokenBytesMap, MAX_INSIGNIFICANT_WS_RUN};
#[cfg(test)]
pub(crate) use rmlx_models::ConstraintEngine;
#[cfg(test)]
pub(crate) use serde_json::Value;
#[cfg(test)]
pub(crate) use std::sync::Arc;
