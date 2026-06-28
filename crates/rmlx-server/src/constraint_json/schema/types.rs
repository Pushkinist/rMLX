//! Public types for the A6.4 JSON-Schema constraint engine.
//!
//! `SchemaError`, `EngagePolicy`, `SchemaNode`.

#![allow(unreachable_pub)]

use std::sync::Arc;

use serde_json::Value;

/// Error returned when the supplied `response_format.json_schema.schema`
/// is not a JSON Schema this engine can compile. The handler maps this to
/// HTTP 400 `invalid_request_error`.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SchemaError {
    /// The schema root is not a JSON object.
    #[error("schema must be a JSON object")]
    NotAnObject,
    /// A required keyword has an unexpected type or value.
    #[error("`{0}` must be {1}")]
    BadKeyword(&'static str, &'static str),
    /// `enum` array is present but empty.
    #[error("`enum` must be a non-empty array")]
    EmptyEnum,
    /// `oneOf`/`anyOf` array is present but empty.
    #[error("`oneOf`/`anyOf` must be a non-empty array")]
    EmptyUnion,
    /// `$ref` that cannot be resolved against the document's local
    /// `$defs`/`definitions` (remote `$ref`, or a dangling local pointer).
    #[error("unresolvable `$ref`: `{0}` (only local `#/$defs/...` and `#/definitions/...` are supported)")]
    UnresolvableRef(String),
    /// A keyword this engine cannot enforce was encountered while
    /// `strict == true`. OpenAI strict-mode semantics require every
    /// keyword to be honoured, so we refuse rather than silently ignore.
    /// The handler maps this to HTTP 400 `unsupported_schema_keyword`.
    #[error("schema keyword `{0}` is not supported in strict mode (rMLX cannot enforce it; resubmit with strict=false to accept a degraded constraint)")]
    UnsupportedInStrict(&'static str),
}

impl SchemaError {
    /// `true` when this error is specifically an unsupported-keyword
    /// rejection in strict mode. The HTTP layer maps these to the
    /// `unsupported_schema_keyword` error code (vs the generic
    /// `invalid_request_error` for malformed schemas).
    pub fn is_unsupported_keyword(&self) -> bool {
        matches!(self, SchemaError::UnsupportedInStrict(_))
    }
}

/// Engage policy for the `SchemaConstraint` warm-up phase.
///
/// - `ValueStarter`: engage on the first non-whitespace byte that is a legal
///   value-starter for this schema's root (e.g. `{` for object, `[` for
///   array). This is the A6.3 behaviour and is correct for container roots
///   where the model reliably emits `{`/`[` as its first answer byte.
///
/// - `Immediate`: engage on the very first post-think token, regardless of
///   its bytes. The model may emit an unquoted bare word (e.g. `medium`
///   instead of `"medium"`) for scalar roots; waiting for a `"`-byte that
///   never comes keeps the constraint permanently disengaged. With
///   `Immediate`, the grammar starts masking from the first answer token and
///   forces the correct encoding (quoted string, digit prefix, `true`/`false`,
///   etc.). Used for all scalar roots: `string`, `enum`, `const`, `number`,
///   `integer`, `boolean`, `null`, and discriminated `union`s.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed dispatch enum — two schema-engagement policies (ValueStarter/Immediate); adding a policy requires updating all EngagePolicy match arms in the constraint state machine"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngagePolicy {
    /// Engage on the first non-whitespace byte that is a legal value-starter for the schema root.
    ValueStarter,
    /// Engage immediately on the first post-think token regardless of its bytes.
    Immediate,
}

/// Recursive schema tree mirroring the supported keyword subset.
///
/// Adapted from llama.cpp `SchemaConverter::visit` keyword dispatch order:
/// `$ref` → `oneOf`/`anyOf` → `const` → `enum` → object → array → scalar.
///
/// The immutable sub-schema fields (`props` and each property value-schema,
/// `required`, array `items`, union branches) are held behind `Arc` so the
/// schema tree is *shared*, not deep-copied. The per-token allow-mask probe
/// resets a scratch grammar once per candidate token (~152K per decode step)
/// and, on a value-start byte, enters a property/branch sub-schema; with each
/// sub-schema behind `Arc` both the reset and the value-entry are refcount
/// bumps rather than a recursive clone of the property list / element schema /
/// branch. `Arc` (not `Rc`) because the constraint engine is driven from a
/// `Send` decode loop.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed schema-AST enum — variants are the complete supported JSON-Schema keyword subset; adding a variant requires updating the compiler, SchemaGrammar, and all SchemaNode match arms"
)]
#[derive(Debug, Clone, PartialEq)]
pub enum SchemaNode {
    /// `type:object`. `props` in schema-declared order; `required` is the
    /// subset (in schema order) that must appear; `additional` allows
    /// extra free-form keys when `true`.
    Object {
        /// Ordered list of `(name, schema)` pairs for declared properties.
        /// Each value-schema is pre-wrapped in `Arc` so entering a property
        /// value (in `step_object`) is a refcount bump, not a deep clone.
        props: Arc<[(String, Arc<SchemaNode>)]>,
        /// Names of properties that must appear in the output object.
        required: Arc<[String]>,
        /// When `true`, additional properties beyond `props` are permitted.
        additional: bool,
    },
    /// `type:array` with homogeneous `items`.
    Array {
        /// Schema applied to every element of the array.
        items: Arc<SchemaNode>,
        /// Minimum number of array elements (`minItems`); `None` = unconstrained.
        min: Option<usize>,
        /// Maximum number of array elements (`maxItems`); `None` = unconstrained.
        max: Option<usize>,
    },
    /// `type:string` (optionally restricted by `enum` of strings).
    Str {
        /// Allowed string values from the `enum` keyword; `None` = any string.
        enum_: Option<Vec<String>>,
    },
    /// `type:number` / `type:integer` (`integer` forbids `.`/`e`/`E`).
    Num {
        /// When `true`, only integer values (no decimal point or exponent) are allowed.
        integer: bool,
    },
    /// `type:boolean`.
    Bool,
    /// `type:null`.
    Null,
    /// `const` (or a non-string `enum` member): an exact JSON literal.
    Const(Value),
    /// `oneOf`/`anyOf` union of branches. Each branch is pre-wrapped in
    /// `Arc` so entering a structural branch value is a refcount bump, not a
    /// deep clone.
    Union(Arc<[Arc<SchemaNode>]>),
    /// Unsupported keyword fallback: any JSON value of any type.
    Any,
}
