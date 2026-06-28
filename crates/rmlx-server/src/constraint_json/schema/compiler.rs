//! JSON-Schema compiler: `SchemaNode::parse` + `parse_node` recursive walker.
//!
//! Keyword dispatch order mirrors llama.cpp `json-schema-to-grammar.cpp:844-960`
//! (`visit`): `$ref` → `allOf` → `not`/`if` → `oneOf`/`anyOf` → `const` →
//! `enum` → scalar-narrowing keywords → `type`.

#![allow(
    clippy::cognitive_complexity,
    clippy::items_after_statements,
    clippy::needless_pass_by_value,
    clippy::self_only_used_in_recursion,
    clippy::unused_self
)]

use std::sync::Arc;

use serde_json::Value;

use super::types::{SchemaError, SchemaNode};

/// One-shot dedup for "ignored unsupported keyword" warnings so a big
/// schema does not spam the log.
pub(super) fn warn_unsupported(keyword: &str) {
    use std::sync::OnceLock;
    static SEEN: OnceLock<parking_lot::Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    let set = SEEN.get_or_init(|| parking_lot::Mutex::new(std::collections::HashSet::new()));
    let mut g = set.lock();
    if g.insert(keyword.to_string()) {
        tracing::warn!(
            keyword,
            "A6.4: unsupported JSON Schema keyword — degrading that node to `Any`"
        );
    }
}

impl SchemaNode {
    /// Returns `true` when this schema root is a scalar value (not a
    /// container). Scalar roots use [`super::types::EngagePolicy::Immediate`] — we do not
    /// wait for a `{`/`[`-byte that will never come.
    pub fn is_scalar_root(&self) -> bool {
        match self {
            SchemaNode::Str { .. }
            | SchemaNode::Num { .. }
            | SchemaNode::Bool
            | SchemaNode::Null
            | SchemaNode::Const(_) => true,
            // A union is scalar when every branch is scalar (e.g. all-const
            // or all-string-enum). If any branch is a container, treat as
            // non-scalar (ValueStarter is safe for containers).
            SchemaNode::Union(branches) => branches.iter().all(|b| b.is_scalar_root()),
            SchemaNode::Object { .. } | SchemaNode::Array { .. } | SchemaNode::Any => false,
        }
    }

    /// Parse a `serde_json::Value` schema into a [`SchemaNode`] tree.
    ///
    /// `strict` applies OpenAI structured-output tightening: all
    /// `properties` become required and `additionalProperties` defaults to
    /// `false`. In strict mode any keyword this engine cannot *enforce*
    /// (see the gap table in the module docs) returns
    /// [`SchemaError::UnsupportedInStrict`] rather than silently degrading
    /// — OpenAI strict semantics require every keyword to be honoured.
    ///
    /// Keyword dispatch order matches llama.cpp
    /// `json-schema-to-grammar.cpp:844-960` (`visit`).
    pub fn parse(schema: &Value, strict: bool) -> Result<SchemaNode, SchemaError> {
        // Extract the document-level `$defs` / `definitions` map once at the
        // root so any nested `$ref` can resolve against it. OpenAI structured
        // outputs nest reusable sub-schemas under `$defs`; some producers
        // still use the Draft-7 `definitions` spelling. Both are checked.
        let defs = schema
            .as_object()
            .and_then(|o| o.get("$defs").or_else(|| o.get("definitions")))
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        Self::parse_node(schema, strict, &defs, 0)
    }

    /// Recursive node parse with a document-level `$defs`/`definitions`
    /// context for local `$ref` resolution. `depth` bounds recursion so a
    /// self-referential `$ref` cannot blow the stack — at the cap the node
    /// degrades to `Any` (non-strict) or errors (strict).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
    )]
    fn parse_node(
        schema: &Value,
        strict: bool,
        defs: &serde_json::Map<String, Value>,
        depth: usize,
    ) -> Result<SchemaNode, SchemaError> {
        // `degrade(kw)`: strict ⇒ hard 400; non-strict ⇒ warn + `Any`.
        // Centralises the strict-vs-permissive policy for every keyword we
        // cannot enforce.
        let degrade = |kw: &'static str| -> Result<SchemaNode, SchemaError> {
            if strict {
                Err(SchemaError::UnsupportedInStrict(kw))
            } else {
                warn_unsupported(kw);
                Ok(SchemaNode::Any)
            }
        };

        let obj = schema.as_object().ok_or(SchemaError::NotAnObject)?;

        // Recursion guard: a recursive `$ref` (e.g. a tree node referencing
        // itself) would otherwise expand forever. Cap depth and degrade.
        const MAX_DEPTH: usize = 64;
        if depth >= MAX_DEPTH {
            return degrade("$ref:recursion-depth");
        }

        // `$ref` → resolve LOCAL `#/$defs/<name>` and `#/definitions/<name>`
        // pointers against the document's defs map. Remote refs (anything
        // not starting `#/$defs/` or `#/definitions/`) and dangling local
        // pointers are unresolvable → 400 (this is a malformed/unsupported
        // request regardless of strict, since we cannot honour the schema).
        if let Some(r) = obj.get("$ref") {
            let pointer = r
                .as_str()
                .ok_or(SchemaError::BadKeyword("$ref", "a string"))?;
            let name = pointer
                .strip_prefix("#/$defs/")
                .or_else(|| pointer.strip_prefix("#/definitions/"));
            return match name.and_then(|n| defs.get(n)) {
                Some(target) => Self::parse_node(target, strict, defs, depth + 1),
                None => Err(SchemaError::UnresolvableRef(pointer.to_string())),
            };
        }

        // `allOf` (intersection): a single-branch `allOf` is equivalent to
        // that branch (common OpenAI pattern: `allOf:[{$ref:...}]`), so we
        // resolve it directly. A genuine multi-branch intersection is not
        // modelled → degrade.
        if let Some(all) = obj.get("allOf") {
            let arr = all
                .as_array()
                .ok_or(SchemaError::BadKeyword("allOf", "an array"))?;
            if arr.len() == 1 {
                return Self::parse_node(&arr[0], strict, defs, depth + 1);
            }
            return degrade("allOf");
        }

        // `not` / `if`/`then`/`else` / `unevaluatedProperties` are not
        // modelled → degrade.
        for kw in ["not", "if", "unevaluatedProperties"] {
            if obj.contains_key(kw) {
                return degrade(match kw {
                    "not" => "not",
                    "if" => "if/then/else",
                    _ => "unevaluatedProperties",
                });
            }
        }

        // `oneOf` / `anyOf` → union.
        if let Some(alts) = obj.get("oneOf").or_else(|| obj.get("anyOf")) {
            let arr = alts.as_array().ok_or(SchemaError::EmptyUnion)?;
            if arr.is_empty() {
                return Err(SchemaError::EmptyUnion);
            }
            let mut branches: Vec<Arc<SchemaNode>> = Vec::with_capacity(arr.len());
            for a in arr {
                branches.push(Arc::new(Self::parse_node(a, strict, defs, depth + 1)?));
            }
            return Ok(SchemaNode::Union(Arc::from(branches)));
        }

        // `const`.
        if let Some(c) = obj.get("const") {
            return Ok(SchemaNode::Const(c.clone()));
        }

        // `enum`: all-string → Str{enum}, else → Union of Const.
        if let Some(e) = obj.get("enum") {
            let arr = e.as_array().ok_or(SchemaError::EmptyEnum)?;
            if arr.is_empty() {
                return Err(SchemaError::EmptyEnum);
            }
            if arr.iter().all(Value::is_string) {
                // all() guard above guarantees every element is a string.
                let lits: Vec<String> = arr
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect();
                return Ok(SchemaNode::Str { enum_: Some(lits) });
            }
            return Ok(SchemaNode::Union(Arc::from(
                arr.iter()
                    .map(|v| Arc::new(SchemaNode::Const(v.clone())))
                    .collect::<Vec<_>>(),
            )));
        }

        // unsupported scalar-narrowing keywords. In strict mode these are a
        // hard 400 (we cannot enforce the narrowing); non-strict warns and
        // keeps the bare `type`.
        for kw in [
            "pattern",
            "format",
            "minimum",
            "maximum",
            "minLength",
            "maxLength",
        ] {
            if obj.contains_key(kw) {
                if strict {
                    return Err(SchemaError::UnsupportedInStrict(match kw {
                        "pattern" => "pattern",
                        "format" => "format",
                        "minimum" => "minimum",
                        "maximum" => "maximum",
                        "minLength" => "minLength",
                        _ => "maxLength",
                    }));
                }
                warn_unsupported(kw);
            }
        }

        let ty = obj.get("type").and_then(|t| t.as_str());

        // Tuple `items` (array form) / `prefixItems` unsupported → strict
        // 400, non-strict degrades the array node to Any-items.
        if obj.get("items").is_some_and(Value::is_array) || obj.contains_key("prefixItems") {
            if strict {
                return Err(SchemaError::UnsupportedInStrict("prefixItems/tuple-items"));
            }
            warn_unsupported("prefixItems/tuple-items");
        }

        match ty {
            Some("object") => Self::parse_object(obj, strict, defs, depth),
            Some("array") => Self::parse_array(obj, strict, defs, depth),
            Some("string") => Ok(SchemaNode::Str { enum_: None }),
            Some("integer") => Ok(SchemaNode::Num { integer: true }),
            Some("number") => Ok(SchemaNode::Num { integer: false }),
            Some("boolean") => Ok(SchemaNode::Bool),
            Some("null") => Ok(SchemaNode::Null),
            Some(_) => degrade("type:<unknown>"),
            None => {
                // No `type`: infer object if `properties` present (mirrors
                // llama.cpp `schema_type.is_null() || == "object"`), else
                // Any. A bare `{}` / no-constraint schema is `Any` even in
                // strict mode (it is fully honoured: it constrains nothing).
                if obj.contains_key("properties") {
                    Self::parse_object(obj, strict, defs, depth)
                } else if obj.contains_key("items") {
                    Self::parse_array(obj, strict, defs, depth)
                } else {
                    Ok(SchemaNode::Any)
                }
            }
        }
    }

    fn parse_object(
        obj: &serde_json::Map<String, Value>,
        strict: bool,
        defs: &serde_json::Map<String, Value>,
        depth: usize,
    ) -> Result<SchemaNode, SchemaError> {
        let mut props: Vec<(String, Arc<SchemaNode>)> = Vec::new();
        if let Some(p) = obj.get("properties") {
            let pmap = p
                .as_object()
                .ok_or(SchemaError::BadKeyword("properties", "an object"))?;
            for (k, v) in pmap {
                props.push((
                    k.clone(),
                    Arc::new(SchemaNode::parse_node(v, strict, defs, depth + 1)?),
                ));
            }
        }

        let declared_required: Vec<String> = match obj.get("required") {
            Some(r) => {
                let arr = r
                    .as_array()
                    .ok_or(SchemaError::BadKeyword("required", "an array"))?;
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            }
            None => Vec::new(),
        };

        // strict: every declared property is required.
        let required: Vec<String> = if strict {
            props.iter().map(|(k, _)| k.clone()).collect()
        } else {
            // keep schema (properties) order for determinism
            props
                .iter()
                .map(|(k, _)| k.clone())
                .filter(|k| declared_required.contains(k))
                .collect()
        };

        let additional = match obj.get("additionalProperties") {
            Some(Value::Bool(b)) => *b,
            // object schema for additionalProperties → permissive (we do
            // not type-check extra values in v1).
            Some(Value::Object(_)) => true,
            _ => !strict, // strict defaults additionalProperties:false
        };

        Ok(SchemaNode::Object {
            props: Arc::from(props),
            required: Arc::from(required),
            additional,
        })
    }

    fn parse_array(
        obj: &serde_json::Map<String, Value>,
        strict: bool,
        defs: &serde_json::Map<String, Value>,
        depth: usize,
    ) -> Result<SchemaNode, SchemaError> {
        let items = match obj.get("items") {
            Some(v) if v.is_object() => {
                Arc::new(SchemaNode::parse_node(v, strict, defs, depth + 1)?)
            }
            // tuple-items / missing items → Any element
            _ => Arc::new(SchemaNode::Any),
        };
        let min = obj
            .get("minItems")
            .and_then(Value::as_u64)
            .map(|n| n as usize);
        let max = obj
            .get("maxItems")
            .and_then(Value::as_u64)
            .map(|n| n as usize);
        Ok(SchemaNode::Array { items, min, max })
    }
}
