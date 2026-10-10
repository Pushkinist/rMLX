//! The transform contract of a Prism Hadamard pack, read from file facts.
//!
//! A pack names its contract file in `config.json` (`hadamard_config`). The
//! contract (`prism.hadamard.*`, version 1) gives the transform, the block
//! size, one `±1` sign vector per input width and the tensors folded forward
//! (linear layers) or inverse (the embedding). `config.json` lists the same
//! modules in `modules`, and each module carries a `<module>.signs` tensor.
//! [`HadamardContract::read`] checks that these facts agree and that the
//! contract is the one rMLX implements, and refuses with one error that names
//! the first fact that does not hold.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;
use std::path::Path;

use serde_json::Value;

use crate::shards::{is_plain_file_name, load_shard_index, ShardHandle, SHARD_INDEX, SINGLE_SHARD};

const CONFIG: &str = "config.json";
/// The `config.json` key that names the contract file of a pack.
pub(crate) const HADAMARD_CONFIG: &str = "hadamard_config";
const GDN_LAYOUT: &str = "grouped";
/// The blocks the vendor runtime validated (`runtime/artifact.py`).
const BLOCKS: [usize; 4] = [512, 1024, 2048, 4096];
const KEY: &str = "prism.hadamard.";
const TRANSFORM: &str = "normalized-sylvester-walsh-hadamard";
const AXIS: &str = "input-last-dimension";
const SIGN_MODE: &str = "explicit";
const MODULE_DTYPE: &str = "float16";

/// The parsed contract of one Hadamard pack.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct HadamardContract {
    /// Hadamard block size: a power of two that divides every input width.
    pub block: usize,
    /// `config.json` `base_model_type`: the architecture the pack folds.
    pub base_model_type: String,
    /// The prefix that turns a `modules` path into a tensor name
    /// (`language_model.` for an mlx-vlm namespace).
    pub tensor_prefix: String,
    /// One sign vector per input width.
    pub signs: BTreeMap<usize, Vec<f32>>,
    /// Module tensor name (prefix included, no `.weight`) -> input width, for
    /// the modules transformed before their linear layer.
    pub forward: BTreeMap<String, usize>,
    /// Module tensor name -> input width, for the modules transformed after
    /// their row lookup (the embedding).
    pub inverse: BTreeMap<String, usize>,
}

/// Why a Hadamard pack is refused. Each variant names the file and the key
/// (or tensor) that holds the fact.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HadamardContractError {
    /// A file cannot be read or parsed.
    #[error("cannot read {file}: {why}")]
    Unreadable {
        /// File name in the snapshot directory.
        file: String,
        /// Reason.
        why: String,
    },
    /// A required fact is absent.
    #[error("{file} has no `{key}`")]
    Missing {
        /// File name in the snapshot directory. For an absent `.signs`
        /// tensor, the comma-separated list of the shard files searched.
        file: String,
        /// Key or tensor name.
        key: String,
    },
    /// A fact names a contract that rMLX does not implement.
    #[error("{file} `{key}` is {found}; rMLX implements {expected}")]
    Unsupported {
        /// File name in the snapshot directory.
        file: String,
        /// Key or tensor name.
        key: String,
        /// The value found.
        found: String,
        /// The value rMLX implements.
        expected: String,
    },
    /// A fact is malformed or contradicts another fact.
    #[error("{file} `{key}`: {why}")]
    Invalid {
        /// File name in the snapshot directory.
        file: String,
        /// Key or tensor name.
        key: String,
        /// What is wrong.
        why: String,
    },
}

impl HadamardContract {
    /// Read the contract of the pack in `model_dir`. `Ok(None)` when
    /// `config.json` names no `hadamard_config` and no shard holds a `.signs`
    /// tensor: the snapshot is not a pack.
    ///
    /// For a pack, reads `config.json`, the contract file, and the headers and
    /// the `.signs` tensors of the shards the load reads: those the
    /// `model.safetensors.index.json` names, or `model.safetensors` when there
    /// is no index. For a snapshot with no `hadamard_config`, reads the
    /// headers of every visible `*.safetensors` file. No weight, and no other
    /// file.
    ///
    /// # Errors
    /// [`HadamardContractError`] naming the first fact that does not hold,
    /// among them a `.signs` tensor in a snapshot with no `hadamard_config`.
    pub fn read(model_dir: &Path) -> Result<Option<Self>, HadamardContractError> {
        let config = read_json(model_dir, CONFIG)?;
        let Some(contract_file) = config.get(HADAMARD_CONFIG) else {
            return refuse_sign_tensors(model_dir).map(|()| None);
        };
        let contract_file = contract_file
            .as_str()
            .ok_or_else(|| invalid(CONFIG, HADAMARD_CONFIG, "not a string"))?;
        if !is_plain_file_name(contract_file) {
            return Err(invalid(
                CONFIG,
                HADAMARD_CONFIG,
                format!("{contract_file:?} is not a file name in the snapshot directory"),
            ));
        }
        let base_model_type = config
            .get("base_model_type")
            .ok_or_else(|| missing(CONFIG, "base_model_type"))?
            .as_str()
            .ok_or_else(|| invalid(CONFIG, "base_model_type", "not a string"))?
            .to_owned();
        flag_off(&config, "/components/mtp", "components.mtp")?;
        // A tied head would reuse the inverse-folded embedding as a forward
        // linear, which the contract does not define.
        flag_off(&config, "/tie_word_embeddings", "tie_word_embeddings")?;
        flag_off(
            &config,
            "/text_config/tie_word_embeddings",
            "text_config.tie_word_embeddings",
        )?;
        match config.get("gdn_activation_layout") {
            None => {}
            Some(Value::String(layout)) if layout == GDN_LAYOUT => {}
            Some(Value::String(layout)) => {
                return Err(unsupported(
                    CONFIG,
                    "gdn_activation_layout",
                    format!("{layout:?}"),
                    GDN_LAYOUT,
                ));
            }
            Some(_) => return Err(invalid(CONFIG, "gdn_activation_layout", "not a string")),
        }

        let contract = Contract::parse(&read_json(model_dir, contract_file)?, contract_file)?;
        let modules = parse_modules(&config, contract.block)?;
        let tensor_prefix = contract.tensor_prefix(&modules)?;
        contract.match_modules(&modules, &tensor_prefix, contract_file)?;
        let (forward, inverse) = sign_widths(model_dir, &modules, &tensor_prefix, &contract.signs)?;
        Ok(Some(Self {
            block: contract.block,
            base_model_type,
            tensor_prefix,
            signs: contract.signs,
            forward,
            inverse,
        }))
    }
}

/// A `config.json` flag at `pointer` that must be absent or `false`.
fn flag_off(config: &Value, pointer: &str, key: &str) -> Result<(), HadamardContractError> {
    match config.pointer(pointer) {
        None | Some(Value::Bool(false)) => Ok(()),
        Some(Value::Bool(true)) => Err(unsupported(CONFIG, key, true, "false")),
        Some(_) => Err(invalid(CONFIG, key, "not a bool")),
    }
}

fn missing(file: &str, key: &str) -> HadamardContractError {
    HadamardContractError::Missing {
        file: file.to_owned(),
        key: key.to_owned(),
    }
}

fn invalid(file: &str, key: &str, why: impl Into<String>) -> HadamardContractError {
    HadamardContractError::Invalid {
        file: file.to_owned(),
        key: key.to_owned(),
        why: why.into(),
    }
}

fn unsupported(
    file: &str,
    key: &str,
    found: impl Display,
    expected: &str,
) -> HadamardContractError {
    HadamardContractError::Unsupported {
        file: file.to_owned(),
        key: key.to_owned(),
        found: found.to_string(),
        expected: expected.to_owned(),
    }
}

/// The per-column signs of a Hadamard module: a tensor whose last name
/// component is `signs`.
fn is_sign_tensor(name: &str) -> bool {
    name.rsplit_once('.')
        .is_some_and(|(_, last)| last == "signs")
}

/// Refuse a `.signs` tensor in a snapshot whose `config.json` names no
/// contract. Scans every visible `*.safetensors` file; a hidden file (an
/// AppleDouble `._*` copy) and a file that cannot be opened or parsed feed no
/// weights, so they are skipped.
fn refuse_sign_tensors(model_dir: &Path) -> Result<(), HadamardContractError> {
    let entries = std::fs::read_dir(model_dir).map_err(|e| HadamardContractError::Unreadable {
        file: ".".to_owned(),
        why: format!("cannot list the snapshot directory: {e}"),
    })?;
    let mut files = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| HadamardContractError::Unreadable {
            file: ".".to_owned(),
            why: format!("cannot read a directory entry: {e}"),
        })?;
        if let Ok(name) = entry.file_name().into_string() {
            if !name.starts_with('.') && name.ends_with(".safetensors") {
                files.push(name);
            }
        }
    }
    files.sort_unstable();
    for file in files {
        let shard = match ShardHandle::open(model_dir, &file) {
            Ok(shard) => shard,
            Err(e) => {
                tracing::warn!(file = %file, error = %e, "Hadamard check skips a file it cannot open");
                continue;
            }
        };
        let header = match shard.safetensors() {
            Ok(header) => header,
            Err(e) => {
                tracing::warn!(file = %file, error = %e, "Hadamard check skips a file it cannot parse");
                continue;
            }
        };
        if let Some(name) = header.names().into_iter().find(|n| is_sign_tensor(n)) {
            return Err(invalid(
                &file,
                name,
                format!("a sign tensor, and {CONFIG} names no `{HADAMARD_CONFIG}`"),
            ));
        }
    }
    Ok(())
}

fn read_json(dir: &Path, file: &str) -> Result<Value, HadamardContractError> {
    let unreadable = |why: String| HadamardContractError::Unreadable {
        file: file.to_owned(),
        why,
    };
    let bytes = std::fs::read(dir.join(file)).map_err(|e| unreadable(e.to_string()))?;
    serde_json::from_slice(&bytes).map_err(|e| unreadable(e.to_string()))
}

/// One `config.json` `modules` record.
struct Module {
    path: String,
    embedding: bool,
}

/// The facts of the contract file.
struct Contract {
    block: usize,
    signs: BTreeMap<usize, Vec<f32>>,
    /// Full tensor names (`.weight` included) folded forward.
    forward: BTreeSet<String>,
    /// Full tensor names folded inverse.
    inverse: BTreeSet<String>,
}

impl Contract {
    fn parse(v: &Value, file: &str) -> Result<Self, HadamardContractError> {
        let obj = v
            .as_object()
            .ok_or_else(|| invalid(file, "prism.hadamard", "not a JSON object"))?;
        let required = |name: &str| -> Result<(String, &Value), HadamardContractError> {
            let key = format!("{KEY}{name}");
            match obj.get(&key) {
                Some(v) => Ok((key, v)),
                None => Err(missing(file, &key)),
            }
        };

        let (key, v) = required("version")?;
        let version = v
            .as_u64()
            .ok_or_else(|| invalid(file, &key, "not an integer"))?;
        if version != 1 {
            return Err(unsupported(file, &key, version, "1"));
        }
        for (name, want) in [
            ("transform", TRANSFORM),
            ("axis", AXIS),
            ("sign_mode", SIGN_MODE),
        ] {
            let (key, v) = required(name)?;
            let found = v
                .as_str()
                .ok_or_else(|| invalid(file, &key, "not a string"))?;
            if found != want {
                return Err(unsupported(file, &key, format!("{found:?}"), want));
            }
        }
        // The Swift reference reads an absent gdn_v_grouped as false; rMLX
        // implements the grouped value-head order only.
        let key = format!("{KEY}gdn_v_grouped");
        match obj.get(&key) {
            Some(Value::Bool(true)) => {}
            None => {
                return Err(unsupported(file, &key, "absent (reads as false)", "true"));
            }
            Some(Value::Bool(false)) => return Err(unsupported(file, &key, false, "true")),
            Some(_) => return Err(invalid(file, &key, "not a bool")),
        }

        let (key, v) = required("block_size")?;
        let found = v
            .as_u64()
            .ok_or_else(|| invalid(file, &key, "not an integer"))?;
        let block = usize::try_from(found)
            .ok()
            .filter(|b| BLOCKS.contains(b))
            .ok_or_else(|| unsupported(file, &key, found, "512, 1024, 2048 or 4096"))?;
        let signs = parse_signs(file, block, &required)?;

        let (_, v) = required("weight_names")?;
        let forward = names(file, "weight_names", Some(v))?;
        let inverse_key = format!("{KEY}inverse_weight_names");
        let inverse = names(file, "inverse_weight_names", obj.get(&inverse_key))?;
        if let Some(both) = forward.intersection(&inverse).next() {
            return Err(invalid(
                file,
                &inverse_key,
                format!("`{both}` is also a forward name"),
            ));
        }
        Ok(Self {
            block,
            signs,
            forward,
            inverse,
        })
    }

    /// The prefix that turns a module path into a contract tensor name, read
    /// from the first module: the one contract name that ends in its
    /// `<path>.weight` at a `.` boundary.
    fn tensor_prefix(&self, modules: &[Module]) -> Result<String, HadamardContractError> {
        let first = modules
            .first()
            .ok_or_else(|| invalid(CONFIG, "modules", "empty"))?;
        let suffix = format!("{}.weight", first.path);
        let mut found = self
            .forward
            .iter()
            .chain(&self.inverse)
            .filter_map(|n| n.strip_suffix(&suffix))
            .filter(|p| p.is_empty() || p.ends_with('.'));
        match (found.next(), found.next()) {
            (Some(prefix), None) => Ok(prefix.to_owned()),
            _ => Err(invalid(
                CONFIG,
                "modules[0].path",
                format!("no single contract name ends in `{suffix}`"),
            )),
        }
    }

    /// Each module is a contract name of its kind, and each contract name is
    /// a module.
    fn match_modules(
        &self,
        modules: &[Module],
        prefix: &str,
        file: &str,
    ) -> Result<(), HadamardContractError> {
        let mut named = BTreeSet::new();
        for (i, m) in modules.iter().enumerate() {
            let name = format!("{prefix}{}.weight", m.path);
            let (set, list) = if m.embedding {
                (&self.inverse, "inverse_weight_names")
            } else {
                (&self.forward, "weight_names")
            };
            if !set.contains(&name) {
                return Err(invalid(
                    CONFIG,
                    &format!("modules[{i}].path"),
                    format!("`{name}` is not in {file} {KEY}{list}"),
                ));
            }
            named.insert(name);
        }
        for (set, list) in [
            (&self.forward, "weight_names"),
            (&self.inverse, "inverse_weight_names"),
        ] {
            if let Some(name) = set.iter().find(|n| !named.contains(*n)) {
                return Err(invalid(
                    file,
                    &format!("{KEY}{list}"),
                    format!("`{name}` has no record in {CONFIG} modules"),
                ));
            }
        }
        Ok(())
    }
}

/// One `±1` vector per width, split from `sign_values` in `sign_widths` order.
fn parse_signs<'a>(
    file: &str,
    block: usize,
    required: &impl Fn(&str) -> Result<(String, &'a Value), HadamardContractError>,
) -> Result<BTreeMap<usize, Vec<f32>>, HadamardContractError> {
    let (key, v) = required("sign_widths")?;
    let widths: Vec<usize> = v
        .as_array()
        .and_then(|a| {
            a.iter()
                .map(|w| w.as_u64().and_then(|w| usize::try_from(w).ok()))
                .collect()
        })
        .ok_or_else(|| invalid(file, &key, "not an array of widths"))?;
    if widths.is_empty() {
        return Err(invalid(file, &key, "empty"));
    }
    if let Some(w) = widths.iter().find(|w| **w == 0 || !w.is_multiple_of(block)) {
        return Err(invalid(
            file,
            &key,
            format!("{w} is not a positive multiple of block {block}"),
        ));
    }
    if widths.iter().collect::<BTreeSet<_>>().len() != widths.len() {
        return Err(invalid(file, &key, "a width appears twice"));
    }
    let total = widths
        .iter()
        .try_fold(0usize, |sum, w| sum.checked_add(*w))
        .ok_or_else(|| invalid(file, &key, "the widths sum past the address space"))?;

    let (key, v) = required("sign_values")?;
    let values = v
        .as_array()
        .ok_or_else(|| invalid(file, &key, "not an array"))?;
    let mut flat = Vec::with_capacity(values.len());
    for (at, x) in values.iter().enumerate() {
        match x.as_f64() {
            Some(s) if s.to_bits() == 1.0f64.to_bits() => flat.push(1.0f32),
            Some(s) if s.to_bits() == (-1.0f64).to_bits() => flat.push(-1.0f32),
            _ => {
                return Err(invalid(
                    file,
                    &key,
                    format!("value {at} is {x}, not +1 or -1"),
                ));
            }
        }
    }
    if flat.len() != total {
        return Err(invalid(
            file,
            &key,
            format!("{} values, the widths need {total}", flat.len()),
        ));
    }
    let mut signs = BTreeMap::new();
    let mut rest = flat.as_slice();
    for w in widths {
        // Cannot fail: the widths sum to `flat.len()`, checked above.
        let (vector, tail) = rest
            .split_at_checked(w)
            .ok_or_else(|| invalid(file, &key, "fewer values than the widths need"))?;
        signs.insert(w, vector.to_vec());
        rest = tail;
    }
    Ok(signs)
}

/// The set of tensor names in contract list `list`; an absent list is empty.
fn names(
    file: &str,
    list: &str,
    v: Option<&Value>,
) -> Result<BTreeSet<String>, HadamardContractError> {
    let key = format!("{KEY}{list}");
    let Some(v) = v else {
        return Ok(BTreeSet::new());
    };
    let all: Vec<&str> = v
        .as_array()
        .and_then(|a| a.iter().map(Value::as_str).collect())
        .ok_or_else(|| invalid(file, &key, "not an array of names"))?;
    let set: BTreeSet<String> = all.iter().map(|n| (*n).to_owned()).collect();
    if set.len() != all.len() {
        return Err(invalid(file, &key, "a name appears twice"));
    }
    Ok(set)
}

fn parse_modules(config: &Value, block: usize) -> Result<Vec<Module>, HadamardContractError> {
    let list = config
        .get("modules")
        .ok_or_else(|| missing(CONFIG, "modules"))?
        .as_array()
        .ok_or_else(|| invalid(CONFIG, "modules", "not an array"))?;
    let mut seen = BTreeSet::new();
    let mut modules = Vec::with_capacity(list.len());
    for (i, record) in list.iter().enumerate() {
        let key = |field: &str| format!("modules[{i}].{field}");
        let field = |field: &str| -> Result<&Value, HadamardContractError> {
            record
                .get(field)
                .ok_or_else(|| missing(CONFIG, &key(field)))
        };
        let path = field("path")?
            .as_str()
            .ok_or_else(|| invalid(CONFIG, &key("path"), "not a string"))?;
        let dtype = field("dtype")?
            .as_str()
            .ok_or_else(|| invalid(CONFIG, &key("dtype"), "not a string"))?;
        if dtype != MODULE_DTYPE {
            return Err(unsupported(
                CONFIG,
                &key("dtype"),
                format!("{dtype:?}"),
                MODULE_DTYPE,
            ));
        }
        let record_block = field("block")?;
        if record_block.as_u64() == Some(0) {
            return Err(invalid(
                CONFIG,
                &key("block"),
                "block 0 (unfolded module) is not implemented",
            ));
        }
        if record_block.as_u64().and_then(|b| usize::try_from(b).ok()) != Some(block) {
            return Err(invalid(
                CONFIG,
                &key("block"),
                format!("{record_block}, the contract block is {block}"),
            ));
        }
        let embedding = field("embedding")?
            .as_bool()
            .ok_or_else(|| invalid(CONFIG, &key("embedding"), "not a bool"))?;
        if !seen.insert(path) {
            return Err(invalid(
                CONFIG,
                &key("path"),
                format!("`{path}` appears twice"),
            ));
        }
        modules.push(Module {
            path: path.to_owned(),
            embedding,
        });
    }
    Ok(modules)
}

/// Module tensor name -> width, forward and inverse, over the shards the
/// load reads. Each module's `.signs` tensor must be a one-axis F32 vector,
/// in one shard only, equal to the contract's vector of its width. No
/// `.signs` tensor may belong to no module, and no `<prefix>...scales` tensor
/// (a quantized linear) may lack a module record.
fn sign_widths(
    model_dir: &Path,
    modules: &[Module],
    prefix: &str,
    signs: &BTreeMap<usize, Vec<f32>>,
) -> Result<(BTreeMap<String, usize>, BTreeMap<String, usize>), HadamardContractError> {
    let tried = if model_dir.join(SHARD_INDEX).exists() {
        SHARD_INDEX
    } else {
        SINGLE_SHARD
    };
    let index = load_shard_index(model_dir).map_err(|e| {
        tracing::debug!(file = tried, error = %e, "Hadamard pack: shards not listed");
        HadamardContractError::Unreadable {
            file: tried.to_owned(),
            why: "cannot list the shards that load".to_owned(),
        }
    })?;
    let files: BTreeSet<&String> = index.weight_map.values().collect();
    let mut handles = Vec::new();
    for file in files {
        // The loader's error names the absolute path; the refusal must not.
        let handle = ShardHandle::open(model_dir, file).map_err(|e| {
            tracing::debug!(file = %file, error = %e, "Hadamard pack: shard not opened");
            HadamardContractError::Unreadable {
                file: file.clone(),
                why: "cannot open or map the shard".to_owned(),
            }
        })?;
        handles.push((file.clone(), handle));
    }
    let mut headers = Vec::new();
    for (file, handle) in &handles {
        let header = handle
            .safetensors()
            .map_err(|e| HadamardContractError::Unreadable {
                file: file.clone(),
                why: e.to_string(),
            })?;
        headers.push((file.as_str(), header));
    }
    let all_files = headers
        .iter()
        .map(|(f, _)| *f)
        .collect::<Vec<_>>()
        .join(", ");

    let mut forward = BTreeMap::new();
    let mut inverse = BTreeMap::new();
    let mut expected = BTreeSet::new();
    for m in modules {
        let module = format!("{prefix}{}", m.path);
        let name = format!("{module}.signs");
        let holders: Vec<_> = headers
            .iter()
            .filter_map(|(f, h)| h.tensor(&name).ok().map(|v| (*f, v)))
            .collect();
        let (file, view) = match holders.as_slice() {
            [] => return Err(missing(&all_files, &name)),
            [one] => one,
            many => {
                let files = many.iter().map(|(f, _)| *f).collect::<Vec<_>>().join(", ");
                return Err(invalid(&files, &name, "in more than one shard"));
            }
        };
        if view.dtype() != safetensors::Dtype::F32 {
            return Err(invalid(
                file,
                &name,
                format!("dtype {:?}, want F32", view.dtype()),
            ));
        }
        let &[width] = view.shape() else {
            return Err(invalid(
                file,
                &name,
                format!("shape {:?}, want one axis", view.shape()),
            ));
        };
        let want = signs.get(&width).ok_or_else(|| {
            invalid(
                file,
                &name,
                format!("width {width} has no sign vector in the contract"),
            )
        })?;
        let got = view
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c));
        if let Some(at) = got.zip(want).position(|(g, w)| g.to_bits() != w.to_bits()) {
            return Err(invalid(
                file,
                &name,
                format!("value {at} differs from the contract's vector of width {width}"),
            ));
        }
        if m.embedding {
            inverse.insert(module, width);
        } else {
            forward.insert(module, width);
        }
        expected.insert(name);
    }
    for (file, header) in &headers {
        let mut names = header.names();
        names.sort_unstable();
        for name in names {
            if is_sign_tensor(name) && !expected.contains(name) {
                return Err(invalid(file, name, "a sign tensor of no module"));
            }
            let quantized_module = name
                .strip_suffix(".scales")
                .filter(|m| m.starts_with(prefix));
            if let Some(module) = quantized_module {
                if !forward.contains_key(module) && !inverse.contains_key(module) {
                    return Err(invalid(
                        file,
                        name,
                        "a quantized linear with no record in config.json modules",
                    ));
                }
            }
        }
    }
    Ok((forward, inverse))
}

#[cfg(test)]
#[path = "hadamard_tests.rs"]
mod hadamard_tests;
