//! `FromStr for KvQuant` against a copy of the parser it replaced.
//!
//! The copy below is the parser as it was when every fieldless spelling was
//! one literal arm of a `match`. The live parser finds those spellings through
//! the codec descriptor. Both must give the same `Ok` value and the same `Err`
//! text for every input, so the lookup cannot drop or add a spelling.

use std::str::FromStr;

use super::{
    parse_kv_side, parse_rotor_k_asym_v_suffix, validate_mixed_side, KvQuant, KvQuantParseError,
    ALL_KV_QUANTS,
};

#[allow(
    clippy::too_many_lines,
    reason = "a verbatim copy of the replaced parser; its value is that it is not edited"
)]
fn reference_from_str(s: &str) -> Result<KvQuant, KvQuantParseError> {
    match s {
        "none" | "bf16" | "f16" => return Ok(KvQuant::None),
        "k8v4" => return Ok(KvQuant::K8V4),
        "k8v8" => return Ok(KvQuant::K8V8),
        "planar" => return Ok(KvQuant::Planar),
        "planar3" => return Ok(KvQuant::Planar3),
        "k8vturbo3" => return Ok(KvQuant::K8VTurbo3),
        "tsym3" => return Ok(KvQuant::TurboSym3),
        "tsym4" => return Ok(KvQuant::TurboSym4),
        "planar_k" => return Ok(KvQuant::PlanarK),
        "k8vturbo2" => return Ok(KvQuant::K8VTurbo2),
        "iso3" => return Ok(KvQuant::Iso3),
        "iso4" => return Ok(KvQuant::Iso4),
        "rotor3" | "rotor_v_3" => return Ok(KvQuant::Rotor3),
        "rotor4" | "rotor_v_4" => return Ok(KvQuant::Rotor4),
        "k8vturbo3tcq" => return Ok(KvQuant::K8VTurbo3Tcq),
        "k8vturbo2tcq" => return Ok(KvQuant::K8VTurbo2Tcq),
        // Symmetric / K-only iso variants.
        "iso3_sym" => return Ok(KvQuant::Iso3Sym),
        "iso4_sym" => return Ok(KvQuant::Iso4Sym),
        "k_iso3" => return Ok(KvQuant::IsoKOnly3),
        "k_iso4" => return Ok(KvQuant::IsoKOnly4),
        // Symmetric / K-only rotor variants.
        "rotor3_sym" => return Ok(KvQuant::Rotor3Sym),
        "rotor4_sym" => return Ok(KvQuant::Rotor4Sym),
        "k_rotor3" => return Ok(KvQuant::RotorKOnly3),
        "k_rotor4" => return Ok(KvQuant::RotorKOnly4),
        _ => {}
    }

    // Withdrawn codecs: reject by name, and name the successor. Not an
    // alias — a retired name has to keep failing, or a recorded bench cell
    // or a saved CLI line would keep running under a codec it does not
    // name. `rot_k_tq4v` (rotated affine-8 K + TurboQuant-4 V) rebuilt a
    // full bf16 K *and* V from its packed store on every decode step and
    // then ran ordinary bf16 SDPA; `rot_k_v4g64` is the same rotated 8-bit
    // K with an MLX-affine 4-bit V that `mixed_quantized_sdpa` consumes
    // without materialising either axis.
    if s == "rot_k_tq4v" {
        return Err(KvQuantParseError::Retired {
            input: s.to_string(),
            replacement: "rot_k_v4g64",
        });
    }

    // "rot_k_v<vb>g<vg>" — RotK Display form round-trip.
    if let Some(rest) = s.strip_prefix("rot_k_v") {
        // The shape already matched, so a malformed numeric component is a
        // bad `rot_k_*` spelling and not an unknown codec: reporting it as
        // `Unknown` printed the whole codec list and never said which part
        // of the tag failed.
        let mk_err = |reason: String| KvQuantParseError::InvalidRotK {
            input: s.to_string(),
            reason,
        };
        let (v_bits, v_group_size) = rest
            .split_once('g')
            .ok_or_else(|| mk_err(format!("missing 'g' separator in 'v{rest}'")))
            .and_then(|(bits_str, group_str)| {
                let v_bits: u8 = bits_str
                    .parse()
                    .map_err(|e| mk_err(format!("bad v_bits in 'v{rest}': {e}")))?;
                let v_group_size: u16 = group_str
                    .parse()
                    .map_err(|e| mk_err(format!("bad v_group_size in 'v{rest}': {e}")))?;
                Ok((v_bits, v_group_size))
            })?;
        // RotK's V slot *is* Mixed's V slot: `KvStorage::new` builds it
        // with `MixedKvState::new_rotated`, which hands (bits, group_size)
        // to the same MLX affine quantizer. Validating it with the same
        // function keeps the two from accepting different sets, so
        // `rot_k_v99g7` fails here and not at its first 99-bit quantize.
        validate_mixed_side('v', v_bits, v_group_size).map_err(mk_err)?;
        return Ok(KvQuant::RotK {
            v_bits,
            v_group_size,
        });
    }

    // "rotor_k_3_asym_v<vb>_g<vg>" / "rotor_k_4_asym_v<vb>_g<vg>".
    if let Some(rest) = s.strip_prefix("rotor_k_3_asym_") {
        let (v_bits, v_group_size) = parse_rotor_k_asym_v_suffix(rest, s)?;
        return Ok(KvQuant::RotorK3Asym {
            v_bits,
            v_group_size,
        });
    }
    if let Some(rest) = s.strip_prefix("rotor_k_4_asym_") {
        let (v_bits, v_group_size) = parse_rotor_k_asym_v_suffix(rest, s)?;
        return Ok(KvQuant::RotorK4Asym {
            v_bits,
            v_group_size,
        });
    }

    // Mixed shape: "mixed_k<kb>g<kg>_v<vb>g<vg>".
    if let Some(rest) = s.strip_prefix("mixed_") {
        // Split on the single '_' between the K- and V-side specs.
        let (k_part, v_part) =
            rest.split_once('_')
                .ok_or_else(|| KvQuantParseError::InvalidMixed {
                    input: s.to_string(),
                    reason: "missing '_' between K-side and V-side".to_string(),
                })?;

        let (k_bits, k_group_size) =
            parse_kv_side(k_part, 'k').map_err(|reason| KvQuantParseError::InvalidMixed {
                input: s.to_string(),
                reason,
            })?;
        let (v_bits, v_group_size) =
            parse_kv_side(v_part, 'v').map_err(|reason| KvQuantParseError::InvalidMixed {
                input: s.to_string(),
                reason,
            })?;
        validate_mixed_side('k', k_bits, k_group_size).map_err(|reason| {
            KvQuantParseError::InvalidMixed {
                input: s.to_string(),
                reason,
            }
        })?;
        validate_mixed_side('v', v_bits, v_group_size).map_err(|reason| {
            KvQuantParseError::InvalidMixed {
                input: s.to_string(),
                reason,
            }
        })?;

        return Ok(KvQuant::Mixed {
            k_bits,
            v_bits,
            k_group_size,
            v_group_size,
        });
    }

    Err(KvQuantParseError::Unknown(s.to_string()))
}

/// Inputs that name a codec, an alias, a retired codec, a payload shape at
/// its edge widths, or nothing, plus every truncation and case change of each.
fn inputs() -> Vec<String> {
    let mut seeds: Vec<String> = ALL_KV_QUANTS.iter().map(ToString::to_string).collect();
    seeds.extend(
        [
            "bf16",
            "f16",
            "rotor_v_3",
            "rotor_v_4",
            "rot_k_tq4v",
            "rotor_v_5",
            "rotor_v_",
            "k_iso5",
            "iso3_sym_",
            "none_bf16",
            "mixed",
            "paged",
            "rot_k",
            "rot_k_",
            "rot_k_v",
            "rot_k_vg",
            "rot_k_v4g",
            "rot_k_vg64",
            "rot_k_v4x64",
            "rot_k_v-1g64",
            "mixed_",
            "mixed_k8g64",
            "mixed_k8g64_",
            "mixed_k8g64_v4",
            "mixed_k8_v4g64",
            "mixed_x8g64_v4g64",
            "mixed_k8g64_x4g64",
            "mixed_k8g64_v4g64_",
            "rotor_k_3_asym_",
            "rotor_k_3_asym_v4",
            "rotor_k_3_asym_4_g64",
            "rotor_k_3_asym_v4g64",
            "rotor_k_5_asym_v4_g64",
            "rotor_k__asym_v4_g64",
            "",
            " ",
            "_",
            "none ",
            " none",
            "NONE",
            "Bf16",
            "k8v4\n",
            "k8v4\0",
            "planar\t",
            "tsym",
            "iso",
            "rotor",
        ]
        .iter()
        .map(ToString::to_string),
    );
    let bits = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "16", "255", "256", "-1", "04",
    ];
    let groups = ["0", "16", "32", "64", "128", "256", "65535", "65536", "064"];
    for b in bits {
        for g in groups {
            seeds.push(format!("rot_k_v{b}g{g}"));
            seeds.push(format!("mixed_k{b}g{g}_v{b}g{g}"));
            seeds.push(format!("mixed_k8g64_v{b}g{g}"));
            seeds.push(format!("mixed_k{b}g{g}_v4g64"));
            seeds.push(format!("rotor_k_3_asym_v{b}_g{g}"));
            seeds.push(format!("rotor_k_4_asym_v{b}_g{g}"));
        }
    }
    let mut all = Vec::new();
    for seed in &seeds {
        all.push(seed.clone());
        all.push(seed.to_uppercase());
        all.push(format!("{seed}x"));
        all.push(format!("x{seed}"));
        for (end, _) in seed.char_indices().skip(1) {
            all.push(seed[..end].to_string());
            all.push(seed[end..].to_string());
        }
    }
    all.sort();
    all.dedup();
    all
}

#[test]
fn from_str_matches_the_replaced_parser_on_every_input() {
    let inputs = inputs();
    assert!(inputs.len() > 5000, "input set shrank to {}", inputs.len());
    let mut accepted = 0usize;
    for input in &inputs {
        let live = KvQuant::from_str(input);
        let reference = reference_from_str(input);
        assert_eq!(live, reference, "input {input:?}");
        assert_eq!(
            live.as_ref().map_err(ToString::to_string),
            reference.as_ref().map_err(ToString::to_string),
            "error text for input {input:?}"
        );
        accepted += usize::from(live.is_ok());
    }
    // Each listed codec, the four aliases and the valid payload widths.
    assert!(
        accepted > ALL_KV_QUANTS.len() + 4,
        "only {accepted} inputs parsed"
    );
}
