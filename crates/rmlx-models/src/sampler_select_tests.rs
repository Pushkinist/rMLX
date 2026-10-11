//! `select_token` against the four selections it forks to, on the same rows.

use super::*;

const VOCAB: usize = 64;

fn row(peak: &[(usize, f32)]) -> Result<Array> {
    let mut v = vec![0.0f32; VOCAB];
    for &(id, x) in peak {
        if let Some(slot) = v.get_mut(id) {
            *slot = x;
        }
    }
    Array::from_f32_slice(&v, &[1, VOCAB as i32])
}

fn id_of(a: &Array) -> Result<i32> {
    a.eval()?;
    let b = a.to_bytes()?;
    let four: [u8; 4] = b
        .get(..4)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| Error::Other("id array shorter than 4 bytes".to_owned()))?;
    Ok(i32::from_le_bytes(four))
}

fn forbid(id: usize) -> Vec<bool> {
    (0..VOCAB).map(|i| i != id).collect()
}

fn sampler(temperature: f32, top_k: u32) -> SamplerConfig {
    SamplerConfig {
        temperature,
        top_p: 1.0,
        top_k,
        min_p: 0.0,
        seed: None,
        top_logprobs_k: 0,
    }
}

fn penalty(rep_penalty: f32) -> PenaltyConfig {
    PenaltyConfig {
        rep_penalty,
        ..PenaltyConfig::default()
    }
}

#[test]
fn greedy_without_mask_or_penalty_is_the_device_argmax() -> Result<()> {
    let logits = row(&[(9, 4.0), (30, 3.0)])?;
    let mut rng = Pcg32::new(1);
    let got = select_token(
        &logits,
        &sampler(0.0, 0),
        None,
        &penalty(1.0),
        &[],
        &mut rng,
        Device::Cpu,
    )?;
    let want = argmax(&logits, -1, Device::Cpu)?;
    assert_eq!(id_of(&got)?, id_of(&want)?);
    assert_eq!(id_of(&got)?, 9);
    Ok(())
}

#[test]
fn greedy_with_a_mask_is_the_masked_argmax() -> Result<()> {
    let logits = row(&[(9, 4.0), (30, 3.0)])?;
    let mask = forbid(9);
    let mut rng = Pcg32::new(1);
    let got = select_token(
        &logits,
        &sampler(0.0, 0),
        Some(&mask),
        &penalty(1.0),
        &[],
        &mut rng,
        Device::Cpu,
    )?;
    let want = apply_mask_argmax(&logits, &mask, Device::Cpu)?;
    assert_eq!(id_of(&got)?, id_of(&want)?);
    assert_eq!(id_of(&got)?, 30);
    Ok(())
}

#[test]
fn greedy_with_a_penalty_is_the_penalised_argmax() -> Result<()> {
    let logits = row(&[(9, 4.0), (30, 3.9)])?;
    let pen = penalty(1.1);
    let mut rng = Pcg32::new(1);
    let got = select_token(
        &logits,
        &sampler(0.0, 0),
        None,
        &pen,
        &[9],
        &mut rng,
        Device::Cpu,
    )?;
    let want = argmax_with_penalties(&logits, None, &pen, &[9], Device::Cpu)?;
    assert_eq!(id_of(&got)?, id_of(&want)?);
    assert_eq!(id_of(&got)?, 30);
    Ok(())
}

/// A seeded sampled row: the same seed gives the same 32 draws through
/// `select_token` as through `sample_token_array`, so the RNG advances once per pick.
#[test]
fn sampling_is_the_host_sampler_with_the_same_stream() -> Result<()> {
    let logits = row(&[(3, 1.0), (4, 1.5), (5, 0.5), (6, 1.2)])?;
    let mask = forbid(4);
    let sp = sampler(0.8, 8);
    let pen = penalty(1.05);
    let mut rng_select = Pcg32::new(11);
    let mut rng_direct = Pcg32::new(11);
    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..32 {
        let got = select_token(
            &logits,
            &sp,
            Some(&mask),
            &pen,
            &[6],
            &mut rng_select,
            Device::Cpu,
        )?;
        let want = sample_token_array(
            &logits,
            &sp,
            Some(&mask),
            &pen,
            &[6],
            &mut rng_direct,
            Device::Cpu,
        )?;
        let id = id_of(&got)?;
        assert_eq!(id, id_of(&want)?);
        assert_ne!(id, 4, "a masked id was sampled");
        seen.insert(id);
    }
    assert!(seen.len() > 1, "32 sampled draws gave one id: {seen:?}");
    Ok(())
}
