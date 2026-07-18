use super::*;
use crate::{Array, Device, Dtype};

/// Sanity check: raw (uncompiled) Closure::from_fn + apply works end-to-end.
/// Uses identity (no-op MLX ops) to isolate the pack/unpack logic.
#[test]
fn raw_closure_apply_works() {
    // Element-wise double via add: arr + arr = 2*arr.
    let cls = Closure::from_fn(|inputs| {
        let arr = inputs
            .into_iter()
            .next()
            .ok_or_else(|| Error::Mlx("no inputs".to_owned()))?;
        let out = crate::add(&arr, &arr, Device::Cpu)?;
        Ok(vec![out])
    });

    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    let arr = Array::from_bytes(bytes, &[1, 4], Dtype::F32).expect("from_bytes");

    let out = cls.apply(&[&arr]).expect("raw apply");
    assert_eq!(out.len(), 1);
    out[0].eval().expect("eval before to_bytes");
    let bytes2 = out[0].to_bytes().expect("materialize");
    let result: Vec<f32> = bytes2
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    // 2x the input.
    assert_eq!(result, vec![2.0_f32, 4.0, 6.0, 8.0]);
}

/// Verify Closure::from_fn + compile_shapeless:
/// - Same shape invoked twice uses the compiled cache (no panic, correct output).
/// - Closure correctly transforms inputs on both the trace call and the cache hit.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test compile_shapeless -- --ignored --test-threads=1"]
fn compile_shapeless_cache_hit() {
    // Simple element-wise double: arr + arr = 2 * arr
    let cls = Closure::from_fn(|inputs| {
        let arr = inputs
            .into_iter()
            .next()
            .ok_or_else(|| Error::Mlx("no inputs".to_owned()))?;
        let out = crate::add(&arr, &arr, Device::Gpu)?;
        Ok(vec![out])
    });
    let compiled = compile_shapeless(cls).expect("compile_shapeless");

    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    let arr = Array::from_bytes(bytes, &[1, 4], Dtype::F32).expect("from_bytes");

    // First invocation — traces + compiles.
    let out1 = compiled.apply(&[&arr]).expect("apply 1");
    assert_eq!(out1.len(), 1);
    // Materialize the lazy MLX array (triggers Metal kernel dispatch).
    out1[0].eval().expect("eval 1");
    out1[0].to_bytes().expect("materialize 1");

    // Second invocation — same shape, should hit the compiled cache.
    let out2 = compiled.apply(&[&arr]).expect("apply 2");
    assert_eq!(out2.len(), 1);

    // Verify output: arr + arr = 2 * arr = [2, 4, 6, 8]
    out2[0].eval().expect("eval 2");
    let bytes2 = out2[0].to_bytes().expect("materialize 2");
    let result: Vec<f32> = bytes2
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let expected = vec![2.0_f32, 4.0, 6.0, 8.0];
    assert_eq!(result, expected, "compile_shapeless output mismatch");
}

/// Verify that shape-aware `compile` (NOT `compile_shapeless`) can trace a
/// closure containing a Rust for-loop emitting static-bound slice ops.
///
/// `compile_shapeless` rejects this with "Slice cannot infer output shapes"
/// because the symbolic tracer doesn't allow per-call output-shape variance
/// from a slice; shape-aware `compile` retraces per shape and accepts it.
/// This is the foundation for the GDN compile cache: cache one compiled
/// closure per distinct (B, T, Hv, Dv, Dk) shape tuple.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test compile_static_slice_loop -- --ignored --test-threads=1"]
fn compile_static_slice_loop() {
    const T: i32 = 4;
    let cls = Closure::from_fn(|inputs| {
        let x = inputs
            .into_iter()
            .next()
            .ok_or_else(|| Error::Mlx("no inputs".to_owned()))?;
        let s = x.shape();
        // x: [B, T, D]
        assert_eq!(s.len(), 3);
        let b = s[0];
        let t_dim = s[1];
        let d = s[2];
        // For each timestep, slice and add to running sum.
        let mut acc = crate::zeros(&[b, d], Dtype::F32, Device::Gpu)?;
        for ti in 0..t_dim {
            let slc = x
                .slice(&[0, ti, 0], &[b, ti + 1, d], &[1, 1, 1], Device::Gpu)?
                .reshape(&[b, d], Device::Gpu)?;
            acc = crate::add(&acc, &slc, Device::Gpu)?;
        }
        Ok(vec![acc])
    });
    let compiled = compile(cls).expect("compile");

    let data: Vec<f32> = (0..(T * 4) as usize).map(|i| i as f32).collect();
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    let arr = Array::from_bytes(bytes, &[1, T, 4], Dtype::F32).expect("from_bytes");

    // First call — traces + compiles. This is the call that previously failed on static-slice loops.
    let out = compiled.apply(&[&arr]).expect("static-slice-loop apply");
    out[0].eval().expect("eval");
    let result_bytes = out[0].to_bytes().expect("materialize");
    let result: Vec<f32> = result_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    // Expected: sum along axis 1: for each d, sum_{t=0..4} x[0, t, d].
    // x[0, t, d] = t*4 + d -> sum over t = (0+1+2+3)*4 + 4*d = 24 + 4d
    // Wait: for t=0..3, x[0,t,d] = t*4 + d -> sum = 6*4 + 4*d = wait...
    // d=0: 0+4+8+12 = 24
    // d=1: 1+5+9+13 = 28
    // d=2: 2+6+10+14 = 32
    // d=3: 3+7+11+15 = 36
    let expected = vec![24.0_f32, 28.0, 32.0, 36.0];
    assert_eq!(result, expected, "static-slice-loop wrong output");
}
