pub fn eval_all(v: mlx_vector_array) -> Result<()> {
    let status = unsafe { sys::mlx_eval(v) };
    Ok(())
}

fn guarded_anchor(a: &Array) -> Result<()> {
    let s = with_eval_lock(|| unsafe { sys::mlx_array_eval(a.inner) });
    unsafe { check_status(s, "anchor") }
}
