pub fn write_snapshot(path: &CStr, arr: &Array) -> Result<()> {
    let status = unsafe { sys::mlx_save_safetensors(path.as_ptr(), map, meta) };
    Ok(())
}

fn guarded_anchor(a: &Array) -> Result<()> {
    let s = with_eval_lock(|| unsafe { sys::mlx_array_eval(a.inner) });
    unsafe { check_status(s, "anchor") }
}
