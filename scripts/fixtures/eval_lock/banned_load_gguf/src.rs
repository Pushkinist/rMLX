pub fn read_gguf(path: &CStr) -> Result<()> {
    let status = unsafe { sys::mlx_load_gguf(&raw mut a, &raw mut b, path.as_ptr()) };
    Ok(())
}

fn guarded_anchor(a: &Array) -> Result<()> {
    let s = with_eval_lock(|| unsafe { sys::mlx_array_eval(a.inner) });
    unsafe { check_status(s, "anchor") }
}
