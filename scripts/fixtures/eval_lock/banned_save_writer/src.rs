pub fn stream_out(w: mlx_io_writer, a: &Array) -> Result<()> {
    let status = unsafe { sys::mlx_save_writer(w, a.inner) };
    Ok(())
}

fn guarded_anchor(a: &Array) -> Result<()> {
    let s = with_eval_lock(|| unsafe { sys::mlx_array_eval(a.inner) });
    unsafe { check_status(s, "anchor") }
}
