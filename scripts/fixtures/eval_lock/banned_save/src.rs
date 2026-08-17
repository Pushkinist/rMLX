pub fn write_snapshot(path: &CStr, arr: &Array) -> Result<()> {
    let status = unsafe { sys::mlx_save_safetensors(path.as_ptr(), map, meta) };
    Ok(())
}
