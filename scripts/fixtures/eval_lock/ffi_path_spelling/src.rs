impl Array {
    pub fn eval_alt(&self) -> Result<()> {
        let status = unsafe { sys::ffi::mlx_array_item_float32(&raw mut o, self.inner) };
        Ok(())
    }
}

fn guarded_anchor(a: &Array) -> Result<()> {
    let s = with_eval_lock(|| unsafe { sys::mlx_array_eval(a.inner) });
    unsafe { check_status(s, "anchor") }
}
