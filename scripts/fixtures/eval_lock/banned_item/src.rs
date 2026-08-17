impl Array {
    pub fn to_scalar(&self) -> f32 {
        let mut out = 0.0f32;
        unsafe { sys::mlx_array_item_float32(&raw mut out, self.inner) };
        out
    }
}

fn guarded_anchor(a: &Array) -> Result<()> {
    let s = with_eval_lock(|| unsafe { sys::mlx_array_eval(a.inner) });
    unsafe { check_status(s, "anchor") }
}
