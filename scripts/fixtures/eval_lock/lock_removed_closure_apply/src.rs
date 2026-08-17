impl Closure {
    pub fn apply(&self, inputs: &[&Array]) -> Result<Vec<Array>> {
        let status = unsafe { sys::mlx_closure_apply(&raw mut vec_out, self.inner, vec_in) };
        unsafe { check_status(status, "Closure::apply") }?;
        Ok(Vec::new())
    }
}

fn guarded_anchor(a: &Array) -> Result<()> {
    let s = with_eval_lock(|| unsafe { sys::mlx_array_eval(a.inner) });
    unsafe { check_status(s, "anchor") }
}
