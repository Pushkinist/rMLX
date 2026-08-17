impl Array {
    pub fn async_eval(&self) -> Result<()> {
        let status = unsafe { sys::mlx_async_eval(vec) };
        unsafe { check_status(status, "Array::async_eval") }
    }
}

fn guarded_anchor(a: &Array) -> Result<()> {
    let s = with_eval_lock(|| unsafe { sys::mlx_array_eval(a.inner) });
    unsafe { check_status(s, "anchor") }
}
