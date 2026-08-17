impl Array {
    pub fn eval(&self) -> Result<()> {
        // SAFETY: inner is a valid mlx_array.
        let status = with_eval_lock(|| unsafe { sys::mlx_array_eval(self.inner) });
        unsafe { check_status(status, "Array::eval") }
    }

    pub fn async_eval(&self) -> Result<()> {
        let status = with_eval_lock(|| unsafe { sys::mlx_async_eval(vec) });
        unsafe { check_status(status, "Array::async_eval") }
    }

    pub fn apply(&self) -> Result<()> {
        let status = with_eval_lock(|| unsafe { sys::mlx_closure_apply(&raw mut o, c, i) });
        unsafe { check_status(status, "Closure::apply") }
    }
}
