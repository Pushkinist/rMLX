impl Array {
    pub fn eval(&self) -> Result<()> {
        let status = with_eval_lock(|| {
            let r = unsafe {
                sys::mlx_array_eval(self.inner)
            };
            r
        });
        unsafe { check_status(status, "Array::eval") }
    }
}
