impl Array {
    pub fn eval(&self) -> Result<()> {
        let status = unsafe { sys::mlx_array_eval(self.inner) }; // with_eval_lock: caller already holds it
        unsafe { check_status(status, "Array::eval") }
    }
}
