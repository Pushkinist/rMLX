impl Array {
    pub fn eval(&self) -> Result<()> {
        let status = unsafe { sys::ffi::mlx_array_eval(self.inner) };
        unsafe { check_status(status, "Array::eval") }
    }
}
