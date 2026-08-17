impl Array {
    pub fn eval(&self) -> Result<()> {
        let status = with_eval_lock(|| {
// A column-0 comment inside the closure. rustfmt does not normalise comment
// indentation, so this shape survives `cargo fmt` and must not be read as a
// block opener that severs the guard chain.
            unsafe { sys::mlx_array_eval(self.inner) }
        });
        unsafe { check_status(status, "Array::eval") }
    }
}
