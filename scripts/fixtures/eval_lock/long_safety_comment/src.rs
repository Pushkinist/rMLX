impl Array {
    pub fn eval(&self) -> Result<()> {
        let status = with_eval_lock(|| {
            // SAFETY: inner is a valid mlx_array handle for the whole call.
            // The lock is held across exactly this FFI call and nothing else.
            // Error capture is thread-local, so check_status stays outside.
            // This comment is deliberately long: it is the crate convention.
            unsafe { sys::mlx_array_eval(self.inner) }
        });
        unsafe { check_status(status, "Array::eval") }
    }
}
