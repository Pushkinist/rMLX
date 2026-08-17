impl Array {
    pub fn eval_twice(&self) -> Result<()> {
        let a = with_eval_lock(|| unsafe { sys::mlx_array_eval(self.inner) });
        let b = unsafe { sys::mlx_array_eval(self.other) };
        Ok(())
    }
}
