//! The reach-set includes `mlx_eval`, every `mlx_array_item_*`,
//! `mlx_array_tostring`, `mlx_save_safetensors` and `mlx_load_gguf`.
//! None of those names in prose is a call site.
impl Array {
    pub fn eval(&self) -> Result<()> {
        let status = with_eval_lock(|| unsafe { sys::mlx_array_eval(self.inner) });
        unsafe { check_status(status, "Array::eval") }
    }
}
