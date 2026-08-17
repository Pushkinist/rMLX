//! The reach-set includes sys::mlx_eval(v), every sys::mlx_array_item_float32(p, a),
//! sys::mlx_array_tostring(s, a), sys::mlx_save_safetensors(p, m, d) and
//! sys::mlx_load_gguf(a, b, c) — named here with their parentheses on purpose,
//! so that deleting the `sys::` anchor from the banned pattern does not leave
//! this fixture silently green.
impl Array {
    pub fn eval(&self) -> Result<()> {
        let status = with_eval_lock(|| unsafe { sys::mlx_array_eval(self.inner) });
        unsafe { check_status(status, "Array::eval") }
    }
}
