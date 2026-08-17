impl std::fmt::Debug for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = unsafe { sys::mlx_string_new() };
        unsafe { sys::mlx_array_tostring(&raw mut s, self.inner) };
        write!(f, "Array")
    }
}
