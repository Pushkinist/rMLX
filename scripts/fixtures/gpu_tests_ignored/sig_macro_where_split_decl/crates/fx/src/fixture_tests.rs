macro_rules! decl {
    ($name:ident) => {
        trait T {
            fn $name<U>(&self, u: U)
            where
                U: Copy;
        }
    };
}

#[test]
fn gpu_k() {
    let device = Device::Gpu;
    if true {
        run(device);
    }
}
