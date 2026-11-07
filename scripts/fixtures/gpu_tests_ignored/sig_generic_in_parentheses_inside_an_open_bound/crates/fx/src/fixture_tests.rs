trait Tr<T, const N: usize> {}
impl Tr<fn(Vec<u8>), 4> for () {}

fn helper<T>()
where
    T: Tr<fn(Vec<u8>), { 4 }
    >,
{
    let d = Device::Gpu;
}

#[test]
fn gpu_s8() {
    helper::<()>();
}
