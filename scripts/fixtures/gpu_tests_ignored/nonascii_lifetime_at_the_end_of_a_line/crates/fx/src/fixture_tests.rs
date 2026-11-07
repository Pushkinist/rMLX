struct S<'é, const C: char>(&'é u8);
type T<'é> = S<'é,
'"'>;

#[test]
fn gpu_hidden() {
    let d = Device::Gpu;
}

const Q: char = '"';
