struct S<'é, const C: char>(&'é u8);
fn mk<'é>(_: S<'é, '"'>) {}

#[test]
fn gpu_hidden() {
    let d = Device::Gpu;
}

const Q: char = '"';
