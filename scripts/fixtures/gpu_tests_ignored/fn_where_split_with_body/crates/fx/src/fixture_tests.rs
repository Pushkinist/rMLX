// A fn with a body whose signature and `where` clause run over several lines. Its lines end in
// `,` until the `{`, so it is not a declaration: its body's Device::Gpu is seen, and the
// un-ignored test that calls it is named.

fn gpu_helper<T>(x: T)
where
    T: Copy,
{
    let device = Device::Gpu;
    run(x, device);
}

#[test]
fn calls_the_where_split_helper() {
    gpu_helper(1);
}
