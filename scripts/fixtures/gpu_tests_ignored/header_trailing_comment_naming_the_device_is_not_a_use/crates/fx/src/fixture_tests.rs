#[test]
fn cpu_one() { run(); } // never touches Device::Gpu

#[test]
fn cpu_two() {
    run();
}
