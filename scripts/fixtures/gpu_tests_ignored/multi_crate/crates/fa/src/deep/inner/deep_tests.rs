macro_rules! gpu_cell {
    ($name:ident, $n:expr) => {
        #[ignore = "GPU Metal context"]
        #[test]
        fn $name() {
            let device = Device::Gpu;
            run($n, device);
        }
    };
}

gpu_cell!(cell_a, 1);

#[ignore = "GPU Metal context"]
#[test]
fn nested_deep_gpu() {
    let device = Device::Gpu;
    run(0, device);
}

// gpu-test-gate: exempt
#[test]
fn exempt_policy_value() {
    pick(Device::Gpu);
}

// gpu-test-gate: exempt
#[ignore = "GPU Metal context"]
#[test]
fn exempt_but_ignored() {
    pick(Device::Gpu);
}

#[test]
fn cpu_only() {
    run(0, Device::Cpu);
}
