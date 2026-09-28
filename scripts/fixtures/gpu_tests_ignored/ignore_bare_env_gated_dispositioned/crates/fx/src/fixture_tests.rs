// The three shapes the reasonless-#[ignore] rule must leave alone: the same
// environment-gated cell with its reason written, the same cell on the GPU
// (classified, so the GPU suite lists and runs it), and a reasonless #[ignore]
// on a test that reads no environment variable, which the rule does not reach.

#[ignore = "placeholder: the live check is not written"]
#[test]
fn env_gated_cpu_cell_with_reason() {
    let Ok(model) = std::env::var_os("RMLX_KV_TEST_MODEL") else {
        return;
    };
    run(model, Device::Cpu);
}

#[ignore]
#[test]
fn env_gated_gpu_cell() {
    let Ok(model) = std::env::var("RMLX_KV_TEST_MODEL") else {
        eprintln!("SKIP env_gated_gpu_cell: RMLX_KV_TEST_MODEL not set");
        return;
    };
    run(model, Device::Gpu);
}

#[ignore]
#[test]
fn ungated_cpu_cell() {
    run_fixed(Device::Cpu);
}
