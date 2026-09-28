// A reasonless #[ignore] on a test that reads an environment variable and
// reaches no Device::Gpu. `make test` skips it and `make gpu-test` never lists
// it, and with no reason text the Metal-claim rule cannot see it.

#[ignore]
#[test]
fn env_gated_cpu_cell() {
    let Ok(model) = std::env::var("RMLX_KV_TEST_MODEL") else {
        eprintln!("SKIP env_gated_cpu_cell: RMLX_KV_TEST_MODEL not set");
        return;
    };
    run(model, Device::Cpu);
}
