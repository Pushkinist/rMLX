// Negative control: the per-test exemption marker works inside a macro body,
// so a device-as-value policy cell does not have to be #[ignore]d into
// never running.

macro_rules! policy_cell {
    ($name:ident) => {
        // gpu-test-gate: exempt
        #[test]
        fn $name() {
            assert!(is_metal(Device::Gpu));
        }
    };
}

policy_cell!(cell_a);
