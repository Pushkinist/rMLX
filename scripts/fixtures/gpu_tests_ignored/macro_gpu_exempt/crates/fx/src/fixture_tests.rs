// Negative control: the per-test exemption marker works inside a macro body,
// so a device-as-value policy cell does not have to be #[ignore]d into
// never running.
//
// Note the blast radius this fixture pins. The body is ONE synthetic test, so
// the marker exempts every cell the macro generates — one comment line, all
// invocations. That is why a marker inside a macro body is reviewed against
// every invocation, not against one test.

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
