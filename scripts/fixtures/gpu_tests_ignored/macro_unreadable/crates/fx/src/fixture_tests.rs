// The fn name is assembled rather than written, so no `fn <name>` line exists
// to classify. The declared #[test] outnumbers the readable items and the gate
// must fail closed instead of reporting a clean scan.

macro_rules! gpu_cell {
    ($name:ident) => {
        paste::paste! {
            #[ignore]
            #[test]
            fn [<gpu_ $name>]() {
                let device = Device::Gpu;
                run(device);
            }
        }
    };
}

gpu_cell!(a);
