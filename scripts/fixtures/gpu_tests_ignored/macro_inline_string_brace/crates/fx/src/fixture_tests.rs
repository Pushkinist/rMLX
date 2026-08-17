// A brace COUNT reads the `}` inside this string literal as a real closing
// brace, decides the line is unbalanced, latches the multi-line capture, and
// swallows the second macro's un-ignored GPU cell. One character in a string
// is the whole difference from a compliant file, and rustfmt leaves it
// byte-identical. The decision must read the line's last significant
// character instead.

macro_rules! gpu_cell {
    ($name:ident, $($arg:expr),*) => {
        #[ignore]
        #[test]
        fn $name() { let d = Device::Gpu; run(d, "}", $($arg),*); }
    };
}
gpu_cell!(cell_a, 1, 2);

macro_rules! second_cell {
    ($name:ident) => {
        #[test]
        fn $name() {
            let device = Device::Gpu;
            run(device);
        }
    };
}
second_cell!(cell_b);
