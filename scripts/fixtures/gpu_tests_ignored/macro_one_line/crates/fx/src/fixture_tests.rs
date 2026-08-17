// The entire macro is one line, so the opener consumes the `#[test]` and no
// `fn` line ever follows. Nothing can be classified from it, and a GPU test
// hides in the gap — so the opener fails closed instead.

macro_rules! m { ($n:ident) => { #[test] fn $n() { let d = Device::Gpu; run(d); } }; }
m!(cell_a);
