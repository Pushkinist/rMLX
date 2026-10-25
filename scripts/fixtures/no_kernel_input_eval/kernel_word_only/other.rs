// Fixture: names the kernel type only.
use x::MetalKernel;
fn a(q: &Array) {
    q.eval()?;
}
