// Fixture: shared scaffold with no MetalKernelInvoke of its own — in scope
// because its name ends in `_common.rs`.
pub(crate) fn materialise(a: &Array) -> Result<()> {
    a.eval()
}
