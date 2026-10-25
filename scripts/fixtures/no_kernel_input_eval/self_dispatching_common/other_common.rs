// Fixture: a scaffold that evaluates.
fn shared(a: &Array) -> Result<()> {
    a.eval()?;
    Ok(())
}
