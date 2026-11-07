// Fixture: a name that only ends in common.rs is no scaffold.
fn shared(a: &Array) -> Result<()> {
    a.eval()?;
    Ok(())
}
