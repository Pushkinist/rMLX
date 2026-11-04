pub fn f() {}
#[cfg(test)]
#[path = "foo_tests.rs"]
pub(crate) mod tests;
