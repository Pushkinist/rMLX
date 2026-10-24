pub fn f() {}
#[cfg(test)]
#[path = "foo_tests.rs"]
mod foo_tests;
#[cfg(test)]
mod tests;
