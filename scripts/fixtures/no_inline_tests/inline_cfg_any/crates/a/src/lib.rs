pub fn f() {}
#[cfg(any(test, feature = "x"))]
mod tests {
    fn t() {}
}
