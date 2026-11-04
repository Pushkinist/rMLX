pub fn f() {}
#[cfg(all(test, feature = "x"))]
mod tests {
    fn t() {}
}
