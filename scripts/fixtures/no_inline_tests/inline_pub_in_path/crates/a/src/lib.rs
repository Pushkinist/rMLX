pub fn f() {}
#[cfg(test)]
pub(in crate::a) mod tests {
    fn t() {}
}
