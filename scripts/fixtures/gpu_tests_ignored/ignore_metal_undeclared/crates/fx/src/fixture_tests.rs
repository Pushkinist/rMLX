// The fatal converse: an #[ignore] whose reason claims a Metal context, on a
// test the scanner can reach no device from, with no declared route.
//
// Such a test runs under NO gate — `make test` skips it because it is ignored,
// `make gpu-test` skips it because it is not classified — so it reads as
// covered at both while covering nothing. It was advisory for a long time and
// six tests accumulated in it, which is why the gate now refuses it.

#[ignore = "GPU Metal context"]
#[test]
fn orphaned_metal_test() {
    // The device lives behind an HTTP boundary, but nothing here says so.
    drive_the_server_over_http();
}
