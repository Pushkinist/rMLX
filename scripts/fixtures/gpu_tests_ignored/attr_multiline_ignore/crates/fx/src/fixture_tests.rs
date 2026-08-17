// The multi-line `#[ignore = "... \` spelling this gate's own remediation text
// recommends. Its continuation line ends in `"]`, never in the `)]` that a
// wrapped `#[cfg(..)]` produces, so a continuation rule keyed only on `)]`
// leaves the attribute capture latched for the REST OF THE FILE: every later
// item is swallowed into the attribute block, nothing is classified, and the
// gate reports a clean scan. The violation below is what that silence hides.

#[ignore = "GPU Metal context — run in isolation: \
            cargo test probe -- --ignored --test-threads=1"]
#[test]
fn probe() {
    let device = Device::Gpu;
    run(device);
}

#[test]
fn later_plain_gpu_no_ignore() {
    let device = Device::Gpu;
    run(device);
}
