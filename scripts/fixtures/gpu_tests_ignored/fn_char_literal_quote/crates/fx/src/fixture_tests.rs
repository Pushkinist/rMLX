// The `"` in b'"' is a char-literal payload, not a string delimiter. A scan
// that only tracks double quotes flips into "inside a string" here and never
// flips back, so the trailing comment is not stripped, the line's last
// significant character becomes comment prose, and this self-contained fn
// latches and swallows the violation below.
//
// The byte literal is already live in the scanned tree (8 occurrences across
// four files under crates/rmlx-server and crates/rmlx-cli); it has simply not
// landed on a decision line yet. Failure is silent: no U, no warning, exit 0.

#[ignore = "GPU Metal context"]
#[test]
fn probe() { let d = Device::Gpu; scan(d, b'"'); } // quoted byte literal

// The escaped spelling desynchronises the same way, one character further in:
// the `'` is stepped over, then the bare `\` does not guard the `"` behind it.
#[ignore = "GPU Metal context"]
#[test]
fn probe_escaped() { let d = Device::Gpu; scan(d, '\"'); } // escaped quote literal

// A payload longer than one escape — `\x1b`, `\u{FFFD}`, or any non-ASCII char
// — matches neither the one-char nor the one-escape shape. A scanner that
// steps only PART of a literal leaves its closing quote behind, re-reads that
// quote as an opener, and from there swallows a real `"`. So the whole literal
// has to be consumed, every payload form included.
#[ignore = "GPU Metal context"]
#[test]
fn probe_long_payloads() { let d = Device::Gpu; scan(d, '\x1b', '\u{FFFD}', 'é', '"'); } // tail

#[test]
fn later_plain_gpu_no_ignore() {
    let device = Device::Gpu;
    run(device);
}
