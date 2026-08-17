// An async test attribute is a test attribute. Matching only the bare
// `#[test]` spelling left every #[tokio::test] unclassified in both
// directions — never flagged, never listed — which is the same hole
// macro-generated tests were in. Both the bare and the parameterised
// spellings must classify.

#[tokio::test]
async fn tokio_gpu_no_ignore() {
    let device = Device::Gpu;
    run(device).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tokio_flavored_gpu_no_ignore() {
    let device = Device::Gpu;
    run(device).await;
}
