//! The in-flight gauge (`gpu_pending`) after every exit path of
//! `admit_request`: a request that waits for the permit and leaves, a closed
//! queue, a rejection, and a request that is admitted. A one-permit queue and
//! no model.

use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Waker};
use std::time::Duration;

use tokio::sync::Semaphore;

use super::{admit_request, Admission, GpuAdmission};

const SETTLE: Duration = Duration::from_secs(2);

fn gate() -> (Arc<Semaphore>, Arc<AtomicUsize>) {
    (Arc::new(Semaphore::new(1)), Arc::new(AtomicUsize::new(0)))
}

fn pending_now(pending: &AtomicUsize) -> usize {
    pending.load(Ordering::Acquire)
}

async fn hold(queue: &Arc<Semaphore>, pending: &Arc<AtomicUsize>) -> GpuAdmission {
    match admit_request(queue, pending, 64).await {
        Admission::Admitted { guard, .. } => guard,
        Admission::QueueFull => panic!("an idle queue admits"),
    }
}

/// Whether `admit_request` refuses the request at once. A request that is
/// counted and waits for the permit is a failure, not a hang.
async fn is_refused(queue: &Arc<Semaphore>, pending: &Arc<AtomicUsize>, depth: usize) -> bool {
    matches!(
        tokio::time::timeout(SETTLE, admit_request(queue, pending, depth)).await,
        Ok(Admission::QueueFull)
    )
}

/// Wait until `n` requests are counted: a spawned waiter has then reached the
/// permit wait.
async fn wait_for_pending(pending: &AtomicUsize, n: usize) {
    let deadline = tokio::time::Instant::now() + SETTLE;
    while pending_now(pending) != n {
        assert!(
            tokio::time::Instant::now() < deadline,
            "gauge did not reach {n}: it reads {}",
            pending_now(pending)
        );
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

/// The request timeout drops the request future while it waits for the permit.
#[tokio::test]
async fn a_waiter_that_times_out_gives_its_slot_back() {
    let (queue, pending) = gate();
    let holder = hold(&queue, &pending).await;

    let waited = tokio::time::timeout(
        Duration::from_millis(50),
        admit_request(&queue, &pending, 64),
    )
    .await;
    assert!(waited.is_err(), "the holder keeps the only permit");
    assert_eq!(pending_now(&pending), 1, "only the holder is counted");

    drop(holder);
    assert_eq!(pending_now(&pending), 0);
}

/// A client that closes its connection drops the handler task.
#[tokio::test]
async fn a_waiter_whose_task_is_aborted_gives_its_slot_back() {
    let (queue, pending) = gate();
    let holder = hold(&queue, &pending).await;

    let (q, p) = (Arc::clone(&queue), Arc::clone(&pending));
    let waiter = tokio::spawn(async move { admit_request(&q, &p, 64).await });
    wait_for_pending(&pending, 2).await;
    waiter.abort();
    assert!(waiter.await.is_err_and(|e| e.is_cancelled()));
    assert_eq!(pending_now(&pending), 1, "only the holder is counted");

    drop(holder);
    assert_eq!(pending_now(&pending), 0);
}

/// A panic in the task that owns the wait unwinds through the wait.
#[tokio::test]
async fn a_waiter_that_unwinds_gives_its_slot_back() {
    let (queue, pending) = gate();
    let holder = hold(&queue, &pending).await;

    let (q, p) = (Arc::clone(&queue), Arc::clone(&pending));
    let waiter = tokio::spawn(async move {
        tokio::select! {
            biased;
            admitted = admit_request(&q, &p, 64) => drop(admitted),
            () = std::future::ready(()) => panic!("scripted panic beside the permit wait"),
        }
    });
    assert!(waiter.await.is_err_and(|e| e.is_panic()));
    assert_eq!(pending_now(&pending), 1, "only the holder is counted");

    drop(holder);
    assert_eq!(pending_now(&pending), 0);
}

/// A closed queue rejects a new request and a request already waiting.
#[tokio::test]
async fn a_closed_queue_rejects_and_gives_every_slot_back() {
    let (queue, pending) = gate();
    let holder = hold(&queue, &pending).await;

    let (q, p) = (Arc::clone(&queue), Arc::clone(&pending));
    let waiter = tokio::spawn(async move { admit_request(&q, &p, 64).await });
    wait_for_pending(&pending, 2).await;

    queue.close();
    assert!(matches!(waiter.await, Ok(Admission::QueueFull)));
    assert_eq!(pending_now(&pending), 1, "only the holder is counted");
    assert!(
        is_refused(&queue, &pending, 64).await,
        "a closed queue refuses a new request"
    );
    assert_eq!(pending_now(&pending), 1, "only the holder is counted");

    drop(holder);
    assert_eq!(pending_now(&pending), 0);
}

/// A request that waits is counted while it waits, the depth bound sees it,
/// and it is counted once after it gets the permit.
#[tokio::test]
async fn a_waiting_request_is_counted_once_from_wait_to_release() {
    let (queue, pending) = gate();
    let holder = hold(&queue, &pending).await;

    let (q, p) = (Arc::clone(&queue), Arc::clone(&pending));
    let waiter = tokio::spawn(async move { admit_request(&q, &p, 2).await });
    wait_for_pending(&pending, 2).await;

    assert!(
        is_refused(&queue, &pending, 2).await,
        "the depth bound counts the waiter"
    );
    assert_eq!(pending_now(&pending), 2, "a rejection counts nothing");

    drop(holder);
    let Ok(Admission::Admitted { guard, depth, .. }) = waiter.await else {
        panic!("the waiter takes the permit the holder released");
    };
    assert_eq!(depth, 2, "depth counts the holder it waited behind");
    assert_eq!(
        pending_now(&pending),
        1,
        "the admitted waiter is counted once"
    );

    drop(guard);
    assert_eq!(pending_now(&pending), 0);
}

/// Sets the flag when its thread unwinds, so the other threads stop.
struct StopOnPanic<'a>(&'a AtomicBool);

impl Drop for StopOnPanic<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.store(true, Ordering::Release);
        }
    }
}

/// Requests that arrive together one place below the depth bound: the count
/// must not pass the bound. In every round, all threads poll their request
/// once, spinning on one start flag, and the counted request stays pending on
/// the permit. A thread that panics stops the others and fails the test.
#[tokio::test]
async fn requests_that_arrive_together_do_not_pass_the_depth_bound() {
    const ROUNDS: usize = 500;
    const THREADS: usize = 2;
    const DEPTH: usize = 2;

    let mut gates = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let (queue, pending) = gate();
        let holder = hold(&queue, &pending).await;
        gates.push((queue, pending, holder));
    }
    let started = AtomicUsize::new(0);
    let polled = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);

    let waiting: Vec<(usize, _)> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..THREADS)
            .map(|_| {
                scope.spawn(|| {
                    let _stop = StopOnPanic(&failed);
                    let mut waiting = Vec::new();
                    for (round, (queue, pending, _)) in gates.iter().enumerate() {
                        let mut request = Box::pin(admit_request(queue, pending, DEPTH));
                        while started.load(Ordering::Acquire) <= round {
                            if failed.load(Ordering::Acquire) {
                                return waiting;
                            }
                            std::hint::spin_loop();
                        }
                        let mut cx = Context::from_waker(Waker::noop());
                        if request.as_mut().poll(&mut cx).is_pending() {
                            waiting.push((round, request));
                        }
                        polled.fetch_add(1, Ordering::AcqRel);
                    }
                    waiting
                })
            })
            .collect();
        'rounds: for round in 0..ROUNDS {
            started.store(round + 1, Ordering::Release);
            while polled.load(Ordering::Acquire) < (round + 1) * THREADS {
                if failed.load(Ordering::Acquire) {
                    break 'rounds;
                }
                std::hint::spin_loop();
            }
        }
        let mut waiting = Vec::new();
        let mut panicked = false;
        for worker in workers {
            match worker.join() {
                Ok(requests) => waiting.extend(requests),
                Err(_) => panicked = true,
            }
        }
        assert!(!panicked, "a request thread panicked");
        waiting
    });

    let mut waiters = vec![0_usize; ROUNDS];
    for (round, _) in &waiting {
        waiters[*round] += 1;
    }
    for (round, (_, pending, _)) in gates.iter().enumerate() {
        assert_eq!(
            pending_now(pending),
            1 + waiters[round],
            "round {round}: the count is the holder and the waiting requests"
        );
        assert!(
            pending_now(pending) <= DEPTH,
            "round {round}: {} requests are counted, the bound is {DEPTH}",
            pending_now(pending)
        );
    }
}
