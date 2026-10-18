//! A collector of the WARN events a closure emits, for a test of a log line.

use std::fmt::Write as _;
use std::sync::{Arc, Mutex, PoisonError};

/// The events of level WARN emitted on this thread while `f` runs, each as its
/// fields in one string.
///
/// The subscriber is scoped to the thread and answers `always` for every
/// callsite, so a callsite another test hit first is not left disabled for this
/// one.
pub(crate) fn warnings_during(f: impl FnOnce()) -> Vec<String> {
    #[derive(Clone, Default)]
    struct Collect(Arc<Mutex<Vec<String>>>);

    struct Fields(String);

    impl tracing::field::Visit for Fields {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            let _ = write!(self.0, "{}={value:?} ", field.name());
        }
    }

    impl tracing::Subscriber for Collect {
        fn register_callsite(
            &self,
            _meta: &'static tracing::Metadata<'static>,
        ) -> tracing::subscriber::Interest {
            tracing::subscriber::Interest::always()
        }
        fn enabled(&self, _md: &tracing::Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }
        fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}
        fn event(&self, event: &tracing::Event<'_>) {
            if *event.metadata().level() == tracing::Level::WARN {
                let mut fields = Fields(String::new());
                event.record(&mut fields);
                self.0
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(fields.0);
            }
        }
        fn enter(&self, _span: &tracing::span::Id) {}
        fn exit(&self, _span: &tracing::span::Id) {}
    }

    let collect = Collect::default();
    tracing::subscriber::with_default(collect.clone(), f);
    let seen = collect.0.lock().unwrap_or_else(PoisonError::into_inner);
    seen.clone()
}
