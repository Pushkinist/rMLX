//! Capture of the span and event fields that the code under test records.

/// The `(field, value)` pairs of every span and event that this thread opens
/// or emits while it is installed, including a value that a span records later.
#[derive(Default)]
struct Recorded {
    fields: std::sync::Mutex<Vec<(String, String)>>,
}

struct Collect<'a>(&'a mut Vec<(String, String)>);

impl tracing::field::Visit for Collect<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0.push((field.name().to_owned(), format!("{value:?}")));
    }
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.0.push((field.name().to_owned(), value.to_owned()));
    }
}

impl tracing::Subscriber for Recorded {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, attributes: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        attributes.record(&mut Collect(&mut self.fields.lock().unwrap()));
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, values: &tracing::span::Record<'_>) {
        values.record(&mut Collect(&mut self.fields.lock().unwrap()));
    }
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        let mut fields = self.fields.lock().unwrap();
        fields.push(("level".to_owned(), event.metadata().level().to_string()));
        event.record(&mut Collect(&mut fields));
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// The fields that spans and events record while `f` runs. An event's message
/// is the field named `message`, and its level is the field named `level`,
/// recorded first.
pub(crate) fn fields_recorded_by(f: impl FnOnce()) -> Vec<(String, String)> {
    let recorded = std::sync::Arc::new(Recorded::default());
    // A second live dispatcher keeps a callsite that another thread registers
    // now from being cached as always off, which would bypass the scoped one.
    let keep = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let guard = tracing::subscriber::set_default(std::sync::Arc::clone(&recorded));
    f();
    drop(guard);
    drop(keep);
    let fields = recorded.fields.lock().unwrap().clone();
    fields
}
