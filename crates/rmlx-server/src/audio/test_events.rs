//! Capture of the events that the code under test emits on this thread, for
//! the tests of the audio routes.

use std::sync::Arc;

/// The fields of every event that this thread emits while it is installed.
/// Each event holds its level, its own fields, and then the fields of each
/// span it is in, each named `span.<field>`.
#[derive(Default)]
pub(crate) struct Events {
    pub(crate) captured: std::sync::Mutex<Vec<Vec<(String, String)>>>,
    /// The fields of each span, at the index of its id less one.
    spans: std::sync::Mutex<Vec<Vec<(String, String)>>>,
    /// The ids of the spans this thread is in, innermost last.
    entered: std::sync::Mutex<Vec<u64>>,
}

struct Fields(Vec<(String, String)>);

impl tracing::field::Visit for Fields {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0.push((field.name().to_owned(), format!("{value:?}")));
    }
}

impl tracing::Subscriber for Events {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, attributes: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        let mut fields = Fields(Vec::new());
        attributes.record(&mut fields);
        let mut spans = self.spans.lock().unwrap();
        spans.push(fields.0);
        tracing::span::Id::from_u64(spans.len() as u64)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        let mut fields = Fields(vec![(
            "level".to_owned(),
            event.metadata().level().to_string(),
        )]);
        event.record(&mut fields);
        let spans = self.spans.lock().unwrap();
        for id in self.entered.lock().unwrap().iter() {
            let index = usize::try_from(id.saturating_sub(1)).unwrap_or(usize::MAX);
            for (key, value) in spans.get(index).into_iter().flatten() {
                fields.0.push((format!("span.{key}"), value.clone()));
            }
        }
        self.captured.lock().unwrap().push(fields.0);
    }
    fn enter(&self, id: &tracing::span::Id) {
        self.entered.lock().unwrap().push(id.into_u64());
    }
    fn exit(&self, id: &tracing::span::Id) {
        let mut entered = self.entered.lock().unwrap();
        if let Some(at) = entered
            .iter()
            .rposition(|entered| *entered == id.into_u64())
        {
            entered.remove(at);
        }
    }
}

/// Capture the events of this thread until the returned guards drop.
pub(crate) fn capture() -> (
    Arc<Events>,
    (tracing::Dispatch, tracing::subscriber::DefaultGuard),
) {
    let events = Arc::new(Events::default());
    // A second live dispatcher keeps a callsite that another thread registers
    // now from being cached as always on, which would bypass the scoped one.
    let keep = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let guard = tracing::subscriber::set_default(Arc::clone(&events));
    (events, (keep, guard))
}

/// The events of `f`, each as its level and its fields.
pub(crate) fn events_of(f: impl FnOnce()) -> Vec<Vec<(String, String)>> {
    let (events, guards) = capture();
    f();
    drop(guards);
    let captured = events.captured.lock().unwrap().clone();
    captured
}

/// The events at warn or error among `events`.
pub(super) fn problems(events: &Events) -> Vec<Vec<(String, String)>> {
    let all = events.captured.lock().unwrap();
    all.iter()
        .filter(|e| matches!(field(e, "level"), Some("WARN" | "ERROR")))
        .cloned()
        .collect()
}

/// The value of the field `name` of `event`.
pub(crate) fn field<'a>(event: &'a [(String, String)], name: &str) -> Option<&'a str> {
    event
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.as_str())
}
