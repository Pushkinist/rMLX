//! A collector of the events a closure emits, for a test of a log line.

use std::cell::RefCell;
use std::fmt::Write as _;

/// One event as its fields, in the order the callsite names them, each as its
/// name and its `Debug` text. The message is the field named `message`.
pub(crate) type EventFields = Vec<(&'static str, String)>;

/// One event: its level and its fields.
#[derive(Clone, Debug)]
pub(crate) struct Event {
    pub(crate) level: tracing::Level,
    pub(crate) fields: EventFields,
}

thread_local! {
    /// The events this thread emits while a collection is open.
    static SINK: RefCell<Option<Vec<Event>>> = const { RefCell::new(None) };
}

struct Fields(EventFields);

impl tracing::field::Visit for Fields {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0.push((field.name(), format!("{value:?}")));
    }
}

/// The process-wide subscriber of the tests of this crate. It hands an event to
/// the thread that emitted it, if that thread has a collection open, and
/// `enabled` is true only for a thread whose sink is armed and for DEBUG or a
/// more severe level, so the engine's opt-in `tracing::enabled!` switches stay
/// off in a test that captures nothing.
///
/// One subscriber is installed for the whole process because per-test default
/// subscribers lost events under parallel tests.
struct Tee;

impl tracing::Subscriber for Tee {
    fn register_callsite(
        &self,
        _meta: &'static tracing::Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        tracing::subscriber::Interest::sometimes()
    }
    fn enabled(&self, md: &tracing::Metadata<'_>) -> bool {
        *md.level() <= tracing::Level::DEBUG && SINK.with(|sink| sink.borrow().is_some())
    }
    fn max_level_hint(&self) -> Option<tracing::level_filters::LevelFilter> {
        Some(tracing::level_filters::LevelFilter::DEBUG)
    }
    fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        SINK.with(|sink| {
            if let Some(out) = sink.borrow_mut().as_mut() {
                let mut fields = Fields(Vec::new());
                event.record(&mut fields);
                out.push(Event {
                    level: *event.metadata().level(),
                    fields: fields.0,
                });
            }
        });
    }
    fn enter(&self, _span: &tracing::span::Id) {}
    fn exit(&self, _span: &tracing::span::Id) {}
}

/// The events, at every level, that `f` emits on this thread.
///
/// Installs [`Tee`] as the process-wide subscriber on first use, and never
/// removes it. A test that needs another subscriber on the whole process cannot
/// have it in this binary.
pub(crate) fn events_during(f: impl FnOnce()) -> Vec<Event> {
    static INSTALLED: std::sync::Once = std::sync::Once::new();
    INSTALLED.call_once(|| {
        // Another global subscriber would leave every collection empty, and the
        // test that reads an empty one fails.
        let _ = tracing::subscriber::set_global_default(Tee);
    });
    SINK.with(|sink| *sink.borrow_mut() = Some(Vec::new()));
    f();
    SINK.with(|sink| sink.borrow_mut().take().unwrap_or_default())
}

/// The `Debug` text of the field `name` of an event, if it has one.
pub(crate) fn field<'e>(fields: &'e EventFields, name: &str) -> Option<&'e str> {
    fields
        .iter()
        .find_map(|(n, value)| (*n == name).then_some(value.as_str()))
}

/// The one event of `events`; the panic names how many there were.
pub(crate) fn only(events: Vec<EventFields>) -> EventFields {
    match <[EventFields; 1]>::try_from(events) {
        Ok([event]) => event,
        Err(events) => panic!("expected one event, found {}", events.len()),
    }
}

/// The events of level INFO that `f` emits on this thread with exactly this
/// message.
pub(crate) fn info_events_during(message: &str, f: impl FnOnce()) -> Vec<EventFields> {
    events_during(f)
        .into_iter()
        .filter(|event| event.level == tracing::Level::INFO)
        .map(|event| event.fields)
        .filter(|fields| field(fields, "message") == Some(message))
        .collect()
}

/// The events of level WARN emitted on this thread while `f` runs, each as its
/// fields in one string.
pub(crate) fn warnings_during(f: impl FnOnce()) -> Vec<String> {
    events_during(f)
        .into_iter()
        .filter(|event| event.level == tracing::Level::WARN)
        .map(|event| {
            event
                .fields
                .into_iter()
                .fold(String::new(), |mut line, (name, value)| {
                    let _ = write!(line, "{name}={value} ");
                    line
                })
        })
        .collect()
}
