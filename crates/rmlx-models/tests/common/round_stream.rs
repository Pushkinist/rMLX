//! Capturing a speculative round stream without changing the run.
//!
//! A subscriber a round-loop test installs around its speculative arm and
//! nothing wider, and the shape of one captured event. Split out of the golden
//! harness because it shares none of its model resolution and only its
//! `#![allow]` header.

/// At TRACE, every round forces its carried arrays before its span closes.
pub const PHASE_SWITCH_TARGET: &str = "rmlx::spec::phase";

/// At TRACE, EAGLE-3's round adds one event per verified position. The same
/// target carries that loop's round event at DEBUG.
pub const EAGLE3_STEP_SWITCH_TARGET: &str = "rmlx_models::speculative::eagle3";

/// The two targets a **speculative round loop** consults to decide what it
/// *does*, as opposed to what it logs. A recorder that enables either is
/// measuring a different, slower run than the one that ships.
///
/// Both are asked at TRACE and at no other level, and both targets carry a
/// round event at DEBUG, so what a capture must decline is the level and not
/// the target.
///
/// **Not every behaviour-changing switch in the crate.** [`RoundStreamRecorder`]
/// answers `true` for every DEBUG callsite, so it turns on any switch keyed on
/// DEBUG — `multimodal_cache`'s `enabled!(Level::DEBUG)` is one, and it is a
/// switch no round loop reaches, so a speculative capture is unaffected by it.
/// A capture installed over any other path has to read that list for itself.
pub const BEHAVIOUR_SWITCH_TARGETS: [&str; 2] = [PHASE_SWITCH_TARGET, EAGLE3_STEP_SWITCH_TARGET];

/// One event as it was emitted: its target, its message, and its fields in the
/// order the emitter wrote them, each rendered by the emitter's own `Debug`.
#[derive(Clone, Debug)]
pub struct CapturedEvent {
    pub target: String,
    pub message: String,
    pub fields: Vec<(String, String)>,
}

impl CapturedEvent {
    /// One field's rendered value.
    #[must_use]
    pub fn field(&self, name: &str) -> Option<&str> {
        self.fields
            .iter()
            .find(|(f, _)| f == name)
            .map(|(_, v)| v.as_str())
    }

    /// The event as one JSON object — `target`, `message`, then every field
    /// under its own name.
    ///
    /// Values stay strings: `tracing`'s typed visits all fall through to
    /// `record_debug`, so a rendered value is the only form every field has,
    /// and re-typing some of them here would make two events comparable only
    /// through this function's guesses.
    #[must_use]
    pub fn json_line(&self) -> String {
        self.render(false)
    }

    /// The same object with every wall-clock field dropped.
    ///
    /// The shared round event carries five of them and they move on every run,
    /// so a digest over the whole line compares two machines' load rather than
    /// two engines. This is the form two runs are held to, and the suffix is
    /// the rule: a timing field is named `*_ms`.
    #[must_use]
    pub fn stable_json_line(&self) -> String {
        self.render(true)
    }

    fn render(&self, drop_timings: bool) -> String {
        let mut obj = serde_json::Map::new();
        obj.insert("target".to_owned(), self.target.clone().into());
        obj.insert("message".to_owned(), self.message.clone().into());
        for (name, value) in &self.fields {
            if drop_timings && name.ends_with("_ms") {
                continue;
            }
            obj.insert(name.clone(), value.clone().into());
        }
        serde_json::Value::Object(obj).to_string()
    }
}

/// What every round event carries, whichever of the five spellings emitted it:
/// the round's index, what it accepted, and how many proposals it accepted them
/// from.
pub const ROUND_EVENT_FIELDS: [&str; 3] = ["round", "accept", "num_draft"];

/// The events a round loop closes a round with, out of everything a run emits.
///
/// Read by shape rather than by message: four of the five spellings carry a
/// different set beyond [`ROUND_EVENT_FIELDS`]. A loop that renamed its message
/// is still found; a loop that stopped reporting one of the three is not a
/// round event any more, which is the answer the caller wants.
#[must_use]
pub fn round_events(events: &[CapturedEvent]) -> Vec<CapturedEvent> {
    events
        .iter()
        .filter(|e| ROUND_EVENT_FIELDS.iter().all(|f| e.field(f).is_some()))
        .cloned()
        .collect()
}

/// A subscriber that keeps every event's target, message and rendered fields,
/// and declines the two switches above.
///
/// Install it with [`tracing::subscriber::with_default`] and **never** with
/// `set_global_default`: a global stays installed for every later test in the
/// binary, and one whose `enabled` answers `true` leaves `phases_charged()` true
/// for all of them.
pub struct RoundStreamRecorder {
    events: std::sync::Mutex<Vec<CapturedEvent>>,
    asked: std::sync::Mutex<Vec<(String, tracing::Level, bool)>>,
}

impl Default for RoundStreamRecorder {
    fn default() -> Self {
        Self {
            events: std::sync::Mutex::new(Vec::new()),
            asked: std::sync::Mutex::new(Vec::new()),
        }
    }
}

impl RoundStreamRecorder {
    #[must_use]
    pub fn new() -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self::default())
    }

    /// Every event this recorder accepted, in order.
    #[must_use]
    pub fn events(&self) -> Vec<CapturedEvent> {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Every `(target, level, answer)` it was asked about.
    #[must_use]
    pub fn questions(&self) -> Vec<(String, tracing::Level, bool)> {
        self.asked
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Whether both behaviour switches were consulted and answered `false`.
    ///
    /// A capture that never saw the questions proves nothing about having
    /// declined them, which is why this reads the questions rather than the run
    /// — and it reads the answer given, not the rule that should have produced
    /// it.
    #[must_use]
    pub fn declined_both_switches(&self) -> bool {
        let asked = self.questions();
        BEHAVIOUR_SWITCH_TARGETS.iter().all(|t| {
            asked.iter().any(|(target, level, answer)| {
                target == t && *level == tracing::Level::TRACE && !answer
            })
        })
    }
}

impl tracing::Subscriber for RoundStreamRecorder {
    fn register_callsite(
        &self,
        _: &'static tracing::Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        tracing::subscriber::Interest::sometimes()
    }

    fn enabled(&self, meta: &tracing::Metadata<'_>) -> bool {
        // Both switches are asked at TRACE and TRACE alone, and both switch
        // targets also carry a round event at DEBUG — the shared one and
        // EAGLE-3's. Declining the target rather than the level would leave
        // four of the seven loops' rounds uncaptured while changing neither
        // switch's answer.
        let answer = *meta.level() <= tracing::Level::DEBUG;
        self.asked
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((meta.target().to_owned(), *meta.level(), answer));
        answer
    }

    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        struct Render {
            message: String,
            fields: Vec<(String, String)>,
        }
        impl tracing::field::Visit for Render {
            fn record_debug(&mut self, f: &tracing::field::Field, v: &dyn std::fmt::Debug) {
                let rendered = format!("{v:?}");
                if f.name() == "message" {
                    // The literal the emitter closed the event with, which
                    // `tracing` carries as a field like any other.
                    self.message = rendered;
                } else {
                    self.fields.push((f.name().to_owned(), rendered));
                }
            }
        }
        let mut r = Render {
            message: String::new(),
            fields: Vec::new(),
        };
        event.record(&mut r);
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(CapturedEvent {
                target: event.metadata().target().to_owned(),
                message: r.message,
                fields: r.fields,
            });
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}
