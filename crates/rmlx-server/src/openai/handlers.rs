//! Route handlers for the OpenAI-compatible API.
//!
//! This file re-exports public items from the split submodules declared in
//! `openai/mod.rs`:
//! - `chat`       — POST /v1/chat/completions handler
//! - `generate`   — non-streaming and streaming generation paths
//! - `streaming`  — SSE streaming helpers (`StreamState`, `handle_streaming_token`)
//! - `lifecycle`  — model lifecycle routes (list/load/unload/status)

pub(crate) use super::chat::chat_completions;
pub(crate) use super::lifecycle::{list_models, load_model, model_status, unload_model};
