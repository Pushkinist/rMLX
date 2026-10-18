# HTTP server: Anthropic routes

Split out of [`SERVER.md`](SERVER.md) to keep that doc under the size cap.

## `POST /v1/messages`

**Request fields**:

| Field | Type | Notes |
|---|---|---|
| `model` | string (required) | Registry model id. |
| `max_tokens` | u32 (required) | Missing is 400. A value above the room under the context ceiling ends the reply with `stop_reason:"max_tokens"` at the ceiling. |
| `messages` | array (required) | `role` and `content` (string or block array). See "Message blocks" below. |
| `system` | string or array | System prompt, rendered through the chat template. |
| `temperature`, `top_p`, `top_k` | number | Same resolution order as the OpenAI route. |
| `stop_sequences` | array | Stop sequences. |
| `stream` | bool | Default false. |
| `tools` | array | `name`, `description`, `input_schema`. |
| `tool_choice` | object | `{type:"auto"\|"any"\|"tool", name?:"…"}`. |
| `metadata` | object | Accepted and ignored. |

**Message blocks.** The route gives the chat template the blocks of every
message, with the values as the client sent them. It builds the same template
messages as the OpenAI route, so one history renders to the same prompt on both
routes.

| Block | Reaches the template as |
|---|---|
| `text` | `content`; the text blocks of a message join with nothing. |
| `thinking` | `reasoning_content` of the message. |
| `tool_use` | An entry of `tool_calls` (`id`, `name`, and `input` as the `arguments` object). |
| `tool_result` | A `tool` message with `tool_call_id` from `tool_use_id`. Its content is the string, or the joined `text` blocks of an array. |
| `input_audio` | `source:{type:"base64",data:…}` of a `user` message goes to the audio tower, as for the OpenAI route. It is not template text. |

A `tool_result` message comes before the message of its turn that holds the
other blocks, in block order. A turn of `tool_result` blocks alone gives no
`user` message.

These blocks have no template field and are not rendered: `signature`,
`redacted_thinking`, `is_error`, and the image blocks of a `tool_result`.
Image blocks are not read on this route.

Both input bounds read the rendered messages. The message cap (413, field
`messages`) counts them, so a message with many `tool_result` blocks counts
once per block. The size bound (`total_input_tokens_estimate`, 413) counts
every string of them: the text, the thinking text, the serialised `tool_use`
calls, the `tool_result` content and call id, the `system` prompt, and the
serialised `tools` array. A block that is not rendered does not count. The
OpenAI route counts the same strings of its messages, with `tool_call_id`,
`name` and the call arguments among them, and its `tools` and
`chat_template_kwargs`. Tools that the route drops because the template cannot
render them do not count.

**Non-streaming response**:

```json
{
  "id": "msg_<request-id>",
  "type": "message",
  "role": "assistant",
  "content": [
    {"type":"thinking","thinking":"…"},
    {"type":"text","text":"…"},
    {"type":"tool_use","id":"call_<hex>","name":"…","input":{…}}
  ],
  "model": "my-model",
  "stop_reason": "end_turn",
  "usage": {"input_tokens":42,"output_tokens":7}
}
```

The `thinking` block appears only when the model produced reasoning text, the
`tool_use` block only for a parsed tool call. `input` is a JSON object, not the
JSON string OpenAI puts in `arguments`. `usage.output_tokens` counts as
`completion_tokens` does, also in the `message_delta` event of a stream.

**Streaming** follows the Anthropic protocol: `message_start`, a `ping`, then
per block `content_block_start`, `content_block_delta` and
`content_block_stop`, then `message_delta` and `message_stop`. Thinking text
arrives in a `thinking` block with `thinking_delta` events, before the `text`
block. A tool call arrives as a `tool_use` block with `input_json_delta`.

## `stop_reason` mapping

`map_stop_reason` (`crates/rmlx-server/src/anthropic/route.rs`) maps the
engine's `finish_reason`:

| Engine `finish_reason` | Anthropic `stop_reason` |
|---|---|
| `"stop"` or none | `"end_turn"` |
| `"length"` | `"max_tokens"` |
| `"tool_calls"` | `"tool_use"` |
| anything else | `"error"` |

An unrecognised reason maps to `"error"`, never to `"end_turn"`.
`"stop_sequence"` comes only from the stop-matching path in `blocking.rs` and
`streaming.rs`, which bypasses `map_stop_reason`. A stream that dies mid-flight
ends with Anthropic's native `error` event and no `message_delta` or
`message_stop`:

```
event: error
data: {"type":"error","error":{"type":"service_unavailable_error","message":"…"}}
```

A `/v1/messages` stream with no `message_stop` did not complete.
