# HTTP server: tool calling

Split out of [`SERVER.md`](SERVER.md) to keep that doc under the size cap.

## Tool Calling

### Parser architecture

`ToolCallStreamParser` parses tool calls out of the raw token stream. The
format is detected once at registry build from markers in
`chat_template.jinja`, with the architecture as the fallback, and cached in
`ModelEntry`.

| `ToolCallFormat` | Used by | Syntax |
|---|---|---|
| `Qwen3XmlFunction` | Qwen3.6 | `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>` |
| `Qwen3JsonToolCall` | `Qwen3ForCausalLM` (Bonsai) | `<tool_call>{"name":"…","arguments":{…}}</tool_call>` |
| `GemmaToolCall` | `Gemma4ForConditionalGeneration` | `<\|tool_call>call:NAME{key:val}<tool_call\|>` |

Gemma registers `<|tool_call>`, `<tool_call|>` and `<|"|>` as special tokens,
which `tokenizer.decode` strips. The engine rebuilds them from the token ids
before the parser sees them.

The parser is split-invariant: any BPE-aligned split of the stream parses the
same as the whole string. Several `<tool_call>` blocks may follow each other.

### Template support probe

At registry build, `probe_tools_supported` renders each template with one
tool and stores the result in `ModelEntry::tools_supported`. When it is
`false`, the request runs without tools instead of failing.

### Multi-turn tool loop

The client drives the loop. It sends `tools`; the model emits tool-call
blocks; the server returns them as `tool_calls` with
`finish_reason:"tool_calls"`; the client runs the tools and sends the results
as `tool` messages. The server renders the whole history through the chat
template each turn.

### `tool_choice=required` / `tool_choice=named` (constrained generation)

A `"required"` or named `tool_choice` engages the constraint engine to force
a valid call as bare JSON. `tool_choice_to_schema` builds the schema:

- **Named**, or **required with one tool**:
  `{"type":"object","properties":{"name":{"const":"<fn>"},"arguments":<fn-schema>},"required":["name","arguments"]}`.
- **Required with several tools**: `{"oneOf":[…]}`, one such branch per tool.

The `SchemaConstraint` runs with `EngagePolicy::Immediate`, so masking starts
at the first token. The output has no `<tool_call>` wrapper, so the marker
parser is bypassed: `bare_json_to_tool_call` turns the text into the
`tool_calls` envelope. Streaming buffers the JSON and emits one `tool_calls`
delta at the end.

A named tool that is not in `tools` builds no schema: the request runs as
`auto`. When the schema does not compile, or the model has no tokenizer file,
the forced call runs with no constraint, and the reply is read as for `auto`:

- A marked call is a tool call, in the content or in the reasoning. Text
  outside it is `content` or `reasoning_content`.
- When no call is marked, the whole content is tried as bare JSON.
- Bare JSON after prose, inside a code fence or on the reasoning channel is
  not a tool call: the reply is text. A `<tool_call>` literal that the
  reasoning opens and does not close hides a call that follows it.
- Streaming sends reasoning deltas as they arrive and holds the content to
  the end.

`stop` differs between the paths for a forced call, with or without the
constraint: streaming does not give the held content to the stop matcher; the
non-streamed path cuts the text first, then reads the call.

### EOF recovery

While tokens arrive, a partial call is not completed. At the end of the
generation, streaming or not, a Bonsai-style JSON call cut off mid-body (for
example at `max_tokens`) is repaired by closing its open strings and
brackets; a truncated Gemma call is dropped.

OpenAI `parameters` and Anthropic `input_schema` tools render identically.
