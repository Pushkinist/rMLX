# Server shutdown

What `rmlx serve` does after SIGINT or SIGTERM, and which work stops when a
request is gone. Split out of [`SERVER.md`](SERVER.md) to keep that doc under
the size cap. The shutdown rule is in `crates/rmlx-server/src/shutdown.rs`, and
it is the same for every route and for every value of `--max-timeout-secs`,
`0` included.

## The rule

1. **First signal.** The server accepts no new connection. Each open request
   has 8 s to send its reply. A reply that ends in that time is complete: a
   transcription returns its text, a stream sends its terminal item.
2. **At 8 s, or at a second SIGINT or SIGTERM,** every request that is still
   open is dropped:
   - A request with no response yet gets 503 in the OpenAI envelope, type
     `service_unavailable`, message `the server is shutting down`, on every
     route, also while it waits for the GPU.
   - A reply that already streams is ended. The client reads a correct
     chunked body. A stream that had not sent its terminal item has none: no
     `[DONE]`, no `finish_reason`, no `message_stop` and no error event. A
     client that does not check for the terminal item reads a short answer as
     a complete one.
3. **Exit.** The process is gone within 10 s of the first signal, or
   2 s after the second signal when that is earlier. The exit code is 0.

With no request in flight the exit follows the first signal at once.

A serve error ends the process within 10 s of the first signal when it came
after one, else within 10 s of the error. A panic of the serve task ends the
process within 10 s of the panic.

## What stops, and what does not

The drop of a request is the only stop signal a blocking job gets. The 408 of
the request timeout and a closed connection drop a request as the shutdown
does, so this section is true for all three. A job that stops releases the GPU
permit when it returns, and the next request starts then. A job that is not
stopped holds the permit to its end; after a shutdown it ends with the
process.

- A transcription reads the stop before each 30 s window, so it stops when
  the window in flight ends ([`SERVER_AUDIO.md`](SERVER_AUDIO.md)). That
  window can take longer than the 2 s allowance: see "Worst-case latency" in
  [`AUDIO.md`](AUDIO.md).
- A chat or messages job with no draft model reads the stop before each
  decode step and before each chunk of a prefill, on every architecture. The
  step or the chunk in flight runs to its end. A prefill is the full prefill
  of a prompt, or the prefill of the tail after a prompt-cache prefix hit on
  Gemma 4 and after a prefix that the SSD tier restored on Qwen3.5.
- An image request on Gemma 4 and Gemma 3 reads the stop before the load of
  each image. An image request on Qwen3-VL also reads the stop before each
  block of its vision tower and before each chunk of its image prefill.
- A chat or messages job with a draft model (speculative decoding) reads the
  stop before each round and before each chunk of a prefill: the chunked
  prefill of the verifier and of a draft model, the capture prefill of the
  verifier, and the prefill of the EAGLE-3 drafter. A round that started runs
  to its end (draft, verify, accept, rollback), so the verifier and the
  drafter stop at a committed prefix. The forward of the last prompt token
  that seeds an MTP, DFlash or assistant drafter has no stop inside it.
- These parts of a chat or messages job have no stop inside them: the load of
  one image (a remote fetch takes up to 20 s, the timeout of the fetch), the
  vision tower of Gemma 4 and Gemma 3, the audio tower of Gemma 4, and the
  prefill of their image or audio prompt, which is one forward. They run to
  their end, and the job stops at its next read of the stop after them. The
  decode of an `input_audio` file stops before its next packet.
- A speech job reads the stop before each talker frame and before each pass
  of the codec decode. The frame or the pass in flight runs to its end. A
  pass decodes at most 300 frames plus 25 frames of left context, and has no
  read of the stop inside it.
- An embeddings job is not stopped.
- A model load is not stopped. The request that started it, and each request
  that waits for it, gets the 503. The load runs on and ends with the
  process.

### A chat job whose request is gone

A chat job decides that its request stopped from the result of the
generation, not from the flag. A generation that ran to its end is reported
as one that ended, also when its request went away after it: it keeps its
completed-request record and its rows, and the send of its terminal item to a
closed reply fails with no effect. A generation that stopped in its prefill,
before a decode step or at the head of a round returns `Error::Cancelled`. Its
job sends no terminal item and writes no row of a completed generation and no
completed-request record; with a draft model that includes the request record
of the round loop, and the round events of the rounds that ran stay. It logs
one `info` event in the span of the request, `the request is gone; the
generation stopped`. A stop in the load of an image or in the decode of an
`input_audio` file, before the first decode step, logs the same event with
`emitted` 0, and no `warn` or `error` event. The field `emitted` is the count
of tokens that were given to the step callback. A token that the last forward
picked is not counted on the paths where the stop drops it.

The rows of the first token stay. A route writes the time-to-first-token rows
when the first token comes, and they are a measurement of an event that
occurred.

A request that stops in its prefill stores no prompt-cache entry from that
prefill. On the Qwen3.5 family with the SSD tier, one entry can come from the
request all the same: the prefix that the request restored from the SSD tier
is put into a memory slot before the first read of the stop, and it can evict
the oldest slot. That entry is complete, and the stop does not cut it. A
request that stops in its decode leaves the entry that its whole prefill
stored ([`PROMPT_CACHE.md`](PROMPT_CACHE.md)). A job with a draft model
stores no prompt-cache entry, so its stop leaves none.

The engine returns the stop as `Error::Cancelled`. The retry envelope does not
replay it ([`SERVER.md`](SERVER.md) § "Classification"). A drop of the replay
stream aborts the task of the envelope at its next await; in a prefill that is
the read of the token stream. The abort drops the reply stream of the attempt,
which sets the stop flag of its job.

### A speech job whose request is gone

A speech job that stops returns the error `Cancelled` from the synthesis. The
job sends no reply body and writes no metrics row. It logs one `info` event in
the span of the request, `the request is gone; the generation stopped`, with
`emitted` 0, because no audio was delivered, and no `warn` or `error` event.
The MLX buffer cache is released once at the end of the frame loop on a stop,
as on every other end of the loop, and once more when a stop ends the codec
decode. A synthesis that ran to its end before its request went away is
reported as one that ended.

## The exit at the deadline

The server ends its runtime at the exit deadline. When the runtime has not
ended by then, the process leaves with work still in it: a job or a model
load that was not stopped, or a connection that is still open.
The server cannot tell these apart, so it treats each as a job that can
still use the GPU:

- The Metal claim stays held until the process is gone, and the kernel frees
  it then, as for a holder that was killed. The next start finds it free.
  When the runtime ended in time, the claim guard is dropped on the way out.
- The log writer is flushed before the process ends. Events that a job emits
  after that are lost.
- From the end of the runtime the process reads no SIGINT and no SIGTERM: the
  handlers stay installed and nothing listens. If the exit itself stalls,
  only SIGKILL ends the process and frees the claim.

No test holds the exit of a process that still has a job on the GPU. A run of
a real server with a model is its proof: exit code 0, the events below, and a
restart that is not refused.

## Log events

`since_signal_ms` is the time since the first signal. `left_ms` is the time to
the exit deadline when the runtime started to end, and `waited_ms` is the time
that end took.

| Level | Event | Fields |
|---|---|---|
| `warn` | `failed to install the signal handler` | `error`, `signal` |
| `info` | `received a shutdown signal` | `signal` |
| `info` | `shutdown started` | `gpu_permit_requests`, `reply_bound_ms`, `exit_bound_ms` |
| `info` | `every connection closed` | — |
| `warn` | `the open requests are dropped` | `reason` (`reply bound reached` or `second signal`), `gpu_permit_requests`, `since_signal_ms` |
| `warn` | `request answered 503 by the server shutdown` | `method`, `path` |
| `warn` | `streaming reply ended by the server shutdown; it can lack its terminal item` | `method`, `path` |
| `warn` | `exit deadline reached with a connection still open` | `since_signal_ms` |
| `info` | `runtime ended before the exit deadline` | `left_ms`, `waited_ms` |
| `warn` | `exit deadline reached before the runtime ended; a blocking job can still run` | `left_ms`, `waited_ms` |
| `info` | `Metal claim stays held until the process is gone` | — |
| `warn` | `Metal claim could not be held past its guard` | `error` |

`gpu_permit_requests` counts the requests that hold or wait for the GPU
permit: chat, messages, transcription and speech. An embeddings request is
not in it.

## Claim file

`claim::try_claim() -> Result<MetalClaim, ClaimError>` guards one
machine-wide Metal claim: an exclusive `flock` on `/var/tmp/rmlx.claim`.
`/var/tmp` is machine-wide and outlives macOS's `tmp_cleaner`, which only
sweeps `/tmp`.

1. It opens (or creates, mode `0666`) the file with `O_NOFOLLOW | O_CLOEXEC |
   O_NONBLOCK`, and refuses anything that is not a regular file with one
   link. A write open on another user's file gets `EACCES`, and it falls
   back to a read-only open.
2. It takes an exclusive, non-blocking `flock` on it — a read-only open
   takes the same `flock` as a writable one.
3. If the open was writable, it truncates the file and writes `<pid>
   <argv>` into it. A read-only claimer skips this step, so its body still
   names whichever process last held it open for writing.

If another process already holds the `flock`, `try_claim` returns
`ClaimError::AlreadyHeld { holder_pid, holder_command, path }`, read from the
body the holder wrote. The `flock` is the only gate — there is no PID
liveness probe. The kernel releases a dead holder's lock the moment it exits
or is killed; the next claimer locks the same file with no reclaim logic and
no operator action.

`MetalClaim` is a RAII guard. Dropping it releases the `flock`, unless a
child it passed the fd to (`rmlx claim run`) still runs — that child's own
copy of the fd keeps the lock held. The file itself is never deleted. `rmlx
serve` drops the guard at the end of its shutdown, unless a job still runs:
then the lock stays until the process is gone. `rmlx claim run -- <command>`
holds the claim for `<command>`'s whole life; see `docs/CLI.md`
§ "Claim file".

**The claim is machine-wide, not per port or per command.** Every GPU
command — `serve`, `chat`, `baseline`, `bench`, `kv-calibrate`,
`healthcheck --full`, and so on — takes the same claim, so only one can hold
the GPU at a time; a second is refused with exit code `11`, naming the
recorded holder. The lock is advisory and does not see non-rMLX MLX
processes such as `mlx_lm.server`; the `ClaimError` message prints a stop
hint (`kill <PID>`), never a delete hint.

**Legacy per-port claim (one release).** Older builds held
`/tmp/rmlx.<port>.claim`. Before it takes the machine-wide claim, this build
opens each such file and takes a brief shared lock on it, released at once —
enough to tell whether it is held, not to hold it. It skips a vanished
entry, a symlink, or anything that is not a regular file. Any other open
error there refuses every GPU command with an I/O error, not exit `11`.
Nothing is deleted. Removed in the release after next.
