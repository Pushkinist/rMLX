# Server shutdown

What `rmlx serve` does after SIGINT or SIGTERM. Split out of
[`SERVER.md`](SERVER.md) to keep that doc under the size cap. The rule is in
`crates/rmlx-server/src/shutdown.rs`, and it is the same for every route and
for every value of `--max-timeout-secs`, `0` included.

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

The drop of a request is the only stop signal a blocking job gets.

- A transcription reads it before each 30 s window, so it stops when the
  window in flight ends ([`SERVER_AUDIO.md`](SERVER_AUDIO.md)).
- A chat, messages, embeddings or speech job is not stopped. It runs on and
  ends with the process.
- A request inside a model load gets no 503 and is not dropped: a load of the
  chat, messages or model-load route runs on its async worker, and nothing
  can interrupt it. Its connection closes when the process exits. While all
  four workers are inside such a load, the server reads no signal and no
  timer, so the bounds start to count when one load returns.

## The exit at the deadline

The server ends its runtime at the exit deadline. When the runtime has not
ended by then, the process leaves with work still in it: a job that was not
stopped, a request inside a model load, or a connection that is still open.
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
