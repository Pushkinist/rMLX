# HTTP server: Metal claim file

Split out of [`SERVER.md`](SERVER.md) to keep that doc under the size cap.

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
