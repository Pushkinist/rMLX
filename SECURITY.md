# Security Policy

## Supported versions

rMLX is pre-1.0. Security fixes land on the latest released version only.

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |
| < 0.1.0 | ❌        |

## Reporting a vulnerability

**Do not open a public issue for security problems.**

Use GitHub's [private vulnerability reporting](https://github.com/Pushkinist/rMLX/security/advisories/new)
(Security → Report a vulnerability). That opens a private advisory only the
maintainers can see.

Please include:

- affected version / commit,
- a minimal reproduction (model arch / quant / request if relevant),
- impact and any known workaround.

You can expect an initial acknowledgement within a few days. Once a fix is
ready, we coordinate a release and credit the reporter (unless anonymity is
requested).

## Scope

rMLX is an Apple-Silicon-only, no-Python inference backend. In scope: the HTTP
server surface (OpenAI / Anthropic-compatible routes), model loading,
quantization codecs, and the FFI bridge. Out of scope: vulnerabilities in
upstream MLX / mlx-c, or in models you choose to load.
