# Security

pagellm is an educational inference server. It has no TLS, authentication or rate limiting, and it binds to `127.0.0.1` by default. If you expose it, put it behind a reverse proxy that provides those.

Hardening that is in place:

- Every request field is type-checked and range-checked before it reaches the engine thread.
- Request bodies over 1 MiB are rejected with 413; `n` is capped at 16 and `stop` at 4 strings.
- An engine fault fails only the in-flight requests and the server keeps running.
- Model weights load with `allow_pickle=False`.
- The Docker image runs as a non-root user.

To report a vulnerability, open a private security advisory on the repository rather than a public issue.
