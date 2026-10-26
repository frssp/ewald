# Security

EWALD opens files, notebooks and `.ewd` bundles from other people and lets an AI model write
notebook cells, so it keeps untrusted content from running code: a notebook, an `.ewd` or text
inside a structure file cannot run code (or steer the AI into running it) until you trust it.
How that works is in [docs/developer/architecture.md](docs/developer/architecture.md) and
`src/ewald/trust.py`.

If you find a way around it, a leaked secret (API keys and SSH passwords belong in the OS
keychain only), or any other vulnerability, please report it privately through GitHub's
**Security › Report a vulnerability** on this repository rather than in a public issue.
Only the latest release and `main` get fixes.

Please keep the details out of public issues and pull requests until a fix is released.
