# Security

sqlite-graph-memory runs locally: it reads your markdown notes, writes one SQLite
file next to them and serves an MCP endpoint over stdio. It makes no network
calls of its own; the embedding and reranker models are downloaded by
sentence-transformers on first use.

## Supported versions

Only the latest release on PyPI receives fixes.

## Reporting a vulnerability

Open a private report through GitHub Security Advisories on this repository
(Security tab, "Report a vulnerability"), or e-mail the maintainer listed in
`pyproject.toml`. Please include the version, the command you ran and what you
observed. You will get an acknowledgement within 7 days; fixes ship as a patch
release with a CHANGELOG entry that credits the reporter unless you ask otherwise.

Please do not open a public issue for anything that could expose other users'
notes or credentials.
