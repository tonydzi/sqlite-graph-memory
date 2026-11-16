# Optional locked local MCP launcher

Preserved from plugin 0.3.8; excluded from the instructions-only directory package.
Requires uv and a separately built index. This is a manual local setup example,
not a server installed or launched by the directory plugin.

Set `CLAUDE_PLUGIN_OPTION_INDEX_DIR` to the index directory (or set
`BRAIN_INDEX_DIR`) and run `uv run --locked server.py` from this folder through
your MCP client's separately configured command. The adjacent lockfile pins the
runtime. The old launcher also accepts `CLAUDE_PLUGIN_DATA` for telemetry storage.
