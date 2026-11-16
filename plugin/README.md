# sqlite-graph-memory recall workflow plugin

This instructions-only plugin helps Claude use an **already configured**
`sqlite-graph-memory` MCP server. It does not install Python packages, build an
index, start a local server, or provide memory by itself.

## Prerequisite

Configure sqlite-graph-memory separately and build an index over your chosen
markdown folder using the [project setup guide](https://github.com/tonydzi/sqlite-graph-memory#readme).
The client must expose the server's `memory_recall` tool. If the tool is absent,
the skill explains the prerequisite instead of installing or launching anything.

## Install and use

Install from the marketplace in Claude Code 2.1.275 or later:

```text
/plugin install sqlite-graph-memory --marketplace tonydzi/sqlite-graph-memory
```
Ask Claude to recall a past decision or search linked markdown notes. The
`recall-notes` skill selects a query, reads the returned evidence, and cites it.
The MCP server must be configured independently before this works.

## Directory release 0.3.10

The directory package contains only Markdown instructions, JSON metadata and its
icon. The previous locked launcher is preserved outside this package in
[examples/locked-mcp](https://github.com/tonydzi/sqlite-graph-memory/tree/main/examples/locked-mcp).
The Python library remains at its own version; this is a plugin-only release.

No server or remote endpoint is declared by this plugin. Notes go only to the
server the user configured separately. See the project's
[privacy policy](https://github.com/tonydzi/sqlite-graph-memory/blob/main/PRIVACY.md)
for the engine's model downloads and local telemetry.
