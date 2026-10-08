# Examples & Agent Integrations

This directory contains reference integrations wiring `sqlite-graph-memory` into different AI agent harnesses.

---

## Available Examples

| File | Harness / Protocol | Description |
| :--- | :--- | :--- |
| `sgm-mcp` ([source](../src/sqlite_graph_memory/mcp_server.py)) | **MCP (Model Context Protocol)** | Stdio MCP server exposing memory recall as an agent tool. Works with Cursor, Claude Desktop, Antigravity, Zed, and Windsurf. |
| [`mcp-config.json`](mcp-config.json) | **MCP Client Config** | Sample configuration block for `claude_desktop_config.json` or Cursor MCP settings. |
| [`claude-code-stop-hook.json`](claude-code-stop-hook.json) | **Claude Code Stop Hook** | Configuration snippet registering `turnstate_hook.py` as an automatic post-turn ledger hook. |

---

## 1. MCP Server Integration (`sgm-mcp`)

The MCP server exposes a single tool (`memory_recall`) that provides associative memory to any MCP-compliant agent.

### Prerequisites

1. Install requirements:
   ```bash
   pip install 'sqlite-graph-memory[embeddings]'
   ```
2. Build the embedding index over your markdown notes:
   ```bash
   sgm-index /path/to/your/notes
   ```

### Tool Definition

- **Tool Name:** `memory_recall`
- **Arguments:**
  - `query` (string, required): The search question or topic.
  - `mode` (string, optional, default: `"associative"`):
    - `"associative"`: Vector search + 1-hop `[[wikilink]]` graph expansion + cross-encoder reranking.
    - `"direct"`: Vector search + cross-encoder reranking (no graph expansion).
    - `"ab"`: Runs both pipelines, logs comparative telemetry to `turnstate.db`, and returns the comparison.

### MCP Client Setup

Add the server to your MCP client config (e.g. `claude_desktop_config.json` or Cursor `Settings > MCP`):

```json
{
  "mcpServers": {
    "sqlite-graph-memory": {
      "command": "python",
      "args": [
        "/absolute/path/to/sqlite-graph-memory/sgm-mcp"
      ],
      "env": {
        "BRAIN_INDEX_DIR": "/absolute/path/to/sqlite-graph-memory/index",
        "TURNSTATE_DB": "/absolute/path/to/sqlite-graph-memory/turnstate.db"
      }
    }
  }
}
```

### Direct CLI Verification

You can verify the MCP server directly from the command line without launching an MCP client:

```bash
# Test default associative recall
sgm-mcp --test "how do I think about agent memory"

# Test direct (vector-only) recall
sgm-mcp --test "how do I think about agent memory" --direct

# Test A/B recall mode
sgm-mcp --test "how do I think about agent memory" --ab
```

---

## 2. Claude Code Stop Hook (`claude-code-stop-hook.json`)

Registers `turnstate_hook.py` to run after every assistant turn in Claude Code. It extracts session context (asks, summaries, files touched, tools used, commands run, decisions made) and appends a row to `turnstate.db` at **0 token cost**.

To register, add the snippet from [`claude-code-stop-hook.json`](claude-code-stop-hook.json) to `~/.claude/settings.json`.

Inspect the recorded turns anytime:
```bash
sgm-turnstate-show --stats
sgm-turnstate-show --n 10
```

---

## Environment Variables Reference

All components share standard configuration variables:

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `BRAIN_INDEX_DIR` | `./index` | Directory where `index_notes.py` writes and `brain_ask.py` reads `_brain_e5.npy` and metadata. |
| `TURNSTATE_DB` | `./turnstate.db` | SQLite database holding per-turn ledger (`turns`) and A/B telemetry (`ab_recall`). |
| `BRAIN_ANSWER_OUT` | `./_brain_answer.txt` | File where the formatted context bundle or answer is mirrored. |
| `BRAIN_INDEX_HIDDEN` | `0` | If `1`, indexes hidden dot-directories too (by default skipped). |
| `BRAIN_AB_JSON` | `None` | If set, `--ab` mode writes a structured JSON diff to this file path. |
| `AB_SOURCE` | `work` | Telemetry label written to `ab_recall.source` (`work`, `eval`, etc.). |

