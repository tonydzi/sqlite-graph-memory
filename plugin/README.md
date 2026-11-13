# sqlite-graph-memory plugin for Claude Code

Recall over your markdown notes as a local stdio MCP server: vector search finds entry points,
your `[[wikilinks]]` are the graph. No graph database, no server to run.

## Install

1. Install [uv](https://docs.astral.sh/uv/).
2. Build the index once over any folder of markdown notes:

   ```bash
   BRAIN_INDEX_DIR=$HOME/.sgm-index uvx --python 3.12 --from 'sqlite-graph-memory[lite]' sgm-index ~/notes
   ```

3. Install the plugin and give it that index folder when asked:

   ```
   /plugin install sqlite-graph-memory --marketplace tonydzi/sqlite-graph-memory
   ```

The server exposes one tool, `memory_recall`. It is read-only: the agent recalls, you write the notes.
The runtime is pinned by `server.py.lock` in this folder (`uv run --locked server.py`).

Project, docs and issues: https://github.com/tonydzi/sqlite-graph-memory
