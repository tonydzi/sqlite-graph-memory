# Privacy policy: sqlite-graph-memory (library, MCP server, Claude Code plugin)

Last updated: 13 November 2026.

**Short version: everything stays on your computer.** There is no service behind this project, so
there is nothing we receive, store or share.

## What the software reads

- The markdown notes in the folder you index with `sgm-index`. Your notes may contain personal
  data (names, emails, addresses); the software reads them only to build a local search index.
- The question your agent passes to the `memory_recall` tool.

## What it writes, and where

- The index (`.npy`, `.pkl`, manifest) in the folder you choose (`BRAIN_INDEX_DIR`).
- An optional local SQLite file with recall telemetry (`TURNSTATE_DB`; the plugin keeps it in its
  own data folder, or in `~/.sqlite-graph-memory/` when Claude Code does not provide one). It records
  queries and which notes were returned. It never leaves your machine.
- Each MCP recall writes its answer to a temporary file in your system temp folder, reads it back and
  returns it to the agent.
- The optional Claude Code Stop hook writes a per-turn ledger to a local SQLite file you configure.

## What it sends over the network

- Nothing about you or your notes. The only network traffic is downloading the open embedding and
  reranker models from Hugging Face on first use, and Python packages when `uv`/`pip` installs them. The plugin
  turns off Hugging Face usage telemetry for those downloads (`HF_HUB_DISABLE_TELEMETRY=1`).
- The code makes no other network calls.

## Retention and deletion

We retain nothing because we receive nothing. To delete everything, delete the index folder and the
SQLite files named above.

## Children

The software is a developer tool and is not directed at children under 18.

## Contact

Questions: open an issue at https://github.com/tonydzi/sqlite-graph-memory/issues
