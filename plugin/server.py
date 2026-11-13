# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "sqlite-graph-memory[lite]==0.3.8",
#   # onnxruntime 1.24+ ships no Intel Mac wheel
#   "onnxruntime<=1.23.2; sys_platform == 'darwin' and platform_machine == 'x86_64'",
# ]
# ///
"""Launcher for the sqlite-graph-memory MCP server inside the Claude Code plugin.

Started as `uv run --locked ${CLAUDE_PLUGIN_ROOT}/server.py`; `server.py.lock` next to
this file pins the package and every dependency by hash (Intel Macs are held at onnxruntime 1.23.2, the last release with a wheel for them). Maps the plugin's option
CLAUDE_PLUGIN_OPTION_INDEX_DIR to the server's BRAIN_INDEX_DIR and keeps recall
telemetry out of the user's project folder.
"""
import os
from pathlib import Path

idx = os.environ.get("CLAUDE_PLUGIN_OPTION_INDEX_DIR")
if idx and not os.environ.get("BRAIN_INDEX_DIR"):
    os.environ["BRAIN_INDEX_DIR"] = idx
if not os.environ.get("TURNSTATE_DB"):
    base = os.environ.get("CLAUDE_PLUGIN_DATA") or str(Path.home() / ".sqlite-graph-memory")
    Path(base).mkdir(parents=True, exist_ok=True)
    os.environ["TURNSTATE_DB"] = str(Path(base) / "turnstate.db")

from sqlite_graph_memory.mcp_server import main  # noqa: E402

if __name__ == "__main__":
    main()
