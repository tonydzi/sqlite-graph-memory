# -*- coding: utf-8 -*-
"""Script-mode and working-directory behaviour of the MCP server.

Kept in its own file so it does not collide with edits to test_mcp_server.py.
Runs offline: no model, no index, no network.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from sqlite_graph_memory import __version__  # noqa: E402
from sqlite_graph_memory import mcp_server  # noqa: E402

SCRIPT = SRC / "sqlite_graph_memory" / "mcp_server.py"


def test_script_mode_starts_without_installed_package(tmp_path):
    """`python src/sqlite_graph_memory/mcp_server.py` must not need the package on sys.path.

    Python puts the script's own directory (not src/) on sys.path, so an
    absolute `from sqlite_graph_memory import ...` at import time crashed with
    ModuleNotFoundError before the server answered anything (issue #24).
    `-S` drops site-packages so an installed or editable copy cannot mask it.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    res = subprocess.run(
        [sys.executable, "-S", str(SCRIPT)],
        input=json.dumps(req) + "\n",
        capture_output=True, text=True, cwd=str(tmp_path), env=env, timeout=30,
    )
    assert res.returncode == 0, res.stderr
    resp = json.loads(res.stdout.strip().splitlines()[-1])
    assert resp["result"]["serverInfo"]["version"] == __version__


def test_recall_subprocess_inherits_server_cwd(tmp_path, monkeypatch):
    """Relative BRAIN_INDEX_DIR / TURNSTATE_DB must resolve against the user's cwd.

    On an installed wheel the package dir is site-packages/sqlite_graph_memory;
    running brain_ask there made `BRAIN_INDEX_DIR=index` miss the user's ./index.
    """
    work = tmp_path / "work"
    work.mkdir()
    stub = tmp_path / "stub_brain_ask.py"
    stub.write_text(
        "import os, pathlib\n"
        "pathlib.Path(os.environ['BRAIN_ANSWER_OUT']).write_text(os.getcwd(), encoding='utf-8')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mcp_server, "_brain_ask_cmd", lambda: [sys.executable, str(stub)])
    monkeypatch.chdir(work)
    text, is_error = mcp_server.run_recall("anything")
    assert is_error is False, text
    assert Path(text).resolve() == work.resolve()
