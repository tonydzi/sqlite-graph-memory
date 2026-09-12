# -*- coding: utf-8 -*-
"""Unit tests for examples/mcp_server.py (MCP Protocol and tool handling).

Runs offline with no model download or network.
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from mcp_server import handle_request, TOOLS, SERVER_INFO



def test_mcp_initialize():
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"}
        }
    }
    resp = handle_request(req)
    assert resp["id"] == 1
    assert resp["jsonrpc"] == "2.0"
    assert resp["result"]["serverInfo"]["name"] == "sqlite-graph-memory"
    assert "tools" in resp["result"]["capabilities"]


def test_mcp_ping():
    req = {"jsonrpc": "2.0", "id": 2, "method": "ping"}
    resp = handle_request(req)
    assert resp["id"] == 2
    assert resp["result"] == {}


def test_mcp_tools_list():
    req = {"jsonrpc": "2.0", "id": 3, "method": "tools/list"}
    resp = handle_request(req)
    assert resp["id"] == 3
    tools = resp["result"]["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "memory_recall"
    assert "query" in tools[0]["inputSchema"]["properties"]
    assert "mode" in tools[0]["inputSchema"]["properties"]
    assert "query" in tools[0]["inputSchema"]["required"]


def test_mcp_unknown_method():
    req = {"jsonrpc": "2.0", "id": 4, "method": "non_existent_method"}
    resp = handle_request(req)
    assert resp["id"] == 4
    assert resp["error"]["code"] == -32601


def test_mcp_notifications_no_response():
    req = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    resp = handle_request(req)
    assert resp is None


def test_mcp_tool_call_missing_query():
    req = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "memory_recall",
            "arguments": {}
        }
    }
    resp = handle_request(req)
    assert resp["id"] == 5
    assert resp["result"]["isError"] is True
    assert "required" in resp["result"]["content"][0]["text"].lower()


def test_mcp_tool_call_unknown_tool():
    req = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {
            "name": "unknown_tool",
            "arguments": {"query": "test"}
        }
    }
    resp = handle_request(req)
    assert resp["id"] == 6
    assert resp["error"]["code"] == -32601


def test_run_recall_success_with_scoped_answer_file(tmp_path, monkeypatch):
    """Verify run_recall passes a scoped BRAIN_ANSWER_OUT and reads it back cleanly."""
    stub = tmp_path / "stub_brain_ask.py"
    stub.write_text(
        "import os, sys\n"
        "out_path = os.environ.get('BRAIN_ANSWER_OUT')\n"
        "if out_path:\n"
        "    with open(out_path, 'w', encoding='utf-8') as f:\n"
        "        f.write('STUB CONTEXT BUNDLE FOR: ' + sys.argv[-1])\n",
        encoding="utf-8"
    )
    import mcp_server
    monkeypatch.setattr(mcp_server, "BRAIN_ASK_SCRIPT", stub)
    res = mcp_server.run_recall("test agent memory query", mode="associative")
    assert "STUB CONTEXT BUNDLE FOR: test agent memory query" in res


def test_run_recall_exit_code_error(tmp_path, monkeypatch):
    """Verify non-zero exit codes from brain_ask.py are captured and returned safely."""
    stub = tmp_path / "stub_fail.py"
    stub.write_text("import sys\nsys.stderr.write('SYNTHETIC_FAILURE_MSG\\n')\nsys.exit(1)\n", encoding="utf-8")
    import mcp_server
    monkeypatch.setattr(mcp_server, "BRAIN_ASK_SCRIPT", stub)
    res = mcp_server.run_recall("test query")
    assert "Recall error (exit code 1)" in res
    assert "SYNTHETIC_FAILURE_MSG" in res


def test_run_recall_stdout_fallback(tmp_path, monkeypatch):
    """Verify stdout is returned when no answer file is produced."""
    stub = tmp_path / "stub_stdout.py"
    stub.write_text("print('Direct stdout text from stub')\n", encoding="utf-8")
    import mcp_server
    monkeypatch.setattr(mcp_server, "BRAIN_ASK_SCRIPT", stub)
    res = mcp_server.run_recall("test query")
    assert "Direct stdout text from stub" in res


def test_run_recall_missing_script(tmp_path, monkeypatch):
    """Verify missing brain_ask.py script returns error rather than raising."""
    import mcp_server
    monkeypatch.setattr(mcp_server, "BRAIN_ASK_SCRIPT", tmp_path / "non_existent.py")
    res = mcp_server.run_recall("test query")
    assert "Error: brain_ask.py not found" in res


def test_run_recall_timeout(tmp_path, monkeypatch):
    """Verify subprocess timeout is caught and returns clear error text."""
    import mcp_server, subprocess
    def mock_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=kwargs.get("args") or [], timeout=60)
    monkeypatch.setattr(mcp_server.subprocess, "run", mock_run)
    res = mcp_server.run_recall("test query")
    assert "timed out after 60 seconds" in res


