# -*- coding: utf-8 -*-
"""Unit tests for examples/mcp_server.py (MCP Protocol and tool handling).

Runs offline with no model download or network.
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from mcp_server import handle_request, TOOLS, SERVER_INFO
import mcp_server



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
    text, is_err = mcp_server.run_recall("test agent memory query", mode="associative")
    assert "STUB CONTEXT BUNDLE FOR: test agent memory query" in text
    assert is_err is False


def test_run_recall_exit_code_error(tmp_path, monkeypatch):
    """Verify non-zero exit codes from brain_ask.py are captured and returned safely."""
    stub = tmp_path / "stub_fail.py"
    stub.write_text("import sys\nsys.stderr.write('SYNTHETIC_FAILURE_MSG\\n')\nsys.exit(1)\n", encoding="utf-8")
    import mcp_server
    monkeypatch.setattr(mcp_server, "BRAIN_ASK_SCRIPT", stub)
    text, is_err = mcp_server.run_recall("test query")
    assert "Recall error (exit code 1)" in text
    assert "SYNTHETIC_FAILURE_MSG" in text
    assert is_err is True


def test_empty_answer_file_does_not_fallback_to_stdout(tmp_path, monkeypatch):
    """When the answer file is empty, stdout must not be returned and isError must be True.

    Regression guard for issue #9: stdout holds startup noise (load reports, tokenizer
    warnings) which must never be dressed up as a recall bundle with isError: false.
    """
    stub = tmp_path / "stub_noisy_empty.py"
    stub.write_text(
        "import sys\n"
        "sys.stdout.write('XLMRobertaModel LOAD REPORT: tokenizer warning noise\\n')\n",
        encoding="utf-8"
    )
    import mcp_server
    monkeypatch.setattr(mcp_server, "BRAIN_ASK_SCRIPT", stub)

    # 1. Direct function call test
    text, is_err = mcp_server.run_recall("test query")
    assert "LOAD REPORT" not in text
    assert text == "(no matching notes found)"
    assert is_err is True

    # 2. Protocol tool call test
    req = {
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {"name": "memory_recall", "arguments": {"query": "test query"}}
    }
    resp = mcp_server.handle_request(req)
    assert resp["id"] == 10
    assert resp["result"]["isError"] is True
    assert "LOAD REPORT" not in resp["result"]["content"][0]["text"]
    assert resp["result"]["content"][0]["text"] == "(no matching notes found)"


def test_run_recall_missing_script(tmp_path, monkeypatch):
    """Verify missing brain_ask.py script returns error rather than raising."""
    import mcp_server
    monkeypatch.setattr(mcp_server, "BRAIN_ASK_SCRIPT", tmp_path / "non_existent.py")
    text, is_err = mcp_server.run_recall("test query")
    assert "Error: brain_ask.py not found" in text
    assert is_err is True


def test_run_recall_timeout(tmp_path, monkeypatch):
    """Verify subprocess timeout is caught and returns clear error text."""
    import mcp_server, subprocess
    def mock_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=kwargs.get("args") or [], timeout=60)
    monkeypatch.setattr(mcp_server.subprocess, "run", mock_run)
    text, is_err = mcp_server.run_recall("test query")
    assert "timed out after 60 seconds" in text
    assert is_err is True


def test_run_recall_answer_file_is_unique_per_call_and_cleaned_up(tmp_path, monkeypatch):
    """The scoped answer file must differ between calls and must not survive them.

    Regression guard for the shared-answer-file race. The unit tests that shipped with
    the scoping fix all pass against a single fixed path too, so none of them actually
    pinned the behaviour; this one fails if the path stops being per-call or is left
    behind on disk.
    """
    recorder = tmp_path / "paths.log"
    stub = tmp_path / "stub_record.py"
    stub.write_text(
        "\n".join(
            [
                "import os",
                "out = os.environ['BRAIN_ANSWER_OUT']",
                "open(r'{rec}', 'a', encoding='utf-8').write(out + chr(10))",
                "open(out, 'w', encoding='utf-8').write('BUNDLE')",
            ]
        ).format(rec=recorder),
        encoding="utf-8",
    )
    import mcp_server
    monkeypatch.setattr(mcp_server, "BRAIN_ASK_SCRIPT", stub)

    text1, is_err1 = mcp_server.run_recall("first query")
    assert "BUNDLE" in text1
    assert is_err1 is False

    text2, is_err2 = mcp_server.run_recall("second query")
    assert "BUNDLE" in text2
    assert is_err2 is False

    used = [l.strip() for l in recorder.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(used) == 2, used
    assert used[0] != used[1], f"both calls shared one answer file: {used[0]}"
    for p in used:
        assert not Path(p).exists(), f"answer file left behind: {p}"


def test_run_recall_read_text_error(tmp_path, monkeypatch):
    """Verify answer file read failure returns distinct error with context."""
    stub = tmp_path / "stub_brain.py"
    stub.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")

    import mcp_server
    monkeypatch.setattr(mcp_server, "BRAIN_ASK_SCRIPT", stub)

    orig_read_text = Path.read_text

    def mock_read_text(self, *args, **kwargs):
        if str(self).endswith(".txt"):
            raise PermissionError("Access denied")
        return orig_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", mock_read_text)
    text, is_err = mcp_server.run_recall("test query")
    assert is_err is True
    assert "Recall error: failed to read answer file: Access denied" in text


def test_handle_request_read_text_error_is_mcp_error(tmp_path, monkeypatch):
    """Verify answer file read error propagates through handle_request as isError: true."""
    stub = tmp_path / "stub_brain.py"
    stub.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")

    import mcp_server
    monkeypatch.setattr(mcp_server, "BRAIN_ASK_SCRIPT", stub)

    orig_read_text = Path.read_text

    def mock_read_text(self, *args, **kwargs):
        if str(self).endswith(".txt"):
            raise OSError("Disk I/O failure")
        return orig_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", mock_read_text)

    req = {
        "jsonrpc": "2.0",
        "id": 11,
        "method": "tools/call",
        "params": {"name": "memory_recall", "arguments": {"query": "test query"}},
    }
    resp = mcp_server.handle_request(req)
    assert resp["id"] == 11
    assert resp["result"]["isError"] is True
    assert "Recall error: failed to read answer file: Disk I/O failure" in resp["result"]["content"][0]["text"]

