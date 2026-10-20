# -*- coding: utf-8 -*-
"""Unit tests for examples/mcp_server.py (MCP Protocol and tool handling).

Runs offline with no model download or network.
"""
import sys, os, json, io, subprocess, pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sqlite_graph_memory.mcp_server import handle_request, TOOLS, SERVER_INFO
from sqlite_graph_memory import mcp_server



def test_server_info_version_tracks_package():
    from sqlite_graph_memory import __version__
    assert SERVER_INFO["version"] == __version__


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
    from sqlite_graph_memory import mcp_server
    monkeypatch.setattr(mcp_server, "_brain_ask_cmd", lambda: [sys.executable, str(stub)])
    text, is_err = mcp_server.run_recall("test agent memory query", mode="associative")
    assert "STUB CONTEXT BUNDLE FOR: test agent memory query" in text
    assert is_err is False


def test_run_recall_exit_code_error(tmp_path, monkeypatch):
    """Verify non-zero exit codes from brain_ask.py are captured and returned safely."""
    stub = tmp_path / "stub_fail.py"
    stub.write_text("import sys\nsys.stderr.write('SYNTHETIC_FAILURE_MSG\\n')\nsys.exit(1)\n", encoding="utf-8")
    from sqlite_graph_memory import mcp_server
    monkeypatch.setattr(mcp_server, "_brain_ask_cmd", lambda: [sys.executable, str(stub)])
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
    from sqlite_graph_memory import mcp_server
    monkeypatch.setattr(mcp_server, "_brain_ask_cmd", lambda: [sys.executable, str(stub)])

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
    from sqlite_graph_memory import mcp_server
    monkeypatch.setattr(mcp_server, "_brain_ask_cmd", lambda: None)
    text, is_err = mcp_server.run_recall("test query")
    assert "brain_ask not found" in text
    assert is_err is True


def test_run_recall_timeout(tmp_path, monkeypatch):
    """Verify subprocess timeout is caught and returns clear error text."""
    from sqlite_graph_memory import mcp_server
    import subprocess
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
    from sqlite_graph_memory import mcp_server
    monkeypatch.setattr(mcp_server, "_brain_ask_cmd", lambda: [sys.executable, str(stub)])

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

    from sqlite_graph_memory import mcp_server
    monkeypatch.setattr(mcp_server, "_brain_ask_cmd", lambda: [sys.executable, str(stub)])

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

    from sqlite_graph_memory import mcp_server
    monkeypatch.setattr(mcp_server, "_brain_ask_cmd", lambda: [sys.executable, str(stub)])

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


def test_handle_request_non_object_json():
    """Verify handle_request returns -32600 error for non-dict requests."""
    for payload in ([1, 2, 3], "literal_string", 123, True):
        resp = handle_request(payload)
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] is None
        assert resp["error"]["code"] == -32600
        assert "Invalid Request" in resp["error"]["message"]


def test_stdio_loop_handles_non_object_json_and_continues(monkeypatch):
    """Verify stdio loop handles array and bare literal payloads with -32600 and keeps running."""
    input_lines = [
        "[1, 2, 3]\n",
        '"bare literal"\n',
        json.dumps({"jsonrpc": "2.0", "id": 42, "method": "ping"}) + "\n"
    ]
    fake_stdin = io.StringIO("".join(input_lines))
    fake_stdout = io.StringIO()

    monkeypatch.setattr(sys, "stdin", fake_stdin)
    monkeypatch.setattr(sys, "stdout", fake_stdout)
    monkeypatch.setattr(sys, "argv", ["sgm-mcp"])

    mcp_server.main()

    output_lines = [line.strip() for line in fake_stdout.getvalue().strip().splitlines() if line.strip()]
    assert len(output_lines) == 3

    resp1 = json.loads(output_lines[0])
    assert resp1["jsonrpc"] == "2.0"
    assert resp1["id"] is None
    assert resp1["error"]["code"] == -32600

    resp2 = json.loads(output_lines[1])
    assert resp2["jsonrpc"] == "2.0"
    assert resp2["id"] is None
    assert resp2["error"]["code"] == -32600

    resp3 = json.loads(output_lines[2])
    assert resp3["jsonrpc"] == "2.0"
    assert resp3["id"] == 42
    assert resp3["result"] == {}
def test_brain_ask_cmd_prefers_sibling_script():
    """Verify _brain_ask_cmd resolves sibling brain_ask.py when it exists."""
    cmd = mcp_server._brain_ask_cmd()
    expected_sibling = str(mcp_server.ROOT_DIR / "brain_ask.py")
    assert cmd == [sys.executable, expected_sibling]


def test_run_recall_injects_pythonpath(tmp_path, monkeypatch):
    """Verify run_recall includes src/ in PYTHONPATH so child process can resolve package imports."""
    stub = tmp_path / "stub_env_check.py"
    stub.write_text(
        "import os, sys\n"
        "from pathlib import Path\n"
        "pp = os.environ.get('PYTHONPATH', '')\n"
        "src_dir = sys.argv[1]\n"
        "if src_dir not in pp.split(os.pathsep):\n"
        "    sys.stderr.write('src_dir not in PYTHONPATH: ' + pp)\n"
        "    sys.exit(2)\n"
        "ans = os.environ.get('BRAIN_ANSWER_OUT')\n"
        "if ans:\n"
        "    Path(ans).write_text('OK', encoding='utf-8')\n"
        "sys.exit(0)\n",
        encoding="utf-8"
    )
    src_dir = str(mcp_server.ROOT_DIR.parent)
    monkeypatch.setattr(mcp_server, "_brain_ask_cmd", lambda: [sys.executable, str(stub), src_dir])

    text, is_err = mcp_server.run_recall("test query")
    assert is_err is False
    assert text == "OK"


def test_main_cli_test_mode_exits_nonzero_on_error(monkeypatch):
    """Verify sgm-mcp --test exits with non-zero code when recall returns an error."""
    monkeypatch.setattr(mcp_server, "run_recall", lambda q, mode="associative": ("Recall error details", True))
    monkeypatch.setattr(sys, "argv", ["sgm-mcp", "--test", "error query"])

    with pytest.raises(SystemExit) as exc_info:
        mcp_server.main()
    assert exc_info.value.code == 1


def test_main_cli_test_mode_exits_zero_on_success(monkeypatch, capsys):
    """Verify sgm-mcp --test exits cleanly (code 0) when recall succeeds."""
    monkeypatch.setattr(mcp_server, "run_recall", lambda q, mode="associative": ("Found memories", False))
    monkeypatch.setattr(sys, "argv", ["sgm-mcp", "--test", "ok query"])

    mcp_server.main()
    captured = capsys.readouterr()
    assert "Found memories" in captured.out

def test_mcp_server_script_execution_without_installed_package():
    """Verify invoking mcp_server.py directly by path without PYTHONPATH succeeds past import."""
    script_path = mcp_server.ROOT_DIR / "mcp_server.py"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    res = subprocess.run(
        [sys.executable, str(script_path), "--test", "smoke"],
        cwd=str(mcp_server.ROOT_DIR.parent.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert "ModuleNotFoundError: No module named 'sqlite_graph_memory'" not in res.stderr
    assert "ModuleNotFoundError: No module named 'sqlite_graph_memory'" not in res.stdout
    assert res.returncode == 1
    assert "=== MCP Test Query: smoke (mode=associative) ===" in res.stdout

