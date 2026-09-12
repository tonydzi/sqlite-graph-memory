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
