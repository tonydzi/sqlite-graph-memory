# -*- coding: utf-8 -*-
"""mcp_server.py — Model Context Protocol (MCP) server for sqlite-graph-memory.

Exposes associative memory recall as a standard MCP tool over stdio transport.
Compatible with any MCP client (Cursor, Claude Desktop, Antigravity, Zed, Windsurf).

Protocol: JSON-RPC 2.0 over stdio (newline-delimited JSON).
Zero third-party dependencies required (pure standard library).

USAGE:
  # Normal stdio mode (started by MCP client)
  python examples/mcp_server.py

  # Direct test mode (runs a recall query without an MCP client)
  python examples/mcp_server.py --test "how do I think about agent memory"

Config (env, optional):
  BRAIN_INDEX_DIR   dir holding embedding index (default: ./index)
  TURNSTATE_DB      SQLite db for telemetry (default: ./turnstate.db)
  BRAIN_ANSWER_OUT  file to mirror answer into (default: ./_brain_answer.txt)
"""
import sys, os, json, subprocess, tempfile
from pathlib import Path

# Ensure UTF-8 for stdio communication on all platforms (including Windows console)
try:
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parents[1]
BRAIN_ASK_SCRIPT = ROOT_DIR / "brain_ask.py"

SERVER_INFO = {
    "name": "sqlite-graph-memory",
    "version": "0.1.3"
}

TOOLS = [
    {
        "name": "memory_recall",
        "description": (
            "Associative memory recall over linked markdown notes. "
            "Retrieves relevant context using vector search, 1-hop [[wikilink]] graph expansion, "
            "and cross-encoder reranking. Returns a formatted context bundle."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query or topic to retrieve memory for."
                },
                "mode": {
                    "type": "string",
                    "enum": ["associative", "direct", "ab"],
                    "default": "associative",
                    "description": (
                        "Retrieval mode: 'associative' (vector + graph expansion, recommended), "
                        "'direct' (vector search only), or 'ab' (run both and log comparison telemetry)."
                    )
                }
            },
            "required": ["query"]
        }
    }
]


def run_recall(query: str, mode: str = "associative") -> str:
    """Invoke brain_ask.py in a crash-safe subprocess and return the retrieved text."""
    if not BRAIN_ASK_SCRIPT.exists():
        return f"Error: brain_ask.py not found at {BRAIN_ASK_SCRIPT}"

    if mode not in ("associative", "direct", "ab"):
        mode = "associative"

    cmd = [sys.executable, str(BRAIN_ASK_SCRIPT)]
    if mode == "ab":
        cmd.append("--ab")
    elif mode == "direct":
        cmd.append("--ask")
    else:  # associative (default)
        cmd.extend(["--graph", "--ask"])
    cmd.append(query)

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
        temp_ans_file = Path(tf.name)

    try:
        call_env = {**os.environ, "BRAIN_ANSWER_OUT": str(temp_ans_file)}
        res = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            env=call_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60
        )
        if res.returncode != 0:
            err = (res.stderr or res.stdout or "").strip()
            return f"Recall error (exit code {res.returncode}):\n{err}"

        # If --ask mode was used, read the scoped per-call answer file
        if temp_ans_file.exists():
            try:
                content = temp_ans_file.read_text(encoding="utf-8", errors="replace").strip()
                if content:
                    return content
            except Exception:
                pass

        return res.stdout.strip() or "(no matching notes found)"
    except subprocess.TimeoutExpired:
        return "Recall error: timed out after 60 seconds."
    except Exception as e:
        return f"Recall exception: {e}"
    finally:
        if temp_ans_file.exists():
            try:
                temp_ans_file.unlink()
            except Exception:
                pass


def handle_request(req: dict) -> dict:
    """Handle a single JSON-RPC 2.0 MCP request."""
    method = req.get("method")
    req_id = req.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": SERVER_INFO
            }
        }

    elif method == "notifications/initialized":
        # Notification from client, no response required
        return None

    elif method == "ping":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {}
        }

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": TOOLS
            }
        }

    elif method == "tools/call":
        params = req.get("params") or {}
        tool_name = params.get("name")
        args = params.get("arguments") or {}

        if tool_name == "memory_recall":
            query = args.get("query", "")
            mode = args.get("mode", "associative")
            if not query:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": "Error: 'query' argument is required."}],
                        "isError": True
                    }
                }
            result_text = run_recall(query, mode=mode)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": result_text}],
                    "isError": False
                }
            }
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Tool not found: {tool_name}"
                }
            }

    else:
        if req_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
        return None


def main():
    # Direct CLI testing mode: python examples/mcp_server.py --test "query"
    if len(sys.argv) > 1 and sys.argv[1] in ("--test", "-t", "--query"):
        flags = {"--test", "-t", "--query", "--direct", "--ab", "--associative"}
        mode = "associative"
        if "--direct" in sys.argv:
            mode = "direct"
        elif "--ab" in sys.argv:
            mode = "ab"
        query_parts = [a for a in sys.argv[1:] if a not in flags]
        q = " ".join(query_parts) if query_parts else "agent memory"
        print(f"=== MCP Test Query: {q} (mode={mode}) ===")
        print(run_recall(q, mode=mode))
        return


    # Standard MCP stdio loop
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            sys.stderr.write(f"JSON decode error: {e}\n")
            continue

        resp = handle_request(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()

