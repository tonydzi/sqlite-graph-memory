"""Every documented uvx line must pin Python, or it can break on a Mac.

onnxruntime has no Python 3.14 wheels for an Intel Mac or for Apple Silicon below macOS 14,
and torch none for 3.13+ on an Intel Mac. An unpinned `uvx --with 'sqlite-graph-memory[lite]' ...`
resolves on uv's default Python. When that default is too new, uv falls back to 0.3.5, a
release without `[lite]`, and `sgm-index` stops with a hint to install `[embeddings]`
(measured 2026-11-04, MacPro192, uv default 3.14: exit 1 with that hint).
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = sorted(
    path
    for pattern in ("*.md", "examples/*.md", "docs/*.md", "llms.txt")
    for path in ROOT.glob(pattern)
    if path.name != "CHANGELOG.md"  # released history keeps its old commands
)
PIN = "3.12"  # newest Python with torch wheels on Intel Macs


def unpinned_uvx_lines(text):
    """uvx commands for this package that do not pass the pin to uvx itself."""
    command = re.compile(r"uvx\s+(?:--|sqlite-graph-memory)")
    pinned = re.compile(rf"uvx\s+--python\s+{re.escape(PIN)}\s")
    return [
        line for line in text.splitlines()
        if "sqlite-graph-memory" in line and command.search(line) and not pinned.search(line)
    ]


def unpinned_json_args(text):
    """Client-config JSON blocks whose uvx args lack --python."""
    bad = []
    for body in re.findall(r"```json\n(.*?)```", text, re.S):
        for section in ("mcpServers", "servers"):
            for name, server in json.loads(body).get(section, {}).items():
                args = server.get("args", [])
                if server.get("command") == "uvx" and args[:2] != ["--python", PIN]:
                    bad.append(name)
    return bad


def test_detector_catches_the_old_lines():
    assert unpinned_uvx_lines("uvx --with 'sqlite-graph-memory[lite]' sqlite-graph-memory")
    assert unpinned_uvx_lines("uvx --python 3.13 --with 'sqlite-graph-memory[embeddings]' sqlite-graph-memory")
    assert unpinned_uvx_lines("uvx --python 3.120 --with 'sqlite-graph-memory[lite]' sqlite-graph-memory")
    assert unpinned_uvx_lines("uvx --with 'sqlite-graph-memory[lite]' sqlite-graph-memory --python 3.12")
    assert unpinned_uvx_lines("Replace `uvx --with ... sqlite-graph-memory` by `sgm-mcp`")
    assert not unpinned_uvx_lines("any client that speaks uvx (registry name io.github.tonydzi/sqlite-graph-memory)")
    old = '```json\n{"mcpServers": {"x": {"command": "uvx", "args": ["--with", "y"]}}}\n```'
    assert unpinned_json_args(old) == ["x"]


def test_docs_pin_python():
    assert any(path.name == "README.md" for path in DOCS)
    for path in DOCS:
        text = path.read_text(encoding="utf-8")
        assert not unpinned_uvx_lines(text), path
        assert not unpinned_json_args(text), path


def test_toml_block_pins_python():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    toml_args = re.findall(r"^args = \[.*$", text, re.M)
    assert toml_args and all(f'"--python", "{PIN}"' in line for line in toml_args)


def test_server_json_pins_python():
    server = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    args = server["packages"][0]["runtimeArguments"]
    assert {"name": "--python", "value": PIN}.items() <= next(
        a for a in args if a["name"] == "--python"
    ).items()
