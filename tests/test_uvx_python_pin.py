"""Every documented uvx line must pin Python, or it breaks on an Intel Mac.

onnxruntime has no x86_64 macOS wheels for Python 3.14 and torch none for 3.13+. An
unpinned `uvx --with 'sqlite-graph-memory[lite]' ...` resolves on uv's default Python.
When that default is too new, uv falls back to 0.3.5, a release without `[lite]`, and
`sgm-index` stops with a hint to install `[embeddings]` (measured 2026-11-04, MacPro192,
uv default 3.14: exit 1 and "does not have an extra named lite").
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ["README.md", "examples/README.md", "llms.txt"]
EXTRAS = ("sqlite-graph-memory[lite]", "sqlite-graph-memory[embeddings]")
PIN = "3.12"  # newest Python with torch wheels on Intel Macs


def unpinned_uvx_lines(text):
    """uvx command lines that install the embeddings extra without --python."""
    return [
        line for line in text.splitlines()
        if "uvx " in line and any(e in line for e in EXTRAS) and f"--python {PIN}" not in line
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
    old = '```json\n{"mcpServers": {"x": {"command": "uvx", "args": ["--with", "y"]}}}\n```'
    assert unpinned_json_args(old) == ["x"]


def test_docs_pin_python():
    for name in DOCS:
        text = (ROOT / name).read_text(encoding="utf-8")
        assert not unpinned_uvx_lines(text), name
        assert not unpinned_json_args(text), name


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
