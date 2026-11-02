"""Every place that names the version agrees, on every PR.

v0.3.4 was tagged on 2026-10-31 and never reached PyPI: the release commit changed only
CHANGELOG.md, so pyproject, __init__ and server.json still said 0.3.3 and the publish job
stopped on its own version check, after the tag. That check runs too late to prevent
anything, so the same comparison runs here, where a red X stops the merge.

A CHANGELOG section may be written ahead of the bump if its header says "unreleased".
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTION = re.compile(r"^## v(\d+\.\d+\.\d+)\b(.*)$", re.M)


def top_released(changelog):
    """Version of the first section not marked unreleased, or None."""
    for m in SECTION.finditer(changelog):
        if "unreleased" not in m.group(2).lower():
            return m.group(1)
    return None


def disagreements(pyproject, init_py, server, changelog):
    """Places that differ from the version in pyproject.toml."""
    want = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M).group(1)
    pkg = server["packages"][0]
    seen = {
        "__init__.__version__": re.search(r'^__version__\s*=\s*"([^"]+)"', init_py, re.M).group(1),
        "server.json version": server["version"],
        "server.json packages[0].version": pkg["version"],
        "server.json runtimeArguments pin": pkg["runtimeArguments"][0]["value"].rsplit("==", 1)[-1],
        "CHANGELOG top released section": top_released(changelog),
    }
    return [f"{k} = {v}, pyproject = {want}" for k, v in seen.items() if v != want]


def test_all_versions_agree():
    bad = disagreements(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        (ROOT / "src/sqlite_graph_memory/__init__.py").read_text(encoding="utf-8"),
        json.loads((ROOT / "server.json").read_text(encoding="utf-8")),
        (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"),
    )
    assert not bad, "version mismatch, a release from here would stop before PyPI:\n  " + "\n  ".join(bad)


PY = 'version = "0.3.3"\n'
INIT = '__version__ = "0.3.3"\n'
SRV = {"version": "0.3.3", "packages": [{"version": "0.3.3",
       "runtimeArguments": [{"value": "sqlite-graph-memory[embeddings]==0.3.3"}]}]}


def test_the_v034_release_commit_is_caught():
    log = "# Changelog\n\n## v0.3.4 - 2026-10-31\n\nx\n\n## v0.3.3 - 2026-10-26\n\ny\n"
    assert disagreements(PY, INIT, SRV, log) == ["CHANGELOG top released section = 0.3.4, pyproject = 0.3.3"]


def test_section_marked_unreleased_may_lead():
    log = "## v0.3.4 - unreleased\n\nx\n\n## v0.3.3 - 2026-10-26\n\ny\n"
    assert disagreements(PY, INIT, SRV, log) == []


def test_forgotten_pin_in_server_json_is_caught():
    srv = json.loads(json.dumps(SRV))
    srv["packages"][0]["runtimeArguments"][0]["value"] = "sqlite-graph-memory[embeddings]==0.3.2"
    assert len(disagreements(PY, INIT, srv, "## v0.3.3\n")) == 1
