# -*- coding: utf-8 -*-
"""`--version` on sgm-stats and sgm-ask prints the package version and exits 0 (issue #75).

Runs offline: no model, no index, no network. Both commands are run as modules so the
test does not need the package installed.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from sqlite_graph_memory import __version__  # noqa: E402


@pytest.mark.parametrize(
    ("module", "prog"),
    [
        ("sqlite_graph_memory.vault_stats", "sgm-stats"),
        ("sqlite_graph_memory.brain_ask", "sgm-ask"),
    ],
)
def test_version_flag_prints_the_version_and_exits_zero(module, prog, tmp_path):
    env = os.environ.copy()
    env["BRAIN_INDEX_DIR"] = str(tmp_path / "no-index")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC), env.get("PYTHONPATH")]))
    res = subprocess.run(
        [sys.executable, "-m", module, "--version"],
        capture_output=True, text=True, cwd=str(tmp_path), env=env, timeout=60,
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip().endswith(__version__), res.stdout
    assert prog in res.stdout
