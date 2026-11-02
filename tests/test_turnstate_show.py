# -*- coding: utf-8 -*-
"""sgm-turnstate-show: corrupt ledgers get a one-line error, not a traceback (#74)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sqlite_graph_memory import turnstate_show  # noqa: E402


def test_corrupt_ledger_exits_one_without_traceback(tmp_path, monkeypatch, capsys):
    bad = tmp_path / "c.db"
    bad.write_text("junk", encoding="utf-8")
    monkeypatch.setenv("TURNSTATE_DB", str(bad))
    turnstate_show.DB = str(bad)
    assert turnstate_show.main(["--stats"]) == 1
    err = capsys.readouterr().err
    assert "not a database" in err
    assert str(bad) in err
    assert "Traceback" not in err
    assert bad.read_text(encoding="utf-8") == "junk"


def test_corrupt_ledger_module_cli_exits_nonzero(tmp_path):
    bad = tmp_path / "c.db"
    bad.write_bytes(b"junk")
    env = os.environ.copy()
    env["TURNSTATE_DB"] = str(bad)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.run(
        [sys.executable, "-m", "sqlite_graph_memory.turnstate_show", "--stats"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "not a database" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert "Traceback" not in proc.stdout


def test_missing_ledger_stays_friendly(tmp_path, monkeypatch, capsys):
    missing = tmp_path / "missing.db"
    monkeypatch.setenv("TURNSTATE_DB", str(missing))
    turnstate_show.DB = str(missing)
    assert turnstate_show.main(["--stats"]) == 0
    out = capsys.readouterr().out
    assert "no ledger yet" in out
    assert str(missing) in out
