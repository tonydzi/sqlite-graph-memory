"""A missing index must say where to tell us about it (stderr), without needing a model."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlite_graph_memory import brain_ask  # noqa: E402


def test_missing_index_prints_feedback_link(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(brain_ask, "EMB", tmp_path / "nope.npy")
    monkeypatch.setattr(sys, "argv", ["sgm-ask", "anything"])
    with pytest.raises(SystemExit) as exc:
        brain_ask.main()
    assert exc.value.code == 1
    out = capsys.readouterr()
    assert "need an embedding index" in out.err
    assert brain_ask.FEEDBACK_URL in out.err


def test_empty_query_does_not_nag(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(brain_ask, "EMB", tmp_path / "nope.npy")
    monkeypatch.setattr(sys, "argv", ["sgm-ask"])
    with pytest.raises(SystemExit) as exc:
        brain_ask.main()
    assert exc.value.code == 1
    assert brain_ask.FEEDBACK_URL not in capsys.readouterr().err
