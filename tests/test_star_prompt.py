# -*- coding: utf-8 -*-
"""The star line after `sgm-ask --ab`: shown once per machine, never to machines.

Each case checks behaviour that changes if the guard is removed: delete the marker
write and the second call speaks again; drop the CI or env check and those calls
speak too.
"""
import io
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sqlite_graph_memory.brain_ask import maybe_star_prompt, STAR_MARKER, STAR_URL  # noqa: E402


def _call(tmp_path, **env):
    buf = io.StringIO()
    shown = maybe_star_prompt(stream=buf, env={"SGM_CONFIG_DIR": str(tmp_path), **env},
                              interactive=True)
    return shown, buf.getvalue()


def test_first_call_asks_once_and_leaves_marker(tmp_path):
    shown, out = _call(tmp_path)
    assert shown and STAR_URL in out
    assert (tmp_path / STAR_MARKER).exists()


def test_second_call_is_silent(tmp_path):
    _call(tmp_path)
    shown, out = _call(tmp_path)
    assert not shown and out == ""


def test_opt_out_env_is_silent_and_writes_nothing(tmp_path):
    shown, out = _call(tmp_path, SGM_NO_STAR_PROMPT="1")
    assert not shown and out == ""
    assert not (tmp_path / STAR_MARKER).exists()


def test_ci_is_silent(tmp_path):
    shown, out = _call(tmp_path, CI="true")
    assert not shown and out == ""
    assert not (tmp_path / STAR_MARKER).exists()


def test_non_terminal_stdout_is_silent(tmp_path):
    buf = io.StringIO()
    shown = maybe_star_prompt(stream=buf, env={"SGM_CONFIG_DIR": str(tmp_path)},
                              interactive=False)
    assert not shown and buf.getvalue() == ""


def test_unwritable_config_dir_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    buf = io.StringIO()
    shown = maybe_star_prompt(stream=buf, env={"SGM_CONFIG_DIR": str(blocker / "sub")},
                              interactive=True)
    assert shown is False


@pytest.mark.parametrize("ci", ["", "0", "false", "False"])
def test_ci_values_that_mean_no_ci_still_ask(tmp_path, ci):
    shown, _ = _call(tmp_path, CI=ci)
    assert shown


def test_ab_json_consumer_is_silent(tmp_path):
    shown, out = _call(tmp_path, BRAIN_AB_JSON=str(tmp_path / "ab.json"))
    assert not shown and out == ""


def test_tilde_in_config_dir_expands(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    buf = io.StringIO()
    assert maybe_star_prompt(stream=buf, env={"SGM_CONFIG_DIR": "~/sgm"}, interactive=True)
    assert (tmp_path / "sgm" / STAR_MARKER).exists()


class _BrokenStream(io.StringIO):
    def write(self, s):
        raise BrokenPipeError


def test_failed_write_does_not_burn_the_marker(tmp_path):
    shown = maybe_star_prompt(stream=_BrokenStream(), env={"SGM_CONFIG_DIR": str(tmp_path)},
                              interactive=True)
    assert shown is False
    assert not (tmp_path / STAR_MARKER).exists()
    shown, out = _call(tmp_path)
    assert shown and STAR_URL in out


def test_main_ab_calls_the_prompt(tmp_path, monkeypatch):
    """The wiring: a successful `sgm-ask --ab` reaches maybe_star_prompt()."""
    import numpy as np
    from sqlite_graph_memory import brain_ask as ba

    emb = tmp_path / "_brain_e5.npy"
    emb.write_bytes(b"x")
    meta = [{"path": "a.md", "title": "a", "snippet": "a"}]
    monkeypatch.setattr(ba, "EMB", emb)
    monkeypatch.setattr(ba, "AB_DB", tmp_path / "t.db")
    monkeypatch.setattr(ba, "load_index", lambda: (np.zeros((1, 2)), meta))
    monkeypatch.setattr(ba, "load_encoder", lambda dev=None: None)
    monkeypatch.setattr(ba, "load_reranker", lambda dev=None: None)
    monkeypatch.setattr(ba, "query_sims", lambda enc, emb, q: np.ones(1))
    monkeypatch.setattr(ba, "retrieve_candidates", lambda sims, meta: [0])
    monkeypatch.setattr(ba, "expand_1hop", lambda sims, meta, order: [])
    monkeypatch.setattr(ba, "rerank_candidates", lambda ce, q, cand, meta: [(i, 1.0) for i in cand])
    monkeypatch.setenv("BRAIN_ANSWER_OUT", str(tmp_path / "out.txt"))
    calls = []
    monkeypatch.setattr(ba, "maybe_star_prompt", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(sys, "argv", ["sgm-ask", "--ab", "question"])
    ba.main()
    assert calls == [1]
