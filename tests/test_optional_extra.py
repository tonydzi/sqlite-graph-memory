# -*- coding: utf-8 -*-
"""A base install must say what to install, not print a traceback.

`pip install sqlite-graph-memory` deliberately leaves out sentence-transformers
and torch (~2 GB), so running the indexer on a base install is the *normal*
first experience, not an edge case. Before these tests existed it ended in a
`ModuleNotFoundError` traceback through site-packages, which reads like a broken
package rather than a missing extra.

RED-FIRST: drop the `_require_sentence_transformers()` call from
`index_notes.main` (or from `load_encoder` / `load_reranker`) and both tests
below fail — `pytest.raises(SystemExit)` sees `ModuleNotFoundError` instead.

No model download, no network: the import is forced to fail on purpose.
"""
import builtins
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlite_graph_memory import brain_ask, index_notes  # noqa: E402


@pytest.fixture
def no_sentence_transformers(monkeypatch):
    """Make `import sentence_transformers` fail the way a base install does."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ModuleNotFoundError(
                "No module named 'sentence_transformers'", name="sentence_transformers"
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_indexer_exits_with_install_instructions(tmp_path, monkeypatch, capsys,
                                                 no_sentence_transformers):
    """`sgm-index` on a base install names the extra instead of raising."""
    # Chunks shorter than 40 characters are dropped by the chunker, so a
    # toy two-line note would make this test pass for the wrong reason.
    (tmp_path / "a.md").write_text(
        "# A\n\nThis note is long enough to survive chunking, and it links\n"
        "to [[B]] so the graph half has something to chew on.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["sgm-index", str(tmp_path)])
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc:
        index_notes.main()

    message = str(exc.value)
    assert "sqlite-graph-memory[embeddings]" in message
    assert "pip install" in message


@pytest.mark.parametrize("loader", ["load_encoder", "load_reranker"])
def test_brain_ask_loaders_exit_with_install_instructions(loader,
                                                          no_sentence_transformers):
    """Both model loaders in brain_ask fail the same, readable way."""
    with pytest.raises(SystemExit) as exc:
        getattr(brain_ask, loader)()

    assert "sqlite-graph-memory[embeddings]" in str(exc.value)


def test_unrelated_missing_module_is_not_swallowed(monkeypatch):
    """A genuinely different ImportError must still surface as itself.

    Otherwise the helper would turn every packaging problem into "install the
    embeddings extra", which would send people chasing the wrong fix.
    """
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ModuleNotFoundError("No module named 'numpy'", name="numpy")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError):
        index_notes._require_sentence_transformers()
