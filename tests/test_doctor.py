"""sgm-doctor: each verdict has to come from the machine, not be hard-coded."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlite_graph_memory import doctor  # noqa: E402


def test_old_python_fails():
    assert doctor.check_python((3, 8, 1))[0] == doctor.FAIL
    assert doctor.check_python((3, 12, 0))[0] == doctor.OK


def test_missing_index_warns_and_names_the_command(tmp_path):
    status, what, fix = doctor.check_index(tmp_path)
    assert status == doctor.WARN and "sgm-index" in fix


def test_half_index_fails(tmp_path):
    (tmp_path / "_brain_e5.npy").write_bytes(b"x")
    status, what, _ = doctor.check_index(tmp_path)
    assert status == doctor.FAIL and "_brain_e5_meta.pkl" in what


def test_full_index_ok(tmp_path):
    import pickle
    (tmp_path / "_brain_e5.npy").write_bytes(b"x")
    (tmp_path / "_brain_e5_meta.pkl").write_bytes(pickle.dumps([]))
    assert doctor.check_index(tmp_path)[0] == doctor.OK


def test_index_with_unresolvable_relative_paths_warns(tmp_path):
    import pickle
    (tmp_path / "_brain_e5.npy").write_bytes(b"x")
    meta = [{"path": "relative/path/that/does/not/exist/note.md", "title": "note"}]
    (tmp_path / "_brain_e5_meta.pkl").write_bytes(pickle.dumps(meta))
    status, what, fix = doctor.check_index(tmp_path)
    assert status == doctor.WARN
    assert "paths in index do not resolve" in what
    assert "sgm-index" in fix


def test_models_missing_when_cache_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.delenv("SENTENCE_TRANSFORMERS_HOME", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "none"))
    assert doctor.check_models()[0] == doctor.WARN
    (tmp_path / "models--intfloat--multilingual-e5-base").mkdir()
    (tmp_path / "models--cross-encoder--mmarco-mMiniLMv2-L12-H384-v1").mkdir()
    assert doctor.check_models()[0] == doctor.OK


def test_missing_torch_is_a_warning_not_a_failure(monkeypatch):
    monkeypatch.setenv("SGM_BACKEND", "torch")
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ModuleNotFoundError
    status, what, fix = doctor.check_embeddings()
    assert status == doctor.WARN and "embeddings" in fix and "lite" in fix


def test_light_backend_is_ok_without_torch(monkeypatch):
    import pytest
    pytest.importorskip("onnxruntime")
    monkeypatch.setenv("SGM_BACKEND", "onnx")
    monkeypatch.setitem(sys.modules, "torch", None)
    status, what, _ = doctor.check_embeddings()
    assert status == doctor.OK and "onnxruntime" in what


import pytest  # noqa: E402


@pytest.mark.parametrize("missing", ["onnxruntime", "tokenizers", "huggingface_hub"])
def test_incomplete_light_backend_fails(monkeypatch, missing):
    """Codex review 2026-11-03: onnxruntime alone used to print ok while recall died."""
    monkeypatch.setenv("SGM_BACKEND", "onnx")
    monkeypatch.setitem(sys.modules, missing, None)
    status, what, fix = doctor.check_embeddings()
    # Names the first missing package, which on a runner without [lite] is onnxruntime.
    assert status == doctor.FAIL and "[lite]" in fix and "incomplete" in what


def test_broken_torch_on_intel_mac_points_at_numpy(monkeypatch):
    monkeypatch.setenv("SGM_BACKEND", "torch")
    class Boom:
        def __getattr__(self, name):
            raise RuntimeError("Numpy is not available")
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "torch":
            raise RuntimeError("Numpy is not available")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    import numpy
    monkeypatch.setattr(numpy, "__version__", "2.0.1")
    status, what, fix = doctor.check_embeddings(system="Darwin", machine="x86_64")
    assert status == doctor.FAIL and "numpy<2" in fix


def test_render_exit_code_follows_failures():
    text, fails = doctor.render([("A", (doctor.OK, "fine", "")), ("B", (doctor.FAIL, "bad", "fix it"))])
    assert fails == 1 and "fix it" in text and "Tell us" in text
    text, fails = doctor.render([("A", (doctor.WARN, "meh", "do x"))])
    assert fails == 0 and "Nothing is broken" in text
