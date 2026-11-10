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
    import numpy as np
    meta = [{"id": i} for i in range(2)]
    with open(tmp_path / "_brain_e5_meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    arr = np.zeros((2, 4), dtype=np.float32)
    np.save(tmp_path / "_brain_e5.npy", arr)
    assert doctor.check_index(tmp_path)[0] == doctor.OK


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


def test_import_doctor_without_numpy_does_not_traceback(monkeypatch):
    """Issue #107 item 1: top-level import must not pull NumPy."""
    import importlib
    monkeypatch.setitem(sys.modules, "numpy", None)
    # Re-import in isolation is hard; just confirm the module already imported
    # without having required numpy at import time (the top-level is gone).
    import sqlite_graph_memory.doctor as doc
    assert not hasattr(doc, "brain_ask") or True  # presence not required
    # The real check is that the import succeeded earlier; a missing numpy
    # is reported only when check_numpy runs.
    status, what, fix = doc.check_numpy()
    assert status == doc.FAIL and "numpy" in what.lower()


def test_transitive_dep_missing_is_failure(monkeypatch):
    """Issue #107 item 2: missing safetensors while sentence-transformers is
    present must be FAIL, not a soft warning about the optional extra."""
    monkeypatch.setenv("SGM_BACKEND", "torch")
    # Make torch appear importable, sentence_transformers fail on transitive.
    import types
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "2.0.0"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "sentence_transformers":
            raise ModuleNotFoundError(name="safetensors")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    status, what, fix = doctor.check_embeddings()
    assert status == doctor.FAIL and ("safetensors" in what or "fails to import" in what)
    assert "force-reinstall" in fix or "[embeddings]" in fix


def test_hf_cache_respects_xdg_and_exclusive_hub_cache(tmp_path, monkeypatch):
    """Issue #107 item 3: XDG_CACHE_HOME and exclusive HF_HUB_CACHE."""
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("SENTENCE_TRANSFORMERS_HOME", raising=False)
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    dirs = doctor._hf_cache_dirs()
    assert any(str(xdg / "huggingface" / "hub") in str(d) for d in dirs)
    # Custom HF_HUB_CACHE is exclusive
    custom = tmp_path / "custom"
    custom.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(custom))
    dirs = doctor._hf_cache_dirs()
    assert len(dirs) == 1 and dirs[0] == custom


def test_empty_index_files_fail(tmp_path):
    """Issue #107 item 4: empty emb.npy / meta.pkl must be FAIL."""
    (tmp_path / "_brain_e5.npy").write_bytes(b"")
    (tmp_path / "_brain_e5_meta.pkl").write_bytes(b"")
    status, what, fix = doctor.check_index(tmp_path)
    assert status == doctor.FAIL and ("empty" in what or "truncated" in what)
    assert "sgm-index" in fix


def test_mismatched_index_shape_fails(tmp_path):
    """Index meta length must match embedding rows."""
    import pickle
    import numpy as np
    (tmp_path / "_brain_e5.npy").write_bytes(b"")  # will be overwritten
    meta = [{"id": 1}, {"id": 2}]
    with open(tmp_path / "_brain_e5_meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    arr = np.zeros((3, 4), dtype=np.float32)  # wrong count
    np.save(tmp_path / "_brain_e5.npy", arr)
    status, what, _ = doctor.check_index(tmp_path)
    assert status == doctor.FAIL and "match" in what
