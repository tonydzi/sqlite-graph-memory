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
    np.save(tmp_path / "_brain_e5.npy", np.zeros((1, 4), dtype="float32"))
    (tmp_path / "_brain_e5_meta.pkl").write_bytes(pickle.dumps([{"path": "note.md"}]))
    assert doctor.check_index(tmp_path)[0] == doctor.OK


def test_empty_or_truncated_index_fails(tmp_path):
    (tmp_path / "_brain_e5.npy").touch()
    (tmp_path / "_brain_e5_meta.pkl").touch()
    status, what, fix = doctor.check_index(tmp_path)
    assert status == doctor.FAIL
    assert "empty" in what or "corrupt" in what or "truncated" in what


def test_mismatched_index_shape_fails(tmp_path):
    import pickle
    import numpy as np
    np.save(tmp_path / "_brain_e5.npy", np.zeros((2, 4), dtype="float32"))
    (tmp_path / "_brain_e5_meta.pkl").write_bytes(pickle.dumps([{"path": "one.md"}]))
    status, what, fix = doctor.check_index(tmp_path)
    assert status == doctor.FAIL
    assert "rows (2) != _brain_e5_meta.pkl chunks (1)" in what


def test_doctor_importable_when_numpy_is_none():
    import os
    import subprocess
    import sys
    src = str(Path(__file__).resolve().parent.parent / "src")
    code = (
        "import sys; "
        f"sys.path.insert(0, {src!r}); "
        "sys.modules['numpy'] = None; "
        "from sqlite_graph_memory import doctor; "
        "assert doctor.check_numpy()[0] == doctor.FAIL"
    )
    env = {**os.environ, "PYTHONPATH": src}
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr


def test_transitive_dependency_missing_fails_not_warns(monkeypatch):
    monkeypatch.setenv("SGM_BACKEND", "torch")
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ModuleNotFoundError("No module named 'safetensors'", name="safetensors")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    status, what, fix = doctor.check_embeddings()
    assert status == doctor.FAIL
    assert "safetensors" in what or "fails to import" in what
    assert "Reinstall" in fix


def test_hf_custom_cache_does_not_mix_default(tmp_path, monkeypatch):
    custom_cache = tmp_path / "custom"
    custom_cache.mkdir()
    fake_default = tmp_path / "default_hf" / "hub"
    fake_default.mkdir(parents=True)
    (fake_default / "models--intfloat--multilingual-e5-base").mkdir()
    (fake_default / "models--cross-encoder--mmarco-mMiniLMv2-L12-H384-v1").mkdir()

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HF_HUB_CACHE", str(custom_cache))
    monkeypatch.delenv("SENTENCE_TRANSFORMERS_HOME", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    status, what, _ = doctor.check_models()
    assert status == doctor.WARN
    assert "Not downloaded yet" in what


def test_xdg_cache_home_respected(tmp_path, monkeypatch):
    xdg_dir = tmp_path / "xdg"
    hub = xdg_dir / "huggingface" / "hub"
    hub.mkdir(parents=True)
    (hub / "models--intfloat--multilingual-e5-base").mkdir()
    (hub / "models--cross-encoder--mmarco-mMiniLMv2-L12-H384-v1").mkdir()

    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_dir))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("SENTENCE_TRANSFORMERS_HOME", raising=False)

    status, what, _ = doctor.check_models()
    assert status == doctor.OK


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
