# -*- coding: utf-8 -*-
"""The light (ONNX) backend: picked when installed, never imports torch, same numbers.

Why: through the MCP server on a Windows CPU laptop (2026-11-01) `import torch` alone took
44 s inside pick_device(), the encoder had not loaded at 90 s, and the client gave up at its
60 s tool timeout. The light path must not touch torch at all, not even to ask about CUDA.

RED-FIRST: drop the `backend() == 'onnx'` early return from brain_ask.pick_device and
test_pick_device_does_not_import_torch fails; drop the onnx branch from load_encoder and
test_loaders_use_onnx_without_sentence_transformers fails with SystemExit.

The last test downloads the two models (~1.6 GB once) and is skipped unless
SGM_LIVE_MODELS=1. Reference logits were measured with sentence-transformers 6.1 on CPU.
"""
import builtins
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlite_graph_memory import brain_ask, onnx_backend  # noqa: E402


@pytest.fixture
def torch_forbidden(monkeypatch):
    """Record every attempt to import torch or sentence_transformers.

    The attempts are recorded, not only refused: pick_device() wraps its torch import in
    `except Exception`, so a raising fake alone would be swallowed and the test stay green.
    """
    real_import = builtins.__import__
    attempts = []

    def fake_import(name, *args, **kwargs):
        if name.split(".")[0] in ("torch", "sentence_transformers"):
            attempts.append(name)
            raise ModuleNotFoundError("blocked by test", name=name)
        return real_import(name, *args, **kwargs)

    for mod in ("torch", "sentence_transformers"):
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    yield attempts
    assert attempts == [], "light backend tried to import %s" % attempts


def test_backend_env_forces_either_stack(monkeypatch):
    monkeypatch.setattr(onnx_backend, "available", lambda: True)
    assert brain_ask.backend({"SGM_BACKEND": "torch"}) == "torch"
    monkeypatch.setattr(onnx_backend, "available", lambda: False)
    assert brain_ask.backend({"SGM_BACKEND": "onnx"}) == "onnx"


@pytest.mark.parametrize("installed,expected", [(True, "onnx"), (False, "torch")])
def test_backend_auto_prefers_onnx_when_installed(monkeypatch, installed, expected):
    monkeypatch.setattr(onnx_backend, "available", lambda: installed)
    assert brain_ask.backend({}) == expected
    assert brain_ask.backend({"SGM_BACKEND": "auto"}) == expected


def test_pick_device_does_not_import_torch(monkeypatch, torch_forbidden):
    monkeypatch.setenv("SGM_BACKEND", "onnx")
    assert brain_ask.pick_device() == "cpu"


def test_loaders_use_onnx_without_sentence_transformers(monkeypatch, torch_forbidden):
    monkeypatch.setenv("SGM_BACKEND", "onnx")
    monkeypatch.setattr(onnx_backend, "Encoder", lambda: "enc")
    monkeypatch.setattr(onnx_backend, "CrossEncoder", lambda: "ce")
    assert brain_ask.load_encoder() == "enc"
    assert brain_ask.load_reranker() == "ce"


class _FakeST:
    """Stands in for sentence_transformers on a machine with cached torch models."""
    def SentenceTransformer(self, *a, **k):
        return "torch-enc"

    def CrossEncoder(self, *a, **k):
        return "torch-ce"


def _offline(*a, **k):
    raise OSError("offline: onnx/model.onnx is not in the cache")


@pytest.mark.parametrize("loader,want", [("load_encoder", "torch-enc"),
                                         ("load_reranker", "torch-ce")])
def test_auto_onnx_failure_falls_back_to_torch(monkeypatch, loader, want):
    """Codex review 2026-11-03: onnxruntime from another package + offline + torch models
    cached must keep recall working instead of failing on a missing ONNX file."""
    monkeypatch.delenv("SGM_BACKEND", raising=False)
    monkeypatch.setattr(onnx_backend, "available", lambda: True)
    monkeypatch.setattr(onnx_backend, "Encoder", _offline)
    monkeypatch.setattr(onnx_backend, "CrossEncoder", _offline)
    monkeypatch.setitem(sys.modules, "sentence_transformers", _FakeST())
    assert getattr(brain_ask, loader)("cpu") == want


def test_forced_onnx_failure_raises(monkeypatch):
    monkeypatch.setenv("SGM_BACKEND", "onnx")
    monkeypatch.setattr(onnx_backend, "Encoder", _offline)
    monkeypatch.setitem(sys.modules, "sentence_transformers", _FakeST())
    with pytest.raises(OSError):
        brain_ask.load_encoder("cpu")


def test_model_cache_is_not_temp(monkeypatch, tmp_path):
    """SGM_MODEL_CACHE is passed through; unset means huggingface_hub's persistent cache."""
    seen = []
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(hub, "hf_hub_download",
                        lambda repo, f, cache_dir=None: seen.append(cache_dir) or f)
    monkeypatch.setenv("SGM_MODEL_CACHE", str(tmp_path))
    onnx_backend._fetch("r", ["a"])
    monkeypatch.delenv("SGM_MODEL_CACHE")
    onnx_backend._fetch("r", ["a"])
    assert seen == [str(tmp_path), None]


REF_PAIRS = [
    ("query: как я думаю о памяти агентов", "Память агента: граф заметок и SQLite."),
    ("query: как я думаю о памяти агентов", "Рецепт борща с фасолью."),
    ("slow import", "torch takes 44 s to import on a laptop without GPU " * 80),
]
REF_LOGITS = [-0.85672015, -9.523896, 0.01399507]


@pytest.mark.skipif(not onnx_backend.available(), reason="lite extra not installed")
@pytest.mark.skipif("SGM_LIVE_MODELS" not in __import__("os").environ,
                    reason="downloads ~1.6 GB; set SGM_LIVE_MODELS=1")
def test_live_numbers_match_sentence_transformers():
    import numpy as np
    scores = onnx_backend.CrossEncoder().predict(REF_PAIRS)
    assert np.allclose(scores, REF_LOGITS, atol=1e-3), scores
    texts = ["query: как я думаю о памяти агентов",
             "passage: The reranker forgives coarse chunk boundaries. " * 60]
    emb = onnx_backend.Encoder().encode(texts)
    assert emb.shape == (2, 768)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return
    ref = SentenceTransformer(brain_ask.E5_MODEL, device="cpu").encode(
        texts, normalize_embeddings=True, convert_to_numpy=True)
    assert float((emb * ref).sum(1).min()) > 0.9999
