# -*- coding: utf-8 -*-
"""onnx_backend.py: the same two models brain_ask uses, run on ONNX Runtime, no torch.

Why it exists: on a laptop without a GPU the torch stack is the slow part of a recall,
not the search. Measured 2026-11-01 on a Windows machine through the MCP server:
`import torch` alone took 44 s, the encoder had not loaded at 90 s, and the client gave
up at its 60 s tool timeout. ONNX Runtime plus `tokenizers` is a ~60 MB install that
imports in under a second.

Same weights, same index: both model repos publish an official ONNX export of the very
checkpoint the torch path loads (`onnx/model.onnx`), so an index built with
sentence-transformers keeps working here and the other way round. The equivalence is
pinned by tests/test_onnx_backend.py against sentence-transformers outputs.

Models are cached by huggingface_hub in its persistent cache (HF_HUB_CACHE, default
~/.cache/huggingface/hub), never in the system temp dir, so Windows cleanup does not
force a second 1.6 GB download. SGM_MODEL_CACHE overrides the folder.

Input:  lists of strings (encode) or (query, passage) pairs (predict).
Output: float32 numpy arrays shaped like the sentence-transformers calls they replace.
Called by: brain_ask.load_encoder / load_reranker when the backend resolves to "onnx".
"""
import os

import numpy as np

E5_REPO = 'intfloat/multilingual-e5-base'
E5_FILES = ('onnx/model.onnx', 'onnx/tokenizer.json')
RERANK_REPO = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
RERANK_FILES = ('onnx/model.onnx', 'tokenizer.json')
MAX_LEN = 512   # sentence-transformers max_seq_length / max_length for both checkpoints


def available():
    """True when the lite stack (onnxruntime, tokenizers, huggingface_hub) is importable."""
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
        import huggingface_hub  # noqa: F401
    except ImportError:
        return False
    return True


def _fetch(repo, files):
    """Download (or reuse from cache) the named files of one model repo -> local paths."""
    from huggingface_hub import hf_hub_download
    cache = os.getenv('SGM_MODEL_CACHE') or None
    return [hf_hub_download(repo, f, cache_dir=cache) for f in files]


def prefetch():
    """Fetch both models now, so the first recall is not also a 1.6 GB download."""
    _fetch(E5_REPO, E5_FILES)
    _fetch(RERANK_REPO, RERANK_FILES)


def _session(model_path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3   # keep stdout/stderr clean for the MCP pipe
    return ort.InferenceSession(model_path, sess_options=opts,
                                providers=['CPUExecutionProvider'])


def _tokenizer(path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(path)
    tok.enable_truncation(max_length=MAX_LEN)
    tok.enable_padding()
    return tok


def _feed(sess, encs):
    ids = np.array([e.ids for e in encs], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
    feed = {'input_ids': ids, 'attention_mask': mask}
    names = {i.name for i in sess.get_inputs()}
    if 'token_type_ids' in names:
        feed['token_type_ids'] = np.array([e.type_ids for e in encs], dtype=np.int64)
    return {k: v for k, v in feed.items() if k in names}, mask


class Encoder:
    """Drop-in for SentenceTransformer(E5_MODEL).encode: mean pooling, optional L2 norm."""

    def __init__(self):
        model, tok = _fetch(E5_REPO, E5_FILES)
        self.sess = _session(model)
        self.tok = _tokenizer(tok)

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True,
               batch_size=32, show_progress_bar=False):
        out = []
        for s in range(0, len(texts), batch_size):
            feed, mask = _feed(self.sess, self.tok.encode_batch(list(texts[s:s + batch_size])))
            hidden = self.sess.run(None, feed)[0]                  # (n, seq, dim)
            m = mask[..., None].astype(np.float32)
            vec = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            if normalize_embeddings:
                vec = vec / np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12, None)
            out.append(vec.astype(np.float32))
            if show_progress_bar:
                print('  encoded %d/%d' % (min(s + batch_size, len(texts)), len(texts)), flush=True)
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)


class CrossEncoder:
    """Drop-in for sentence_transformers.CrossEncoder(RERANK_MODEL).predict.

    Returns the raw logit, as sentence-transformers does for this checkpoint (measured
    2026-11-03 with sentence-transformers 6.1: -0.857 / -9.52 / 0.014 on the test pairs,
    no sigmoid), so `rr=` in the CLI output reads the same whichever backend ran.
    """

    def __init__(self):
        model, tok = _fetch(RERANK_REPO, RERANK_FILES)
        self.sess = _session(model)
        self.tok = _tokenizer(tok)

    def predict(self, pairs, batch_size=32):
        pairs = [(str(a), str(b)) for a, b in pairs]
        out = []
        for s in range(0, len(pairs), batch_size):
            feed, _ = _feed(self.sess, self.tok.encode_batch(pairs[s:s + batch_size]))
            out.append(self.sess.run(None, feed)[0].reshape(-1))
        return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)
