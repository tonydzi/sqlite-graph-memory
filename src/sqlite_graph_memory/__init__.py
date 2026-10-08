"""Graph RAG on SQLite for AI agents — vectors for entry points, hand-written
``[[wikilinks]]`` as the graph, cross-encoder rerank on top.

The package ships four command-line entry points rather than a library API,
because that is how the pilot is actually used:

``sgm-index``
    Walk a folder of markdown notes, extract ``[[wikilinks]]``, embed the
    chunks and write everything into one SQLite file.
``sgm-ask``
    Answer a question over that file: vector search for entry points, one hop
    along the wikilink graph, then rerank.
``sgm-turnstate-hook`` / ``sgm-turnstate-show``
    Record and inspect per-session agent memory ("what happened last turn").

Importing the submodules directly is supported and stable — the tests do it —
but the heavy ML dependencies are loaded lazily inside the functions that need
them, so importing this package costs nothing beyond ``numpy``. Install the
``embeddings`` extra to get ``sentence-transformers`` and ``torch``; without it
the indexer and the reranker will tell you what is missing instead of failing
on an ``ImportError`` at startup.
"""

__version__ = "0.3.0"

__all__ = ["__version__"]
