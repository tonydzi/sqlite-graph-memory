---
title: 'sqlite-graph-memory: Graph RAG over human-curated wikilinks on SQLite'
tags:
  - Python
  - retrieval-augmented generation
  - graph RAG
  - SQLite
  - personal knowledge management
authors:
  - name: Anton Dziatkovskii
    orcid: 0000-0001-7408-3054
    affiliation: 1
affiliations:
  - name: Palo Alto AI Research Lab, Palo Alto, CA, USA
    index: 1
date: 8 November 2026
bibliography: paper.bib
---

# Summary

`sqlite-graph-memory` is a retrieval layer for AI agents that work over a linked
note corpus (an Obsidian or Logseq vault, or any folder of Markdown). It combines
dense vector retrieval (multilingual-e5), a bounded one-hop expansion over the
*human-curated wikilink graph* that such corpora already contain, and cross-encoder
reranking. The index (vectors, chunk metadata, ledger) is stored in one SQLite file and served through the Model
Context Protocol (MCP) and a command-line interface. Wikilinks are read from the
notes at query time and are not copied into a graph database, which avoids a
separately maintained edge copy; the index itself still needs a refresh after edits. A per-turn memory ledger records what the agent retrieved and decided, which turns agent memory into
queryable telemetry (recording costs no model tokens; the agent's own output does).

# Statement of need

Users of linked note-taking systems place `[[wikilinks]]` by hand, often over years.
Each link is a deliberate claim that two notes are related. Researchers who study
retrieval over personal corpora, and practitioners who wire long-term memory into
coding agents, need a small and inspectable system that (a) consumes this human graph
as it is, (b) runs on a laptop against a private corpus (the software itself sends nothing off the
machine; what an MCP client then does with retrieved passages is outside its control),
and (c) exposes its own behaviour for measurement. The software targets these two
groups. It was extracted from a deployment used daily on a corpus of about 7,500
chunks.

# State of the field

The dominant Graph RAG recipes build the graph with a language model: GraphRAG
[@graphrag] extracts entities and community summaries, HippoRAG [@hipporag] builds
an open knowledge graph for associative retrieval, and Zep [@zep] maintains a
temporal knowledge graph for agent memory. Their extraction step costs tokens, can
invent edges, and needs a graph store that must be kept in sync with the corpus.
Obsidian-oriented MCP servers mostly offer keyword or vector search and ignore the
link structure. I built a new tool rather than contributing to these systems
because the design question is different: the graph here is given, not inferred, so
the contribution is the query-time use of an existing graph and the means to measure
whether it helps, not a better extractor.

# Software design

Three decisions shape the code. First, the wikilink walk is used only for candidate
generation: neighbours of the top vector hits enter the pool, and the cross-encoder
reranker remains the sole arbiter of final ranking. The expansion limit bounds candidate growth; its effect on quality is an empirical
question, because reranking can still promote irrelevant neighbours. A switch turns
the expansion off for ablation. Second, SQLite
is the only store, with a documented schema shipped inside the wheel, so an index is
one file that can be copied, diffed and backed up. Third, measurement is part of the
product: the software includes a paired evaluation protocol with a query-class-stratified benchmark (entity, theme, bridge, compare,
temporal) (`eval/`, `bench/`). This article does not claim that graph expansion improves retrieval; it provides the means to test that on private data. An owner of a
linked corpus can replicate the evaluation without disclosing the corpus.
Facts that change over time are kept bitemporally (`docs/bitemporal.md`), with
superseded links resolved through the entity gate. The test suite needs no model
download or network, because embedding libraries are imported lazily and the tests
use small fixture corpora, and one
continuous-integration check deliberately breaks the nDCG metric to prove that the
suite would notice.

# Research impact statement

The protocol and pilot telemetry are described in a companion preprint
[@dziatkovskii2026graphrag], and the operating context in [@dziatkovskii2026fleet];
both are archived on Zenodo with DOIs. The repository is public since 3 July 2026 and
has tagged releases, a changelog, continuous integration on Python 3.9 to 3.13, and a
published PyPI package. Outside contributors have merged pull requests, including
the ablation diagnostics added to the evaluation script, and the same author maintains a separate LangChain
retriever package built on it. The author uses the system daily as personal
agent memory. Adoption beyond the authors is small: at the time of writing the
repository has 14 stars and 14 forks (GitHub, 8 November 2026), and I do not claim more.

# AI usage disclosure

Generative AI tools were used in this project: Claude Code (Anthropic) and Codex
CLI (OpenAI) for code generation, refactoring, test scaffolding and documentation,
and Claude for drafting and copy-editing this paper. The human author set the
problem, the design decisions above and the evaluation protocol, and reviewed, edited
and validated all generated code and text. Correctness was checked by the automated
test suite, by the continuous-integration job that sabotages the metric on purpose,
and by review of the author's changes by a second model family.

# Acknowledgements

No external funding. I thank the outside contributors to the repository.

# References
