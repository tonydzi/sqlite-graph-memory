---
title: 'sqlite-graph-memory: Graph RAG over human-curated wikilinks for agent memory'
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
date: 16 November 2026
bibliography: paper.bib
---

# Summary

`sqlite-graph-memory` is a retrieval layer for AI agents that work over a linked
note corpus (an Obsidian or Logseq vault, or any folder of Markdown). It combines
dense vector retrieval (`multilingual-e5-base`, 1,200-character chunks, top 60), a
bounded one-hop expansion over the *human-curated wikilink graph* that such corpora
already contain, and optional cross-encoder reranking (off by default since 0.3.7). The index (a float32 NumPy matrix and a pickled chunk-metadata file) lives in one directory and is served through the Model
Context Protocol (MCP) and a command-line interface; a SQLite file holds only the per-turn ledger and the A/B log. Wikilinks are read from the
notes at query time and are not copied into a graph database, which avoids a
separately maintained edge copy; the index itself still needs a refresh after edits. A per-turn ledger records what the agent retrieved and decided, which turns agent memory into
queryable telemetry.

# Statement of need

Users of linked note-taking systems place `[[wikilinks]]` by hand, often over years.
Each link is a deliberate claim that two notes are related. Researchers who study
retrieval over personal corpora, and practitioners who wire long-term memory into
coding agents, need a small and inspectable system that (a) consumes this human graph
as it is, (b) runs on a laptop against a private corpus (the software itself sends nothing off the
machine), and (c) exposes its own behaviour for measurement.

# State of the field

The dominant Graph RAG recipes build the graph with a language model: GraphRAG
[@graphrag] extracts entities and community summaries, HippoRAG [@hipporag] builds
an open knowledge graph for associative retrieval, and Zep [@zep] maintains a
temporal knowledge graph for agent memory. Their extraction step costs tokens and can
invent edges.
I built a new tool because the design question differs: the graph is given, not
inferred, so the contribution is query-time use of an existing graph and the means
to measure whether it helps, not a better extractor.

# Software design

Three decisions shape the code. First, the wikilink walk is used only for candidate
generation: neighbours of the top 15 vector hits (at most 40) enter the pool, and
ranking stays with the dense similarity or, when enabled, the cross-encoder. With dense ordering, the default, a neighbour outside the top 60 cannot enter the
top 12, so the graph changes results only with reranking or PageRank fusion.
Second, storage is
plain: the index is one directory of files, SQLite holds only the turn ledger and the A/B log
under a documented schema. Third, measurement is part of the
product: the software includes a paired evaluation protocol with a query-class-stratified benchmark (`eval/`)
and a public leaderboard (`bench/`). This article does not claim that graph expansion improves retrieval; it provides the means to test that on private data.
A query-side entity gate switches expansion off for short name or tool lookups, where
it added noise; a note's `superseded_by` link is still followed, so replaced facts
stay reachable. Persisted bitemporal edges exist only as a design note
(`docs/bitemporal.md`). The test suite needs no model
download or network, and one
continuous-integration check deliberately breaks the evaluation metric to prove that the
suite would notice.

# Research impact statement

The protocol and pilot telemetry are in a companion preprint
[@dziatkovskii2026graphrag], archived on Zenodo with a DOI. The repository is public since 3 July 2026 and
has tagged releases, a changelog, continuous integration on Python 3.9 to 3.13, and a
published PyPI package. Outside contributors have merged pull requests. Adoption beyond the author is small: 15 stars and 15 forks on
GitHub (16 November 2026), and I do not claim more.

The repository also ships `bench/`, an open leaderboard (memory-bench v0.1: a
synthetic vault of 202 notes and 91 questions) in which memory systems are compared
on recall@5 and on abstaining when the answer is absent. CI-reproduced rows are rerun in
continuous integration, and two reference rows came from an outside author. Its classes are fact (28 questions), bridge (15), paraphrase (10), paraphrase-bridge
(6), temporal (10) and two absent-answer classes (22). Recall measures retrieval
alone, as the share of a question's support notes found in the top 5 of the 69 answerable
questions; abstention is decided separately, by a different rule in each adapter (for
sqlite-graph-memory, a top reranker score below zero). Table 1 compares version 0.3.0, reranker on, with a plain dense
baseline (same chunker and bi-encoder, no graph hop, no reranker) and with BM25.

| system | R@5 | bridge R@5 | paraphrase R@5 | abstains when absent | false abstain |
|---|---:|---:|---:|---:|---:|
| plain dense RAG | 0.93 | 0.87 | 1.00 | 0.18 | 0.20 |
| sqlite-graph-memory, vector only | 0.88 | 0.97 | 0.80 | 0.45 | 0.26 |
| sqlite-graph-memory, graph mode | 0.85 | 1.00 | 0.50 | 0.45 | 0.26 |
| BM25 | 0.77 | 1.00 | 0.00 | 0.36 | 0.04 |

Table: memory-bench v0.1 (`bench/LEADERBOARD.md`).

Graph mode does not beat the dense baseline overall. Within the same reranked
pipeline, expansion moves bridge R@5 from 0.97 to 1.00 and paraphrase R@5 from 0.80
to 0.50 (ten questions). Plain dense retrieval shares the first stage and has no
reranker, so its 0.93 against 0.88 points at the reranker, though this is not a controlled
ablation. Retrieval can succeed while the answer is withheld: the reranker threshold
discards every paraphrase answer (false abstain 1.00 on that class). With so few
questions per class the numbers are directional, not significant, and come from one synthetic vault.

# AI usage disclosure

Generative AI tools were used in this project: Claude Code (Anthropic) and Codex
CLI (OpenAI) for code generation, refactoring, test scaffolding and documentation,
and Claude for drafting and copy-editing this paper. The human author set the
problem, the design decisions above and the evaluation protocol, and reviewed, edited
and validated all generated code and text. Correctness was checked by the test suite,
by a continuous-integration job that sabotages the metric on purpose, and by a second
model family reviewing the author's changes.

# Acknowledgements

No external funding.

# References
