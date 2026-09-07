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
date: 7 September 2026
bibliography: paper.bib
---

# Summary

`sqlite-graph-memory` is a retrieval layer for AI agents working over a linked
note corpus (Obsidian, Logseq, or any plain-Markdown wiki). It combines dense
vector retrieval (multilingual-e5), a bounded 1-hop expansion over the
*human-curated wikilink graph* already present in such corpora, and
cross-encoder reranking, persisting everything in a single SQLite file. The
graph is parsed from the notes at query time and never materialized in a graph
database, eliminating graph–corpus drift by construction. A zero-token
per-turn memory ledger records what the agent retrieved and decided, so agent
memory becomes queryable telemetry rather than opaque context.

Unlike the dominant Graph RAG recipe [@graphrag; @hipporag; @zep], no LLM
extraction pass builds the graph: every edge is a deliberate human assertion
that two notes are related. The wikilink walk is used strictly for candidate
generation — the cross-encoder reranker remains the sole arbiter of ranking —
which bounds the damage a noisy neighbourhood can do.

# Statement of need

A large population of corpora already contains a hand-built knowledge graph:
users of linked note-taking systems place `[[wikilinks]]` by hand, over years.
Existing Graph RAG tooling ignores this signal and rebuilds a graph by model
extraction, which hallucinates edges, requires a graph database, and drifts
out of sync with the corpus. Researchers studying retrieval over personal
corpora, and practitioners wiring long-term memory into agents, need a small,
inspectable system that (a) consumes the human graph as-is, (b) runs on a
laptop against a private corpus without sending data anywhere, and (c) exposes
its behaviour for measurement. `sqlite-graph-memory` is that system, extracted
from a production deployment used daily on a ~7,500-chunk corpus.

The design and its evaluation protocol — a query-class-stratified benchmark
(entity, theme, bridge, compare, temporal) with a paired statistical analysis
plan — are described in the companion preprint [@dziatkovskii2026graphrag];
pilot A/B telemetry from production use is reported there with its sample size
stated plainly. The evaluation protocol ships with the repository so that any
owner of a linked corpus can replicate the measurement on private data without
disclosing that data.

# Acknowledgements

The system is developed and operated within a production multi-machine agent
fleet [@dziatkovskii2026fleet].

# References
