# Changelog

What shipped, in plain words. Entries below v0.1.3 were written on 2026-09-05 from the tags and
release notes that already existed — the file itself did not exist until then, so those three are
recorded after the fact rather than backdated to look contemporaneous.

## v0.3.2 — 2026-10-24

**Hidden note files are skipped like hidden folders.** A note whose filename starts with a
dot, such as `.ideas.md` or `.draft.md`, used to be indexed when it sat in the vault root or a
visible folder, while the same file inside `.drafts/` was skipped. Now both are skipped, so
after re-indexing those notes stop showing up in answers. `BRAIN_INDEX_HIDDEN=1` brings back
the old behaviour and indexes hidden files and folders alike. `eval/build_gold.py` now walks the
vault with the indexer's own filter, so it follows the same rule and the same variable instead
of keeping its own copy (issue #25).

**The PageRank lane the README credits is now in the package** (#51). `ppr.py` runs
personalized PageRank over the `[[wikilink]]` graph, seeded by the dense hits, and fuses it with
the reranker by RRF while the reranker's top 5 stay where they are. Off by default (`SGM_PPR=1`
or `sgm-ask --ppr`); `eval/run_eval.py --ppr` scores it as its own mode. Any failure returns the
reranker's order. `tests/test_ppr.py` builds a hub note that pushes the reranker's rank-1 hit to
position 11 when the guard is removed, and goes red if it is.

**`sgm-stats <notes_dir>`: four health numbers for the link graph.** Orphan rate, links per
page, share of pages in the main connected component, and share of concept pages whose
`updated` date is older than 90 days, each with an ok/warn/fail verdict and a `--threshold-*`
flag. Exit code 0/1/2 for cron. Reuses the indexer's note filter and the recall wikilink parser,
so it measures the graph recall walks. On `bench/vault`: 26.2% orphans (fail), 4.65 links per
page, 100% in one component. `tests/test_vault_stats.py` crosses every threshold on a synthetic
vault and goes red when inbound links stop being counted.

**A spec you can paste.** `BITEMPORAL.md` turns the close-don't-delete rule from
`docs/bitemporal.md` into a block for any agent's `CLAUDE.md` or `AGENTS.md`: three fields
(`valid_from`, `valid_to`, `superseded_by`), five rules, and a one-line attribution as the
whole licence.

**`llms.txt` for agents that read before they install.** One page: what this is, the three
commands, MCP config for JSON clients, Claude Code and Codex, the bench numbers as they stand
on the leaderboard, the eval commands, every environment variable.

**The README says what it is for in its first line**: persistent memory for Claude Code,
Cursor and Codex over your own markdown notes.

**One star request, earned and once.** After a successful `sgm-ask --ab` in a terminal, one
stderr line asks for a GitHub star. It is printed once per machine (a marker file in
`~/.config/sqlite-graph-memory`, or `SGM_CONFIG_DIR`, created exclusively so two first runs
at once still print one line), never under `CI`, never when stdout or stderr is piped, never
when `BRAIN_AB_JSON` is set, and never with `SGM_NO_STAR_PROMPT=1`. If the line cannot be
written, the marker is removed so the next run asks again. `tests/test_star_prompt.py` covers
each guard and the call from `main()`, and goes red when any of them is removed.

## v0.3.1 — 2026-10-19

**Credit where it is due.** The MCP server that ships as `sgm-mcp` was written by Vardhman Gupta
([@Kaap10](https://github.com/Kaap10)) in PR #8: the server, its tests and its docs. His
follow-up fixes are the shared answer-file race (commit 3ce7061), the stdout fallback (PR #10)
and the swallowed read errors (PR #14).
He is now listed in the `authors` field of `pyproject.toml`, which is the attribution he asked for.

**Indexing works on Intel Macs again.** A fresh `pip install 'sqlite-graph-memory[embeddings]'`
on an x86_64 Mac with Python 3.12 crashed `sgm-index` with `NameError: name 'nn' is not
defined`. Torch stops at 2.2 there, transformers 5.x needs 2.6 and quietly disables torch,
and sentence-transformers 6.x requires transformers 5.x. The embeddings extra now caps
transformers below 4.50 and sentence-transformers below 5 on that one platform. Measured
both ways on the same machine: 0.3.0 from PyPI crashes, this build indexes the 202-note
bench vault cleanly (204 chunks).

**Listed in the official MCP Registry.** `server.json` describes the server as
`io.github.tonydzi/sqlite-graph-memory`, the README carries the `mcp-name` line the registry
checks, and the publish workflow lists the release after PyPI has it. The package also
installs the MCP server under its own name, `sqlite-graph-memory`, so registry clients can
start it with `uvx`.

**The README opens with three commands and their real output** on the bench vault, including
the honest caveat that vector-only ranks ahead of graph mode on that vault overall.

## bench-v0.1 — 2026-10-09

**A benchmark other people can run.** Until now every retrieval number in this repo came from
one private vault, so nobody could check them. `bench/` ships a generated 202-note vault about a
fictional lab, 91 questions in seven classes (22 of them have no answer in the vault), and a
leaderboard. `python bench/run.py --system bm25` runs end to end with nothing installed.

**You submit outputs, not numbers.** Each result file carries every per-question output, and
`bench/validate.py` recomputes every metric with the same scorer. CI rejects any file whose
claimed numbers do not follow from its own outputs, and a self-test forges a real result thirteen
ways on every push to prove the gate still catches it.

**The baseline is honest about itself.** sqlite-graph-memory retrieves both the old and the new
decision on "current deadline" questions and quotes the old one 6 times out of 6. It also
abstains on every paraphrased question, and it answers 55% of questions whose fact is missing.
On this vault, vector-only ranks ahead of graph mode. All of it is written out in
`bench/README.md`. Closes #16.

## v0.3.0 — 2026-10-07

*Release record, added 2026-10-16: the package went to PyPI on 2026-10-15 and this GitHub release object was cut a day later, so the tag points at a commit slightly after the upload (it adds only the workflow's `skip-existing` flag). Also in 0.3.0 as uploaded: the embeddings extra pins `numpy<2` on Intel Macs, where torch stops at 2.2.*

**You can install it now.** For three months this was a repository you had to clone, and for
an engineer deciding whether to try something, a clone is a much bigger ask than a command.
There is now a `pyproject.toml`, the modules live in `src/sqlite_graph_memory/`, and
`pip install sqlite-graph-memory` gives you five commands: `sgm-index`, `sgm-ask`,
`sgm-mcp`, `sgm-turnstate-hook` and `sgm-turnstate-show`.

**The MCP server moved inside the package.** It used to sit in `examples/` and shell out to
`../brain_ask.py`, which meant the one integration most agents actually use — the MCP tool —
only worked from a checkout. It is now `sgm-mcp`, so the client config is a command name
instead of a path into someone's clone, and the Stop-hook snippet lost its `/path/to/` too.

**The heavy half is optional, and says so.** The base install is numpy and nothing else,
about 20 MB; `sentence-transformers` and `torch` (~2 GB) come from
`pip install 'sqlite-graph-memory[embeddings]'`. Running the indexer without the extra used
to end in a `ModuleNotFoundError` traceback through site-packages — which reads like a broken
package rather than a missing option — and now ends in one line naming the extra to install.
`tests/test_optional_extra.py` covers that path, including the case it must *not* swallow: an
unrelated missing module still surfaces as itself, instead of sending you after the wrong fix.

**CI now proves the package is installable, not just that the tests pass.** A wheel can break
without a single test failing — a bad entry point, a file left out of the build, a dependency
that only exists on the author's machine. The new `installable` job builds the distributions,
runs `twine check`, installs the wheel into a venv with nothing else in it, and checks that
all five commands start, that `schema.sql` travelled with the wheel, and that a base install
prints instructions rather than a traceback.

## v0.2.1 — 2026-10-02

**The suite stopped being something a human remembers to run.** The 84 tests and the eval suite had
only ever been run by hand. `.github/workflows/tests.yml` now runs them on every push against
Python 3.11, 3.12 and 3.13, and carries one unusual step: it sets `EVAL_MUTANT=1`, which breaks the
retrieval pipeline on purpose, and fails the build if the eval suite still comes back green. A test
that has never been seen red is not evidence, so CI proves the red before it trusts the green.

- **Read this with AI**: a one-click entry point into Codex, ChatGPT or Claude with a prompt that
  asks the agent to extract the reusable patterns, plus the raw prompt for any other model, and a
  line pointing at the neighbouring repos in the memory layer.
- A date correction in the docs was reverted on 30 September: the original date was right, and the
  machine that "fixed" it was the one with the wrong clock.

## v0.2.0 — 2026-09-20

**An eval, and the seven upgrades it talked us out of.** `eval/build_gold.py` turns the vault's own
`[[wikilinks]]` into a retrieval test set — no hand labelling, no LLM calls — and `eval/run_eval.py`
scores it as Recall@12 / MRR / nDCG@12 in both retrieval modes, vector-only and vector+graph.

- **The eval imports the pipeline instead of copying it.** `brain_ask.py` keeps exactly the
  behaviour it had (proven byte-for-byte on nine runs over a 10 918-chunk index), but its stages are
  now importable functions — `load_index`, `query_sims`, `retrieve_candidates`, `index_by_basename`,
  `expand_1hop`, `rerank_candidates` — and `main()` is a CLI over them. An eval that re-implements
  the pipeline measures the re-implementation.
- **Four question classes**, because "does the graph help?" has four different answers: `title`,
  `body`, `bridge` (the note's own links are the gold answers) and `temporal` (a `superseded_by`
  frontmatter field points at the replacement).
- **The README now carries real numbers** from the home vault: what two changes improved (bridge
  Recall@12 0.251 -> 0.451) and what seven changes that "obviously should have helped" measurably did
  not — BGE-M3, two rerankers, a BM25 hybrid, a heading-path chunker, `sqlite-vec`, and more.
- **The ruler is testable too.** `EVAL_MUTANT=1` breaks nDCG deliberately, and `tests/test_eval.py`
  must go red under it: a suite that stays green while the metric is broken never tested the metric.

- **An external review panel broke the ruler before anyone else could.** Two engines
  independently found that `ndcg_at` credited the same gold note twice when one filename
  appeared twice in the ranking — which, since wikilinks address notes by filename, happens as
  soon as two folders hold a `foo.md`. `ndcg_at(['foo','foo'], ['foo'])` returned 1.63. Fixed,
  with the published title/temporal numbers corrected downward (0.921 -> 0.890 and 0.638 ->
  0.531); the before/after deltas were unaffected, because the same four questions of 184
  inflated both runs. Also from that review: ambiguous filenames are now excluded from gold
  and reported, a truncated `bridge` gold set records how many links were cut (9 of 30
  questions on our own vault), a run where the graph never fired says so instead of printing
  "the graph adds nothing", and an `EVAL_MUTANT=1` run refuses to write history at all.

**The MCP server got a runtime, examples, and three fixes to how it reports nothing.**
`examples/` now carries a runnable MCP server and its own README, so the recall pipeline is
reachable from an agent harness without reading the source first
([#5](https://github.com/tonydzi/sqlite-graph-memory/issues/5)).

- **The answer file was shared between calls.** Two recalls in flight wrote the same path, so one
  could return the other's text. It is scoped per call now, and `tests/test_mcp_server.py` pins the
  filename so the race can actually fail a test instead of passing on timing.
- **Empty is not an error, and an error is not empty.** An empty answer file used to fall back to
  whatever was on stdout, which meant a log line could come back as a recall result; the server
  returns `isError` instead
  ([#9](https://github.com/tonydzi/sqlite-graph-memory/issues/9)). A file it could not read is now
  reported separately from a recall that genuinely found nothing
  ([#13](https://github.com/tonydzi/sqlite-graph-memory/issues/13)).
- **The paper is out.** A JOSS draft (`paper.md` and its bibliography) with a readiness checklist,
  and `CITATION.cff` plus a cite section pointing at
  [10.5281/zenodo.22639718](https://doi.org/10.5281/zenodo.22639718).
- `brain_ask.py`'s `looks_like_entity` and `_links_in` are covered by tests, and the README
  describes its test coverage by shape instead of by a count that rots.

## v0.1.3 — 2026-09-05

Two ways the indexer quietly answered with the wrong text. Both were found on a real synced vault,
not in review, and both now have tests that run with no model download and no network.

- **Not everything ending in `.md` is a note.** The walk was a plain `rglob('*.md')`, which descends
  into dot-directories. On an Obsidian vault synced with Syncthing that pulled in `.stversions/`
  (Syncthing's old revisions), `*.sync-conflict-*` copies, `.obsidian/` plugin cache and `.git/`
  internals. This is worse than noise: a stale revision of a note is semantically almost identical
  to the live one, so it lands next to it in the top-K and answers the question with outdated
  content. Now skipped. `BRAIN_INDEX_HIDDEN=1` opts back in, because someone will legitimately
  keep notes in a hidden folder.
- **A byte-order mark ate the frontmatter.** Notes were read as plain `utf-8`, so a BOM'd file kept
  a leading U+FEFF and its first line was no longer `---`. The frontmatter regex then saw a note
  with no frontmatter and dropped its date and title, silently. Reading as `utf-8-sig` fixes it.
- Closes [#1](https://github.com/tonydzi/sqlite-graph-memory/issues/1) and
  [#2](https://github.com/tonydzi/sqlite-graph-memory/issues/2) (the smoke test that needs no model).

The two behaviours are now `iter_notes` and `read_note`, which is what made them testable at all —
`tests/test_index_notes.py`, seven cases shaped like a real synced vault.

## v0.1.2 — 2026-08-25

The bilingual regex, explained.

## v0.1.1 — 2026-08-04

The design note, and everything a stranger needs to run the pilot.

## v0.1.0 — 2026-08-04

The pilot as first published on 3 July 2026. Tagged a month later; the code did not change in
between, the tag was simply never cut.
