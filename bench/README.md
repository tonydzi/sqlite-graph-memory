> Mycroft here, Anton's synthetic AI cofounder. AI agents (me, plus Codex as the reviewer) built this benchmark: the generator, the scorer, the CI gate and the baseline runs. A human maintainer, [@tonydzi](https://github.com/tonydzi), owns it and reviews every submission. Machines are not taking over, they are opening pull requests and waiting for someone to merge them.

# memory-bench v0.1: retrieval from a second brain

This bench asks one question: when an agent's memory is a folder of markdown notes linked
with `[[wikilinks]]`, can the memory find the right note, quote it word for word, and say
"not in the vault" when the answer really is not there?

- **202 synthetic notes** about a fictional lab: 50 people, 10 teams, 12 sites, 20 tools,
  25 projects, 32 decisions (10 of them later replaced), 40 meeting notes that act as
  distractors, 12 concept notes, and one home note. Nothing in them is real. They contain
  no personal data and no names of real people, and the generator is in the repo.
- **91 questions** in seven classes, each with gold answers, the support notes, and the exact
  evidence sentence:

| class | n | what it tests |
|---|---:|---|
| `fact` | 28 | one note holds the answer, and the question names the entity |
| `bridge` | 15 | the answer needs two notes joined by a wikilink |
| `paraphrase` | 10 | the question describes a project's goal and shares **zero** content words with the note |
| `paraphrase_bridge` | 6 | the same paraphrase, followed by one wikilink hop |
| `temporal` | 10 | a decision was replaced later, and the question asks for the current value or the original one |
| `absent_entity` | 10 | the question asks about a person, project, tool or site that does not exist |
| `absent_attribute` | 12 | the entity exists, but the vault never records the attribute the question asks for |

The 22 `absent_*` questions have no answer. A system that answers them anyway is making
the answer up.

## Run it

```bash
git clone https://github.com/tonydzi/sqlite-graph-memory && cd sqlite-graph-memory
python bench/run.py --system bm25        # one command, stdlib only, about a second
python bench/validate.py                 # what CI runs
```

To reproduce the sqlite-graph-memory rows (CPU, 15 minutes on a laptop, 1.6 GB of models
on the first run):

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r bench/requirements.txt
python bench/run.py --system sqlite-graph-memory
python bench/run.py --system sqlite-graph-memory-vector
```

**[Leaderboard →](LEADERBOARD.md)** · **[Add your system in about an hour →](CONTRIBUTING.md)**

## Metrics

| metric | over | means |
|---|---|---|
| `recall@5`, `recall@10` | answerable | share of support notes in the top k |
| `answer_acc` | answerable | the answer contains a gold answer string (word-bounded, case-insensitive) |
| `citation_verbatim` | answerable | some citation quotes at least 15 characters that appear word for word in a support note and contain the answer |
| `citation_faithful` | all citations | share of quotes that really appear in the note they cite. Invented quotes lower this |
| `abstain_correct` | unanswerable | share where the system returned no answer |
| `false_abstain` | answerable | share where it returned no answer although one exists (lower is better) |

There is no combined score. A system that retrieves well and makes up an answer to a
missing fact is not a slightly worse version of one that abstains. It fails in a different
way. Read the columns.

## Baseline: what our own system does badly

These are the numbers for [sqlite-graph-memory](../README.md) at v0.3.0 on CPU, in both
retrieval modes: graph with the entity gate (`sgm-ask --graph`) and vector-only (`sgm-ask`). The library retrieves and does not write
answers, so its answers come from a deliberately simple extractor: the sentence in the top
3 notes that shares the most words with the question. Abstention follows one rule that was
fixed before the first run: return nothing when the reranker's best score is below 0.

| mode | R@5 | R@10 | answer | verbatim cite | abstains when absent | false abstain |
|---|---:|---:|---:|---:|---:|---:|
| graph (`--graph`) | 0.85 | 0.91 | 0.33 | 0.33 | 0.45 | 0.26 |
| vector (default) | 0.88 | 0.92 | 0.33 | 0.33 | 0.45 | 0.26 |

What the per-class table in [LEADERBOARD.md](LEADERBOARD.md) shows, worst first:

1. **The current value loses to the replaced one.** On the six "what is the current
   deadline" questions, retrieval scores a perfect R@5 of 1.00, and the extracted answer
   cites the replaced decision in **6 of 6**. The old and the new decision read almost the
   same, the old one ranks first every time, and nothing in the pipeline reads
   `superseded_by`. A recall metric alone would have reported this class as solved. This is
   the most useful thing the bench has found so far, and it is the case the bi-temporal
   design note ([`docs/bitemporal.md`](../docs/bitemporal.md)) was written for. That design
   is not in the retrieval path yet.
2. **Paraphrase gets found and then thrown away.** Semantic retrieval finds most of the
   paraphrased goals in the top 5 (R@5 0.80 vector, 0.50 graph, where BM25 scores 0.00). But the reranker's
   best score on those questions is always below 0 (median −4.8), so the abstention rule
   discards every answer: false abstain is 1.00 on `paraphrase` and on `paraphrase_bridge`.
   One threshold is wrong for two different jobs.
3. **Missing facts get answered anyway.** The system abstains on 60% of questions about
   entities that do not exist and on 33% of questions asking for an attribute the vault
   never records. The rest of the time it returns a confident sentence about something
   else.
4. **Two-hop questions retrieve both notes and answer from the first one.** On `bridge`,
   R@5 is 1.00 and answer accuracy is 0.13 (BM25 gets the same 0.13). That number belongs to the extractor and not to
   retrieval, and it is the main reason an LLM reader would move this row.
5. **On this vault the graph costs more than it gives.** The entity gate turns the hop off
   for only 5 of the 91 questions (it only fires on queries of five words or fewer), so the
   graph really runs on the other 86. It helps where it should: `bridge` R@5 goes from 0.97
   to 1.00. It hurts more on `paraphrase`, where R@5 drops from 0.80 to 0.50, because
   link neighbours of the wrong seed notes crowd the reranker pool. Overall, vector-only
   ranks ahead. That matches what the home-vault eval found per class (graph helps
   theme and link-following queries, hurts lookups), and it makes the case for the
   [ablation matrix (#19)](https://github.com/tonydzi/sqlite-graph-memory/issues/19):
   the graph should switch on for the questions that need a hop, not for every question.

BM25 is on the board as the floor that needs no model at all. It wins on names and gets
0.00 on paraphrase, which shows the paraphrase class works as intended.

## How the board stays honest

- **You submit outputs, not numbers.** Each result file carries every per-question output:
  ranked note ids, the answer, the quotes. CI recomputes every metric with the same scorer
  `run.py` uses and rejects the file if a single claimed number differs. A quote that does
  not appear in the cited note is caught the same way.
- **A forgery test runs on every push.** `validate.py --selftest` forges a real result thirteen
  ways (an inflated headline, a flipped class metric, invented quotes, a dropped question, a
  different vault, a self-awarded CI badge, a fake quote padded with real ones, an invented
  class, a +0.0001 nudge, a version, URL, notes or handle that draws extra cells into the board) and fails the
  build if any forgery gets through. Citations are capped at 5 per question and deduplicated,
  so repeating a real quote cannot dilute a made-up one.
- **Reruns where possible.** BM25 is rerun on every push and has to match exactly. The two
  sqlite-graph-memory rows are rerun on CPU from a fresh index and have to match within
  0.02 on every metric, and none of the 91 questions may differ in top-10, answer or
  quotes, so a row cannot hand-edit even one answer and keep its badge. Which systems CI reruns is decided by [`ci_reproduced.txt`](ci_reproduced.txt), a
  maintainer-owned list, and never by the result file. Rows that CI cannot rerun, because
  they need an API key or a paid service, are labelled **self-reported**.
- **The vault comes from code and is never hand-edited.** `make_vault.py --check` regenerates
  it, and every result carries the vault hash and the questions hash. If the generator
  changes, the old results become stale on purpose.

What this does not stop: the gold answers are public, so a self-reported row could simply
copy them into its outputs and score perfectly. Validation cannot tell a copied answer from a
correct one. Only a rerun can, and that is the reason **self-reported** and **yes** sit in
different columns. A maintainer can also regenerate the vault with an unpublished seed
(`make_vault.build(seed=...)`) and rerun any keyless adapter against questions that nobody
has seen.

## Limits, stated up front

- 202 notes is small. Dense retrieval at this size is easier than in a 100k-note vault,
  so the bench measures failure modes, and it does not show how a system scales.
- The text is English only and comes from templates. The sentences are cleaner than real
  notes, and that flatters extractive systems.
- A question tagged `absent_attribute` is checked against the generator's vocabulary.
  The words "phone", "birthday", "license" and "owns the building" occur nowhere in the
  vault. The generator only asserts the absence, it does not prove it in the semantic sense.
- The answer extractor in `adapters/_extract.py` sets the floor and is not meant as a
  reader. Put an LLM on top of the same retrieval and submit that as a separate row.

## Files

```
bench/
  make_vault.py        generator: vault + questions, deterministic, --check for CI
  vault/               202 generated notes (do not edit; edit the generator)
  questions.jsonl      91 questions with answers, support notes, evidence
  bench_core.py        scorer + result-file contract (stdlib)
  run.py               run one system -> results/<name>.json
  validate.py          the CI gate (+ --selftest, --reproduce)
  leaderboard.py       results/*.json -> LEADERBOARD.md
  adapters/            template.py, bm25.py, sgm.py (+ _extract.py, the shared answer floor)
  results/             one JSON per system
```

Issues this closes or starts:
[#16 synthetic mini-vault](https://github.com/tonydzi/sqlite-graph-memory/issues/16) (closed by it),
[#19 ablation matrix](https://github.com/tonydzi/sqlite-graph-memory/issues/19) (its first two cells: 0 hops and 1 hop).
