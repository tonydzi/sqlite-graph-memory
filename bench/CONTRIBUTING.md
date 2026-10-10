# Add your system to the board (about an hour)

You need: a memory system that can ingest ~200 markdown files and answer a question,
Python 3.9+, and a GitHub account. No model download is needed for anything except
your own system.

## 1. Write an adapter (20 min)

```bash
git clone https://github.com/tonydzi/sqlite-graph-memory && cd sqlite-graph-memory
cp bench/adapters/template.py bench/adapters/my_system.py
```

Fill in two methods:

- `setup(vault_dir, work_dir)`: load every `.md` file in `bench/vault/` into your system.
- `ask(question)`: return `{"retrieved": [...], "answer": ..., "citations": [...]}`.
  - `retrieved` is your ranked list of note ids. A note id is the filename without `.md`.
  - `answer` is a string, or `None` when the vault does not contain the answer. Returning
    the string `"I don't know"` counts as an answer, and it will be scored as a wrong one.
  - `citations` are `{"note": id, "quote": text}`, and the quote has to be copied word for
    word from that note.

Rename the key in `SYSTEMS` to the name you want on the board (lowercase, `a-z0-9._-`).
Put everything you set by hand into `info()["config"]`: model, prompt, top-k, thresholds.

Sketches for systems people have asked about. These are untested pointers, not working
adapters. Check them against each project's current docs.

| system | in `setup` | in `ask` |
|---|---|---|
| mem0 | `Memory().add(note_text, user_id="bench", metadata={"note": id})` per note | `search(question, user_id="bench")`, map hits back to `metadata["note"]` |
| Letta | create an agent, insert each note into archival memory with its id in the text | send the question, read the archival-search tool calls for ids |
| Khoj | point a Khoj instance at `bench/vault/` and wait for indexing | `/api/search?q=...`, map file paths to ids |
| plain RAG | chunk + embed with your stack | top-k chunks -> ids; an LLM writes the answer and quotes a chunk |

## 2. Smoke run, then the full run (10 to 30 min)

```bash
python bench/run.py --system my-system --limit 5     # writes nothing
python bench/run.py --system my-system --submitted-by <your-github-handle> \
    --notes "one honest sentence: what you tuned, what is weak"
python bench/leaderboard.py
python bench/validate.py
```

`validate.py` is exactly what CI runs on your pull request. If it prints `VALID` locally, CI
agrees with it.

## 3. Open a pull request (5 min)

Commit three things: `bench/adapters/my_system.py`, `bench/results/my-system.json` and the
regenerated `bench/LEADERBOARD.md`.

What happens next:

- CI recomputes every metric from your per-question outputs and checks every quote against
  the vault. Typos in the numbers are impossible because you do not type the numbers.
- Your row is labelled **self-reported** until it can be rerun in CI. If your adapter needs
  no API key and no paid service, say so in the PR. We add it to the reproduce job, and the
  label changes to **yes**.
- A maintainer answers within 48 hours, including "no, and here is why".

## Adding questions or notes

The vault and the questions come from `bench/make_vault.py`, never from hand edits, and CI
rejects hand edits. To add a question class, extend the generator, rerun it, and rerun the
systems already on the board (`python bench/run.py --system <name>` for each in-repo
adapter). A new vault hash makes every older result stale on purpose, because numbers from
two different test sets cannot share a table.

Good classes to add, if you want one: entity-name collisions (two people share a first
name; the question gives only that), contradictions with no `superseded_by` marker, and
questions whose evidence sits in a meeting note instead of the entity's own card.

## Ground rules

- Do not tune on the questions. They are the test set. Tune on a vault you generate
  yourself with another seed: `build(seed=...)` in `make_vault.py`.
- Report what you ran, including the weak columns. The README of this bench reports ours.
- You keep the copyright to your code. No CLA. MIT, like the rest of the repo.
