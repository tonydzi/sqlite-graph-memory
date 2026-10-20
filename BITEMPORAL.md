# Bi-temporal memory spec v1.0

A block you can paste into an agent's instruction file (`CLAUDE.md`, `AGENTS.md`,
`.cursor/rules`, a system prompt) so the agent stops overwriting what it knew and starts
closing it instead. It is the rule this repo's design note
[`docs/bitemporal.md`](docs/bitemporal.md) applies to graph edges, restated for any memory
record an agent writes: a note, a fact in a memory file, a row in a table.

The licence is one line. Keep the attribution line inside the block when you paste it;
that is all we ask.

## The block

Copy everything between the two rules.

---

```markdown
## Memory records: close, never overwrite

Adapted from the bi-temporal memory spec v1.0, https://github.com/tonydzi/sqlite-graph-memory/blob/main/BITEMPORAL.md

Every durable fact I record (memory file entry, note, table row) carries a validity window:

- `valid_from`: when the fact became true or was first observed (ISO-8601 date or datetime).
- `valid_to`: when it stopped being true. Empty means it is still true.
- `superseded_by`: what replaced it (a note name, a [[wikilink]], a row id). Empty if nothing did.

Rules:

1. When a fact changes, I do not edit it in place and I do not delete it. I set `valid_to`
   (and `superseded_by` if there is a replacement) on the old record, then write the new
   record with its own `valid_from`.
2. When I recall, I prefer records whose `valid_to` is empty. A closed record is history:
   I may cite it only as "was true until <valid_to>", never as the current state.
3. To answer "what was true on date D", I keep records with `valid_from <= D` and
   (`valid_to` empty or `valid_to > D`).
4. A wrong record is closed the same way, with a one-line reason, not erased. The
   history of a mistake is part of the memory.
5. Deleting a memory record outright is something only the human who owns the memory
   decides.
```

---

## Where the fields go

| Store | How the three fields look |
|---|---|
| Markdown note | YAML frontmatter: `valid_from: 2026-10-19`, `valid_to:`, `superseded_by: "[[newer-note]]"` |
| One-line memory file | a suffix on the line: `(valid_from 2026-10-19; valid_to 2026-11-02; superseded_by deploy-v2)` |
| SQLite table | three `TEXT` columns, plus an index on `valid_to` so "open records only" stays fast |

The eval in this repo already reads the frontmatter form: its `temporal` question class
checks that an outdated note's title surfaces the note named in its `superseded_by` field
(see the README, "Measure before you swap").

## Why closing beats overwriting

- **History survives.** "When did this stop being true?" has an answer.
- **Recall can prefer the present** by filtering on an empty `valid_to`, without losing
  the past.
- **Mistakes stay auditable.** A corrected fact shows what was believed and when it was
  fixed.

What it costs: records only grow, so a store that lives for years wants a periodic move of
long-closed records into an archive table or folder. That is housekeeping, not deletion.
The SQL version with diff-upsert and confidence decay is in
[`docs/bitemporal.md`](docs/bitemporal.md).

## Versions

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-10-19 | First portable version: three fields, five rules, store table. |

A change to the rules bumps the minor version; a change that makes an old block wrong
bumps the major one. The attribution line names the version you pasted, so you can diff it
against this table.
