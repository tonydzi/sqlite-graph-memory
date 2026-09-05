# Changelog

What shipped, in plain words. Entries below v0.1.3 were written on 2026-09-05 from the tags and
release notes that already existed — the file itself did not exist until then, so those three are
recorded after the fact rather than backdated to look contemporaneous.

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
