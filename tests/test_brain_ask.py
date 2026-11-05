# -*- coding: utf-8 -*-
"""Unit tests for pure functions in brain_ask.py.

Covers looks_like_entity() and _links_in() without model downloads or GPUs.
Addresses issue #4:
- looks_like_entity(): verifies entity gate switches off graph expansion for
  proper nouns, CamelCase, snake_case, known tools, and non-Latin (Cyrillic) names,
  while keeping graph on for themes and acronyms.
- _links_in(): table-driven parsing of wikilinks in markdown files using tmp_path,
  including aliases, anchors, multiple links per line, fenced code blocks, and no links.
"""
import os
import subprocess
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sqlite_graph_memory.brain_ask import looks_like_entity, _links_in  # noqa: E402


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # Known tools and products
        ("obsidian", True),
        ("sqlite", True),
        ("whatsapp", True),
        # Snake_case identifiers
        ("turnstate_hook", True),
        ("my_project", True),
        # CamelCase tools and libraries
        ("NotebookLM", True),
        ("PyTorch", True),
        ("FastAPI", True),
        # Proper nouns and names (single word or two tokens)
        ("Feynman", True),
        ("Turing", True),
        ("Alan Turing", True),
        ("Richard Feynman", True),
        # Proper noun in short query (idx > 0)
        ("notes on Feynman", True),
        ("who is Turing", True),
        # Non-Latin script (Cyrillic entities and names)
        ("Тьюринг", True),
        ("Алан Тьюринг", True),
        ("«Тьюринг»", True),
        ("заметки про Тьюринга", True),
        # Empty / whitespace / None
        ("", False),
        ("   ", False),
        (None, False),
        # Query with > 5 tokens
        ("this is a very long query with too many words", False),
        # Lowercase common-noun phrases / themes (graph stays on)
        ("graph memory", False),
        ("approaches to agent memory", False),
        ("second brain", False),
        # Non-Latin script lowercase themes
        ("векторный поиск", False),
        ("теория графов", False),
        # Bare topic acronyms (uppercase <= 3 chars)
        ("AI", False),
        ("RAG", False),
        ("DAO", False),
        ("РФ", False),
        # Multi-word sentence query where only first word is capitalized
        ("How does graph memory work", False),
    ],
)
def test_looks_like_entity(query, expected):
    assert looks_like_entity(query) is expected


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        # A file with no links at all
        ("A plain note with no wikilinks inside.", []),
        # A link with an alias
        ("See [[note|display text]] for details.", ["note"]),
        # A link with a heading anchor
        ("Refer to [[note#section]] above.", ["note"]),
        # A link with both heading anchor and alias
        ("Check [[note#section|display text]] here.", ["note"]),
        # A line with two links on it
        ("Two links on one line: [[first]] and [[second]].", ["first", "second"]),
        # Multiple links across lines
        ("Line one [[alpha]].\nLine two [[beta]].", ["alpha", "beta"]),
        # A wikilink inside a fenced code block (backticks) should NOT count
        ("```python\nx = [[code_note]]\n```", []),
        # A wikilink inside a fenced code block (tildes) should NOT count
        ("~~~markdown\n[[ignored_note]]\n~~~", []),
        # Mixed note: links outside code blocks count, links inside code blocks do not
        (
            "Link outside [[valid_note]].\n"
            "```\n"
            "[[inside_code]]\n"
            "```\n"
            "Another outside [[another_valid#anchor|alias]].",
            ["valid_note", "another_valid"],
        ),
        # A wikilink inside inline code should NOT count
        ("Write `[[example]]` to link.", []),
        # Inline code hides its link, the surrounding sentence still counts
        ("Outside [[real]]. Inside `[[example]]` only.", ["real"]),
    ],
)
def test_links_in_content_cases(tmp_path, content, expected):
    note_file = tmp_path / "note.md"
    note_file.write_text(content, encoding="utf-8")
    assert _links_in(note_file) == expected


def test_links_in_nonexistent_file(tmp_path):
    missing_file = tmp_path / "does_not_exist.md"
    assert _links_in(missing_file) == []


@pytest.mark.parametrize(
    ("frontmatter", "expected"),
    [
        # Wikilink value -> resolved via WIKILINK_RX, same as a body link.
        ('superseded_by: "[[decision-001]]"', ["decision-001"]),
        # Bare (non-wikilink) value -> returned as-is.
        ('superseded_by: decision-001', ["decision-001"]),
        ('superseded_by: "decision-001"', ["decision-001"]),
        # No superseded_by field at all.
        ('title: "decision-005"', []),
        # Empty quoted value resolves to [].
        ('superseded_by: ""', []),
    ],
)
def test_superseded_links_in_frontmatter_cases(tmp_path, frontmatter, expected):
    from sqlite_graph_memory.brain_ask import _superseded_links_in

    note = tmp_path / "note.md"
    note.write_text("---\n%s\n---\n# body\nUnrelated [[other-note]] link.\n" % frontmatter,
                     encoding="utf-8")
    assert _superseded_links_in(note) == expected


def test_superseded_links_in_no_frontmatter(tmp_path):
    from sqlite_graph_memory.brain_ask import _superseded_links_in

    note = tmp_path / "note.md"
    note.write_text("# body\nNo frontmatter here, just [[a-link]].\n", encoding="utf-8")
    assert _superseded_links_in(note) == []


def test_superseded_links_in_nonexistent_file(tmp_path):
    from sqlite_graph_memory.brain_ask import _superseded_links_in

    missing_file = tmp_path / "does_not_exist.md"
    assert _superseded_links_in(missing_file) == []


def test_expand_1hop_resolves_folder_and_md_suffix_links(tmp_path):
    """Issue #71 audit: expand_1hop resolves [[Sub/c]] and [[c.md]] targets."""
    import numpy as np
    from sqlite_graph_memory.brain_ask import expand_1hop, index_by_basename

    sub = tmp_path / "sub"
    sub.mkdir()
    a_path = tmp_path / "a.md"
    b_path = tmp_path / "b.md"
    c_path = sub / "c.md"
    a_path.write_text("[[Sub/c]]\n", encoding="utf-8")
    b_path.write_text("[[c.md]]\n", encoding="utf-8")
    c_path.write_text("# c\n", encoding="utf-8")

    meta = [
        {"path": str(a_path), "title": "a", "snippet": ""},
        {"path": str(b_path), "title": "b", "snippet": ""},
        {"path": str(c_path), "title": "c", "snippet": ""},
    ]
    sims = np.array([0.9, 0.8, 0.4], dtype="float32")
    by_base = index_by_basename(meta)

    assert expand_1hop(sims, meta, [0], by_base=by_base) == [2]
    assert expand_1hop(sims, meta, [1], by_base=by_base) == [2]


def test_expand_1hop_prefers_exact_vault_relative_path_over_suffix_collision(tmp_path):
    """When both sub/c.md and other/sub/c.md exist, [[sub/c]] resolves to sub/c.md even if other/sub/c.md has higher similarity."""
    import numpy as np
    from sqlite_graph_memory.brain_ask import expand_1hop, index_by_basename

    (tmp_path / "sub").mkdir()
    (tmp_path / "other" / "sub").mkdir(parents=True)
    a_path = tmp_path / "a.md"
    exact_c = tmp_path / "sub" / "c.md"
    nested_c = tmp_path / "other" / "sub" / "c.md"
    a_path.write_text("[[sub/c]]\n", encoding="utf-8")
    exact_c.write_text("# exact c\n", encoding="utf-8")
    nested_c.write_text("# nested c\n", encoding="utf-8")

    meta = [
        {"path": str(a_path), "title": "a", "snippet": ""},
        {"path": str(exact_c), "title": "c", "snippet": ""},
        {"path": str(nested_c), "title": "c", "snippet": ""},
    ]
    sims = np.array([0.9, 0.3, 0.85], dtype="float32")
    by_base = index_by_basename(meta)

    assert expand_1hop(sims, meta, [0], by_base=by_base) == [1]


def test_expand_1hop_single_subfolder_layout_and_precomputed_root(tmp_path, monkeypatch):
    """Pin all-notes-under-one-subfolder resolution and ensure commonpath is not called per link."""
    import numpy as np
    import sqlite_graph_memory.brain_ask as ba

    sub = tmp_path / "sub"
    sub.mkdir(parents=True)
    a_path = sub / "a.md"
    c_path = sub / "c.md"
    a_path.write_text("# a\nSee [[sub/c]].\n", encoding="utf-8")
    c_path.write_text("# c\nTarget.\n", encoding="utf-8")

    meta = [
        {"path": str(a_path), "title": "a", "snippet": ""},
        {"path": str(c_path), "title": "c", "snippet": ""},
    ]
    sims = np.array([0.9, 0.5], dtype="float32")
    by_base = ba.index_by_basename(meta)

    def _forbid_commonpath(_paths):
        raise AssertionError("os.path.commonpath should not run inside expand_1hop when by_base is precomputed")

    monkeypatch.setattr(ba.os.path, "commonpath", _forbid_commonpath)
    assert ba.expand_1hop(sims, meta, [0], by_base=by_base) == [1]


def test_expand_1hop_only_superseded_follows_frontmatter_not_body_links(tmp_path):
    """Issue #93: an entity-gated query ("Project Heron deadline (2025-02-07)") must
    still reach an outdated note's superseded_by replacement at topk=1, while general
    (body) wikilinks stay gated off."""
    import numpy as np
    from sqlite_graph_memory.brain_ask import expand_1hop, index_by_basename, looks_like_entity

    query = "Project Heron deadline (2025-02-07)"
    assert looks_like_entity(query) is True  # exactly the query shape the entity gate fires on

    outdated_path = tmp_path / "decision-005.md"
    replacement_path = tmp_path / "decision-001.md"
    unrelated_path = tmp_path / "unrelated-note.md"
    outdated_path.write_text(
        '---\n'
        'superseded_by: "[[decision-001]]"\n'
        '---\n'
        '# Project Heron deadline (2025-02-07)\n'
        'Outdated. See also [[unrelated-note]].\n',
        encoding="utf-8",
    )
    replacement_path.write_text("# decision-001\nCurrent decision.\n", encoding="utf-8")
    unrelated_path.write_text("# unrelated-note\nNot the replacement.\n", encoding="utf-8")

    meta = [
        {"path": str(outdated_path), "title": "decision-005", "snippet": ""},
        {"path": str(replacement_path), "title": "decision-001", "snippet": ""},
        {"path": str(unrelated_path), "title": "unrelated-note", "snippet": ""},
    ]
    sims = np.array([0.9, 0.5, 0.8], dtype="float32")
    by_base = index_by_basename(meta)
    base_order = [0]  # topk=1: only the outdated note enters the candidate pool

    # Fails on current main: expand_1hop has no only_superseded param at all yet, and
    # the ordinary hop would ALSO pull in unrelated-note via the body wikilink.
    added_full = expand_1hop(sims, meta, base_order, by_base=by_base)
    assert added_full == [1, 2]

    added_superseded_only = expand_1hop(sims, meta, base_order, by_base=by_base, only_superseded=True)
    assert added_superseded_only == [1]  # decision-001 only, never unrelated-note


def test_missing_embedding_index_reports_error_and_exits_nonzero(tmp_path):
    """A missing index must reach CLI callers as an error, not empty results."""
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["BRAIN_INDEX_DIR"] = str(tmp_path / "missing-index")
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(repo_root / "src"), env.get("PYTHONPATH")])
    )

    result = subprocess.run(
        [sys.executable, "-m", "sqlite_graph_memory.brain_ask", "agent memory"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert "need an embedding index" in result.stderr
    assert "index_notes.py first" in result.stderr
