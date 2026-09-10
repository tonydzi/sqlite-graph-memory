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
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain_ask import looks_like_entity, _links_in  # noqa: E402


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
    ],
)
def test_links_in_content_cases(tmp_path, content, expected):
    note_file = tmp_path / "note.md"
    note_file.write_text(content, encoding="utf-8")
    assert _links_in(note_file) == expected


def test_links_in_nonexistent_file(tmp_path):
    missing_file = tmp_path / "does_not_exist.md"
    assert _links_in(missing_file) == []
