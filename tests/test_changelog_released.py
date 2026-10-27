"""A released CHANGELOG section is frozen.

Once `vX.Y.Z` is tagged, its `## vX.Y.Z` section is what PyPI users read as
"what shipped in that version". On 2026-10-24 the #36 entry landed inside the
already-released v0.3.2 section, so the changelog claimed v0.3.2 contained a
change it did not; v0.3.3 (#59) had to move it out by hand. This test compares
every tagged section at HEAD with the same section in the CHANGELOG as it was
at the tag, and fails if they differ.

New entries belong under a version that has no tag yet. A deliberate edit of a
released section (a typo, a broken link) is allowed when the section carries a
line `<!-- edited after release: <reason> -->`, so the edit is visible to readers.
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HEADER = re.compile(r"^## (v\d+\.\d+\.\d+)\b", re.M)
OVERRIDE = "<!-- edited after release:"


def sections(text):
    """{version: body} for every `## vX.Y.Z` header, body up to the next `## `."""
    out = {}
    marks = list(re.finditer(r"^## ", text, re.M))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        chunk = text[m.start():end]
        h = HEADER.match(chunk)
        if h:
            # The header line itself may change: v0.3.1 was tagged as "v0.3.1 — unreleased"
            # and got its date afterwards. Only the body is frozen.
            out[h.group(1)] = chunk.split("\n", 1)[1].strip() if "\n" in chunk else ""
    return out


def frozen_violations(head_text, tagged_texts):
    """Versions whose released section changed. tagged_texts: {tag: CHANGELOG at that tag}."""
    head = sections(head_text)
    bad = []
    for tag, old_text in tagged_texts.items():
        then = sections(old_text).get(tag)
        now = head.get(tag)
        if then is None:
            continue  # section written after the fact (pre-2026-09-05 history)
        if now is None:
            bad.append(f"{tag}: released section was removed")
        elif now != then and OVERRIDE not in now:
            bad.append(f"{tag}: released section changed after the tag")
    return bad


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def test_released_sections_are_frozen():
    tags = [t for t in _git("tag", "--list", "v*").stdout.split() if re.fullmatch(r"v\d+\.\d+\.\d+", t)]
    if not tags:
        if os.environ.get("CI"):
            pytest.fail("no version tags in the checkout: CI must fetch tags (fetch-depth: 0), "
                        "otherwise this guard is blind")
        pytest.skip("no version tags in this checkout")
    tagged = {}
    for t in tags:
        r = _git("show", f"{t}:CHANGELOG.md")
        if r.returncode == 0:
            tagged[t] = r.stdout
    head_text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    bad = frozen_violations(head_text, tagged)
    assert not bad, ("released CHANGELOG sections must not change; put new entries under an "
                     "untagged version, or mark a deliberate fix with "
                     "'<!-- edited after release: reason -->':\n  " + "\n  ".join(bad))


OLD = "# Changelog\n\n## v0.3.2 — 2026-10-24\n\nPPR lane.\n\n## v0.3.1\n\nFixes.\n"


def test_entry_slipped_into_released_section_is_caught():
    head = OLD.replace("PPR lane.", "PPR lane.\n\nA change that shipped later.")
    assert frozen_violations(head, {"v0.3.2": OLD}) == ["v0.3.2: released section changed after the tag"]


def test_new_untagged_section_is_fine():
    head = OLD.replace("# Changelog\n\n", "# Changelog\n\n## v0.3.3 — 2026-10-26\n\nNew.\n\n")
    assert frozen_violations(head, {"v0.3.2": OLD, "v0.3.1": OLD}) == []


def test_release_date_filled_in_after_tag_is_fine():
    then = OLD.replace("## v0.3.2 — 2026-10-24", "## v0.3.2 — unreleased")
    assert frozen_violations(OLD, {"v0.3.2": then}) == []


def test_marked_edit_is_allowed():
    head = OLD.replace("PPR lane.", "PPR lane (link fixed).\n<!-- edited after release: dead link -->")
    assert frozen_violations(head, {"v0.3.2": OLD}) == []
