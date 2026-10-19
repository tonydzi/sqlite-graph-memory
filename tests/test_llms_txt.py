# -*- coding: utf-8 -*-
"""llms.txt copies bench numbers from bench/LEADERBOARD.md; they must not drift apart.

LEADERBOARD.md is generated from bench/results/*.json, so it is the source. When a
new result moves a number, this test fails until llms.txt is updated to match.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# leaderboard row label -> how llms.txt names that system
SYSTEMS = {
    "sqlite-graph-memory-vector": "sqlite-graph-memory, vector only",
    "sqlite-graph-memory": "sqlite-graph-memory, vector+graph",
    "bm25": "bm25 reference",
}


def _leaderboard_recall():
    rows = {}
    for line in (ROOT / "bench" / "LEADERBOARD.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*\[([^\]]+)\]\([^)]*\)\s*`[^`]*`\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", line)
        if m and m.group(1) in SYSTEMS and m.group(1) not in rows:
            rows[m.group(1)] = (m.group(2), m.group(3))
    return rows


def test_leaderboard_has_every_system_llms_txt_quotes():
    assert set(_leaderboard_recall()) == set(SYSTEMS)


def test_llms_txt_recall_matches_leaderboard():
    text = (ROOT / "llms.txt").read_text(encoding="utf-8")
    for system, (r5, r10) in _leaderboard_recall().items():
        lines = [l for l in text.splitlines() if l.startswith("- " + SYSTEMS[system])]
        assert len(lines) == 1, "llms.txt has no single line for %s" % system
        assert "R@5 %s, R@10 %s" % (r5, r10) in lines[0], (
            "llms.txt says %r, LEADERBOARD.md says R@5 %s, R@10 %s" % (lines[0], r5, r10))
