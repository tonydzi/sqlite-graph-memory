# -*- coding: utf-8 -*-
"""sgm-stats: each health number crosses its threshold on a synthetic vault.

Every vault here is a handful of files in tmp_path, built so that exactly one metric
moves. No model download, no network.
"""
import datetime
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sqlite_graph_memory import vault_stats  # noqa: E402

TODAY = datetime.date(2026, 10, 20)


def _write(root, name, links=(), fm=""):
    body = " ".join(f"[[{t}]]" for t in links)
    head = f"---\n{fm}\n---\n" if fm else ""
    (root / f"{name}.md").write_text(f"{head}# {name}\n\n{body}\n", encoding="utf-8")


def _ring(root, n, extra=0):
    """n pages, each linking to the next `1 + extra` pages around a ring: no orphans,
    one component, degree = 1 + extra."""
    for i in range(n):
        _write(root, f"p{i}", [f"p{(i + k) % n}" for k in range(1, 2 + extra)])


def _metric(report, name):
    return next(m for m in report["metrics"] if m["name"] == name)


def _run(root, **kw):
    return vault_stats.compute(root, today=TODAY, th=kw or None)


# ------------------------------------------------------------------ healthy baseline

def test_healthy_vault_is_all_ok_and_exits_zero(tmp_path, capsys):
    _ring(tmp_path, 20, extra=3)            # degree 4, no orphans, one component
    assert vault_stats.main([str(tmp_path), "--today", "2026-10-20"]) == 0
    out = capsys.readouterr().out
    assert "verdict: ok" in out
    for name in ("orphan_rate", "avg_degree", "main_component", "stale_concepts"):
        assert name in out


# ------------------------------------------------------------------ orphan rate

@pytest.mark.parametrize("orphans,expected", [(0, "ok"), (2, "warn"), (5, "fail")])
def test_orphan_rate_thresholds(tmp_path, orphans, expected):
    # 20-page ring (degree 4) plus `orphans` pages that link in but are never linked to.
    _ring(tmp_path, 20 - orphans, extra=3)
    for i in range(orphans):
        _write(tmp_path, f"o{i}", ["p0", "p1", "p2", "p3"])
    m = _metric(_run(tmp_path), "orphan_rate")
    assert m["value"] == pytest.approx(100.0 * orphans / 20, abs=0.1)
    assert m["verdict"] == expected


def test_inbound_links_are_what_make_a_page_not_an_orphan(tmp_path):
    # hub links out to everyone; nobody links to hub -> exactly one orphan.
    _write(tmp_path, "hub", ["a", "b", "c"])
    for n in "abc":
        _write(tmp_path, n, [x for x in "abc" if x != n])
    m = _metric(_run(tmp_path), "orphan_rate")
    assert m["examples"] == ["hub"]


def test_dangling_and_self_links_are_not_edges(tmp_path):
    _write(tmp_path, "a", ["a", "nowhere", "b"])
    _write(tmp_path, "b", ["a"])
    r = _run(tmp_path)
    assert _metric(r, "avg_degree")["detail"].startswith("2 resolved links")


# ------------------------------------------------------------------ average degree

@pytest.mark.parametrize("extra,expected", [
    (0, "fail"),    # degree 1   (< 2)
    (1, "warn"),    # degree 2   (2-3)
    (3, "ok"),      # degree 4   (3-8)
    (8, "warn"),    # degree 9   (8-10)
    (11, "fail"),   # degree 12  (> 10)
])
def test_degree_thresholds(tmp_path, extra, expected):
    _ring(tmp_path, 20, extra=extra)
    m = _metric(_run(tmp_path), "avg_degree")
    assert m["value"] == pytest.approx(1 + extra)
    assert m["verdict"] == expected


# ------------------------------------------------------------------ main component

def test_fragmented_vault_warns_on_main_component(tmp_path):
    # two separate 10-page rings: main component holds 50%.
    for side in "xy":
        for i in range(10):
            _write(tmp_path, f"{side}{i}", [f"{side}{(i + k) % 10}" for k in range(1, 5)])
    m = _metric(_run(tmp_path), "main_component")
    assert m["value"] == 50.0 and m["verdict"] == "warn"
    assert "2 components" in m["detail"]


# ------------------------------------------------------------------ stale concepts

@pytest.mark.parametrize("stale,expected", [(2, "ok"), (3, "warn")])
def test_stale_concepts_threshold(tmp_path, stale, expected):
    _ring(tmp_path, 20, extra=3)
    for i in range(4):
        updated = "2026-01-01" if i < stale else "2026-10-01"
        _write(tmp_path, f"c{i}", ["p0", "p1", "p2", "p3"],
               fm=f"type: concept\nupdated: {updated}")
    m = _metric(_run(tmp_path), "stale_concepts")
    assert m["value"] == pytest.approx(100.0 * stale / 4)
    assert m["verdict"] == expected


def test_today_flag_moves_the_staleness_line(tmp_path):
    (tmp_path / "concepts").mkdir()
    _write(tmp_path / "concepts", "idea", fm="updated: 2026-06-01")
    early = vault_stats.compute(tmp_path, today=datetime.date(2026, 7, 1))
    late = vault_stats.compute(tmp_path, today=datetime.date(2026, 10, 1))
    assert _metric(early, "stale_concepts")["value"] == 0.0
    assert _metric(late, "stale_concepts")["value"] == 100.0


def test_no_concepts_is_not_a_failure(tmp_path):
    _ring(tmp_path, 20, extra=3)
    m = _metric(_run(tmp_path), "stale_concepts")
    assert m["verdict"] == "n/a" and m["value"] is None


# ------------------------------------------------------------------ CLI contract

def test_exit_code_follows_worst_verdict(tmp_path):
    _ring(tmp_path, 20, extra=1)            # degree 2 -> warn, rest ok
    assert vault_stats.main([str(tmp_path), "--json", "--today", "2026-10-20"]) == 1
    _ring(tmp_path, 20, extra=0)            # rewrite: degree 1 -> fail
    assert vault_stats.main([str(tmp_path), "--json", "--today", "2026-10-20"]) == 2


def test_threshold_flags_override_defaults(tmp_path, capsys):
    _ring(tmp_path, 20, extra=1)            # degree 2: warn by default
    code = vault_stats.main([str(tmp_path), "--json", "--today", "2026-10-20",
                             "--threshold-degree-ok-min", "2"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0 and _metric(report, "avg_degree")["verdict"] == "ok"


def test_skips_what_the_indexer_skips(tmp_path):
    _ring(tmp_path, 20, extra=3)
    (tmp_path / ".stversions").mkdir()
    _write(tmp_path / ".stversions", "old")                 # would be an orphan
    _write(tmp_path, "p0.sync-conflict-20261001-ABC")       # would be an orphan
    r = _run(tmp_path)
    assert r["pages"] == 20 and _metric(r, "orphan_rate")["value"] == 0.0


def test_bench_vault_numbers(tmp_path):
    """The shipped synthetic vault, measured. If the vault or the parser changes,
    this changes with it, on purpose."""
    vault = Path(__file__).resolve().parents[1] / "bench" / "vault"
    if not vault.is_dir():
        pytest.skip("bench/vault not in this checkout")
    r = _run(vault)
    assert r["pages"] == 202
    assert _metric(r, "orphan_rate")["value"] == 26.2
    assert _metric(r, "main_component")["value"] == 100.0
