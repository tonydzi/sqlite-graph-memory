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


def test_link_in_inline_code_is_not_an_edge(tmp_path):
    (tmp_path / "a.md").write_text("Write `[[x]]` to link.\n", encoding="utf-8")
    (tmp_path / "x.md").write_text("# x\n", encoding="utf-8")
    pages, edges = vault_stats.build_graph(tmp_path)
    assert pages.keys() == {"a", "x"}
    assert edges == set()


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


# ------------------------------------------------------------------ review regressions (PR #50)

def test_same_name_in_two_folders_is_two_pages(tmp_path):
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    _write(tmp_path / "a", "x", ["y"])
    _write(tmp_path / "b", "x", ["y"])
    _write(tmp_path, "y", ["x"])            # reaches both x pages
    r = _run(tmp_path)
    assert r["pages"] == 3
    assert _metric(r, "orphan_rate")["value"] == 0.0


def test_concepts_folder_above_the_vault_does_not_count(tmp_path):
    vault = tmp_path / "concepts" / "vault"
    vault.mkdir(parents=True)
    _write(vault, "note", fm="date: 2020-01-01")
    m = _metric(_run(vault), "stale_concepts")
    assert m["verdict"] == "n/a" and m["detail"] == "no concept pages"


def test_stale_share_is_over_all_concepts_and_names_undated(tmp_path):
    _write(tmp_path, "c0", fm="type: concept\nupdated: 2020-01-01")
    for i in range(1, 10):
        _write(tmp_path, f"c{i}", fm="type: concept")
    m = _metric(_run(tmp_path), "stale_concepts")
    assert m["value"] == 10.0 and m["verdict"] == "ok"
    assert "9 undated" in m["detail"]


def test_printed_value_and_verdict_agree_at_the_boundary(tmp_path):
    # 5 orphans of 101 pages = 4.95%, printed as 5.0: must be warn, like 5.0 is.
    for i in range(96):
        _write(tmp_path, f"p{i}", [f"p{(i + k) % 96}" for k in range(1, 5)])
    for i in range(5):
        _write(tmp_path, f"o{i}", ["p0", "p1", "p2", "p3"])
    m = _metric(_run(tmp_path), "orphan_rate")
    assert m["value"] == 5.0 and m["verdict"] == "warn"


def test_impossible_date_is_undated_not_a_crash(tmp_path):
    _write(tmp_path, "c", fm="type: concept\nupdated: 2026-02-30")
    m = _metric(_run(tmp_path), "stale_concepts")
    assert m["value"] == 0.0 and "1 undated" in m["detail"]


def test_empty_vault_exits_with_fail_code(tmp_path, capsys):
    assert vault_stats.main([str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert "no notes" in captured.err
    assert captured.out == ""


def test_empty_vault_json_output(tmp_path, capsys):
    code = vault_stats.main([str(tmp_path), "--json"])
    assert code == 2
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["verdict"] == "fail"
    assert report["exit_code"] == 2
    assert report["pages"] == 0
    assert report["notes_dir"] == str(tmp_path)
    assert "no notes" in report["error"]


def test_single_quoted_frontmatter(tmp_path):
    _write(tmp_path, "c", fm="type: 'concept'\nupdated: '2020-01-01'")
    m = _metric(_run(tmp_path), "stale_concepts")
    assert m["value"] == 100.0


def test_folder_and_extension_wikilinks_resolve(tmp_path):
    """Issue #71: [[Sub/c]] and [[c.md]] resolve to existing notes."""
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "a.md").write_text("[[Sub/c]] [[c.md]]\n", encoding="utf-8")
    (sub / "c.md").write_text("# c\n", encoding="utf-8")
    pages, edges = vault_stats.build_graph(tmp_path)
    assert set(pages.keys()) == {"a", "sub/c"}
    assert sorted(edges) == [("a", "sub/c")]


def test_folder_link_disambiguates_same_stem_in_different_folders(tmp_path):
    """A folder-prefixed link [[sub/x]] resolves to sub/x rather than every x in the vault."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "a.md").write_text("[[Sub/x.md]]\n", encoding="utf-8")
    (tmp_path / "sub" / "x.md").write_text("# sub x\n", encoding="utf-8")
    (tmp_path / "other" / "x.md").write_text("# other x\n", encoding="utf-8")
    _, edges = vault_stats.build_graph(tmp_path)
    assert sorted(edges) == [("a", "sub/x")]


def test_folder_link_prefers_exact_vault_relative_path_over_suffix_match(tmp_path):
    """When both sub/c.md and other/sub/c.md exist, [[sub/c]] resolves only to sub/c."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "other" / "sub").mkdir(parents=True)
    (tmp_path / "a.md").write_text("[[sub/c]]\n", encoding="utf-8")
    (tmp_path / "sub" / "c.md").write_text("# exact sub/c\n", encoding="utf-8")
    (tmp_path / "other" / "sub" / "c.md").write_text("# nested other/sub/c\n", encoding="utf-8")
    _, edges = vault_stats.build_graph(tmp_path)
    assert sorted(edges) == [("a", "sub/c")]


# ------------------------------------------------------------------ breaker panel regressions (2026-10-26)

@pytest.mark.parametrize("fm", [
    "type: [concept]",
    "type: [note, concept]",
    "type:\n  - concept",
    "type:\n  - note\n  - concept",
    "type:\n- concept",
    "type: 'concept'  # evergreen",
    "type: concept # evergreen",
    "type: Concept",
])
def test_type_concept_in_any_yaml_form(tmp_path, fm):
    _write(tmp_path, "c", fm=fm + "\nupdated: 2020-01-01")
    m = _metric(_run(tmp_path), "stale_concepts")
    assert m["value"] == 100.0, fm


def test_type_block_list_does_not_swallow_the_next_key(tmp_path):
    # `type:` empty and the next line is another key, not a list item.
    _write(tmp_path, "c", fm="type:\nupdated: 2020-01-01")
    assert _metric(_run(tmp_path), "stale_concepts")["verdict"] == "n/a"


def test_frontmatter_only_note_without_final_newline(tmp_path):
    (tmp_path / "c.md").write_text("---\ntype: concept\nupdated: 2020-01-01\n---",
                                   encoding="utf-8")
    assert _metric(_run(tmp_path), "stale_concepts")["value"] == 100.0


@pytest.mark.parametrize("flag,val", [
    ("--threshold-stale-days", "nan"),
    ("--threshold-stale-days", "inf"),
    ("--threshold-stale-days", "1e9"),
    ("--threshold-stale-days", "-1"),
    ("--threshold-orphan-warn", "nan"),
])
def test_bad_threshold_is_rejected_with_fail_code(tmp_path, flag, val):
    _ring(tmp_path, 20, extra=3)
    with pytest.raises(SystemExit) as e:
        vault_stats.main([str(tmp_path), flag, val])
    assert e.value.code == 2


def test_early_today_does_not_crash(tmp_path):
    _write(tmp_path, "c", fm="type: concept\nupdated: 0001-01-01")
    code = vault_stats.main([str(tmp_path), "--json", "--today", "0001-01-02"])
    assert code in (0, 1, 2)


def test_crash_exits_fail_not_warn(tmp_path, monkeypatch, capsys):
    _ring(tmp_path, 20, extra=3)
    def boom(*a, **k):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(vault_stats, "compute", boom)
    assert vault_stats.main([str(tmp_path)]) == 2
    assert "disk on fire" in capsys.readouterr().err


def test_unreadable_notes_are_counted_and_warn(tmp_path, monkeypatch):
    _ring(tmp_path, 20, extra=3)
    real = vault_stats.read_for_links
    monkeypatch.setattr(vault_stats, "read_for_links",
                        lambda p: None if Path(p).stem in ("p3", "p7") else real(p))
    r = _run(tmp_path)
    m = _metric(r, "unreadable")
    assert m["value"] == 2 and m["examples"] == ["p3", "p7"]
    assert r["exit_code"] >= 1


def test_duplicate_names_do_not_inflate_degree(tmp_path):
    # 30 README.md in 30 folders; 20 notes each link [[README]] once plus 3 ring links.
    for i in range(30):
        (tmp_path / f"d{i}").mkdir()
        _write(tmp_path / f"d{i}", "README", ["p0"])
    for i in range(20):
        _write(tmp_path, f"p{i}", ["README"] + [f"p{(i + k) % 20}" for k in range(1, 4)])
    r = _run(tmp_path)
    # 20 pages x 4 link names + 30 READMEs x 1 = 110 links over 50 pages
    assert _metric(r, "avg_degree")["value"] == 2.2
    assert _metric(r, "orphan_rate")["value"] == 0.0        # every README is reached


def test_each_note_is_read_once(tmp_path, monkeypatch):
    _ring(tmp_path, 20, extra=3)
    _write(tmp_path, "c", ["p0"], fm="type: concept\nupdated: 2026-10-01")
    calls = []
    real = vault_stats.read_for_links
    monkeypatch.setattr(vault_stats, "read_for_links", lambda p: calls.append(p) or real(p))
    _run(tmp_path)
    assert len(calls) == 21 and len(set(calls)) == 21


def test_hash_inside_quotes_is_not_a_comment(tmp_path):
    _write(tmp_path, "c", fm='type: "concept # draft"\nupdated: 2020-01-01')
    assert _metric(_run(tmp_path), "stale_concepts")["verdict"] == "n/a"


def test_unreadable_notes_are_named_in_text_output(tmp_path, monkeypatch, capsys):
    _ring(tmp_path, 20, extra=3)
    real = vault_stats.read_for_links
    monkeypatch.setattr(vault_stats, "read_for_links",
                        lambda p: None if Path(p).stem == "p3" else real(p))
    vault_stats.main([str(tmp_path)])
    assert "e.g. p3" in capsys.readouterr().out
