"""신기술 레이더 G0 — arXiv 해석·단어 표·급증 등록·EDGAR 연결·읽기 가드. 합성 데이터만, 네트워크 없음."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.radar import arxiv_collect as ac
from core.radar import edgar_link as el
from core.radar import term_monthly as tm

OAI = b"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>
<record><header><identifier>oai:arXiv.org:1501.00001</identifier><datestamp>2020-01-01</datestamp></header>
<metadata><arXiv xmlns="http://arxiv.org/OAI/arXiv/"><id>1501.00001</id><created>2015-01-02</created><updated>2019-05-05</updated>
<title>Deep   Learning
 for X</title><categories>cs.LG stat.ML</categories><abstract>We study deep learning.</abstract></arXiv></metadata></record>
<record><header status="deleted"><identifier>oai:arXiv.org:x</identifier><datestamp>2020-01-01</datestamp></header></record>
<record><header><identifier>oai:arXiv.org:2201.00002</identifier><datestamp>2022-01-03</datestamp></header>
<metadata><arXiv xmlns="http://arxiv.org/OAI/arXiv/"><id>2201.00002</id><created>2022-01-03</created>
<title>Late</title><categories>cs.AI</categories><abstract>a</abstract></arXiv></metadata></record>
<resumptionToken cursor="0" completeListSize="3">tok123</resumptionToken>
</ListRecords></OAI-PMH>"""


def test_parse_uses_v1_created_date_and_token():
    recs, tok, info = ac.parse_list_records(OAI)
    assert tok == "tok123" and info["complete_size"] == 3 and info["n_deleted"] == 1
    assert recs[0] == {"id": "1501.00001", "v1_date": "2015-01-02", "categories": "cs.LG stat.ML", "primary": "cs.LG",
                       "title": "Deep Learning for X", "abstract": "We study deep learning."}
    assert [r["id"] for r in ac.keep_until(recs)] == ["1501.00001"]  # 2022 v1은 버림, 수정일(updated)은 안 봄


def test_parse_last_page_has_no_token_and_error():
    recs, tok, info = ac.parse_list_records(
        b'<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><error code="noRecordsMatch">none</error></OAI-PMH>')
    assert recs == [] and tok is None and info["error"][0] == "noRecordsMatch"


def test_doc_terms_rules():
    t = tm.doc_terms("Convolutional Neural Networks for the Internet of Things",
                     r"In this paper we use solid-state batteries. $x^2$ \alpha perovskite solar cells; 5 GHz.")
    assert {"convolutional neural network", "internet of thing", "solid state battery", "perovskite solar cell"} <= t
    assert "in this paper" not in t and "this paper" not in t  # 첫·끝 불용어
    assert not any("lpha" in x or "alpha" in x for x in t)  # \명령 제거
    assert "network for" not in t
    assert all(2 <= len(x.split()) <= 3 for x in t)
    assert "cell 5" not in t  # 숫자만인 단어는 끊는 자리


def test_singular_rules():
    assert tm.singular("networks") == "network" and tm.singular("batteries") == "battery"
    assert tm.singular("physics") == "physics" and tm.singular("analysis") == "analysis" and tm.singular("series") == "series"
    assert tm.normalize_term("Internet of Things") == "internet of thing"
    assert tm.normalize_term("solid-state battery") == "solid state battery"


def test_sketch_overestimates_only():
    sk = np.zeros((2, 1 << 8), dtype=np.uint32)
    for _ in range(30):
        tm.sketch_add(sk, ["deep learning"], 8)
    tm.sketch_add(sk, [f"t{i} x" for i in range(500)], 8)
    assert tm.sketch_min(sk, "deep learning", 8) >= 30
    assert tm.hashes("deep learning", 26) == tm.hashes("deep learning", 26)  # 실행마다 같은 해시


def test_register_rule():
    T = 60
    total = np.full(T, 1000.0)
    n = np.zeros((3, T))
    n[0, :] = 5  # 꾸준함 → 등록 안 됨
    n[1, 50:] = 20  # 50번째 달부터 새로 나옴: 12개월 합 ≥ 50이 되는 달(50+2=52번째)에 등록
    n[2, :48] = 1
    n[2, 48:] = 4  # 4배 → 5배 미달
    reg = tm.register(n, total)
    assert reg[0] == -1 and reg[2] == -1
    assert reg[1] == 52  # 20×3 = 60 ≥ 50
    # 미래 데이터 방지: t달까지 자른 결과 = 전체 결과의 t달 판정
    for t in (47, 52, 55):
        cut = tm.register(n[:, :t + 1], total[:t + 1])
        assert all((c == r) if r <= t else (c == -1) for c, r in zip(cut, reg))


def test_register_needs_full_history():
    total = np.full(40, 100.0)
    n = np.zeros((1, 40))
    n[0, 30:] = 50
    assert tm.register(n, total)[0] == -1  # 47번째 달 전에는 판정하지 않음


def test_boilerplate():
    r = np.array([[0.02] * 10, [0.02] * 4 + [0.0] * 6])
    assert list(tm.boilerplate(r)) == [True, False]


def test_edgar_query_and_mentions():
    assert el.query("internet of thing") == '"internet of thing" OR "internet of things"'
    assert el.query("solid state battery") == '"solid state battery" OR "solid state batteries"'
    txt = el.html_to_text("<html><style>x{}</style><p>Our Solid-State Batteries and solid state battery cells.</p>&amp;</html>")
    n, ex = el.mentions(txt, "solid state battery")
    assert n == 2 and ex.startswith("Our Solid-State") and "&" in txt
    assert el.mentions("subsolid state battery", "solid state battery")[0] == 0  # 단어 경계


def test_match_ticker_point_in_time():
    iv = el.build_intervals([(1, "FB", "2012-05-18", "2022-06-08"), (1, "META", "2022-06-09", "2026-01-01"),
                             (2, "OLD", "2001-01-01", "2005-12-31")])
    assert el.match_ticker(iv, 1, "2015-02-01") == "FB"
    assert el.match_ticker(iv, 2, "2007-01-01") is None  # 그 시점 구간 밖 → 매칭 실패
    assert el.match_ticker(iv, 9, "2015-01-01") is None and el.match_ticker(iv, None, "2015-01-01") is None


def test_parse_hits():
    j = {"hits": {"total": {"value": 2}, "hits": [
        {"_id": "0001-09-1:d10k.htm", "_source": {"ciks": ["0001108524"], "file_date": "2009-03-09", "form": "10-K",
                                                  "display_names": ["SALESFORCE (CRM)"]}}]}}
    h, total = el.parse_hits(j)
    assert total == 2 and h[0] == {"adsh": "0001-09-1", "filename": "d10k.htm", "cik": 1108524, "file_date": "2009-03-09",
                                   "form": "10-K", "name": "SALESFORCE (CRM)"}


def test_guard_blocks_price_caches():
    from data.radar import guard

    assert not guard.allowed(guard.CACHE / "sp500" / "prices" / "AAPL.csv")
    assert not guard.allowed(guard.CACHE / "QQQ.parquet")
    assert not guard.allowed(guard.CACHE / "rs26" / "QQQ.csv")
    assert not guard.allowed(guard.ROOT / "data" / "state.db")
    assert guard.allowed(guard.CACHE / "arxiv" / "parsed" / "page_00001.parquet")
    assert guard.allowed(guard.CACHE / "sp500" / "identity.csv")
    assert guard.allowed(guard.ROOT / "docs" / "research" / "x.md")


def test_radar_modules_never_name_price_caches():
    """레이더 코드가 가격·수익 캐시 경로를 직접 쓰지 않는다(가드와 별도의 정적 점검)."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    for f in list((root / "core" / "radar").glob("*.py")) + list((root / "data" / "radar").glob("*.py")):
        s = f.read_text(encoding="utf-8")
        for bad in ("prices", "tiingo", "backtest", "rs26", "value_score", "QQQ"):
            if f.name == "guard.py":
                continue
            assert f'"{bad}"' not in s and f"/{bad}" not in s, (f.name, bad)


# ── G0-보완 (core/radar/g0b.py) ─────────────────────────────────────────────

from core.radar import g0b  # noqa: E402


def test_g0b_filters():
    assert g0b.is_industrial("cs.LG") and g0b.is_industrial("eess.SP") and g0b.is_industrial("quant-ph")
    assert g0b.is_industrial("cond-mat.mtrl-sci") and not g0b.is_industrial("cond-mat.str-el") and not g0b.is_industrial("hep-th")
    assert g0b.generic_edge("cnn based") and g0b.generic_edge("model achieve") and g0b.generic_edge("large language model")
    assert not g0b.generic_edge("deep learning") and not g0b.generic_edge("modeling tool")
    assert g0b.event_term("novel coronavirus") and g0b.event_term("respiratory syndrome coronavirus") and g0b.event_term("second wave")
    assert not g0b.event_term("sarsa learning") and not g0b.event_term("wave background")
    assert g0b.contains("convolutional neural network", "convolutional neural") and g0b.contains("deep neural network", "neural network")
    assert not g0b.contains("deep neural network", "deep network")


def test_register_growth_rule():
    T = 72
    total = np.full(T, 10_000.0)
    n = np.zeros((3, T))
    n[0] = np.round(30 * 1.4 ** (np.arange(T) / 12))  # 연 40% 성장 → 12개월 합 ≥ 200이 되면 등록
    n[1] = np.round(30 * 1.2 ** (np.arange(T) / 12))  # 연 20% → 미등록
    n[2, 40:] = 50  # 3년 전 0 → 꾸준한 성장 아님(급증 규칙 몫)
    reg = g0b.register_growth(n, total)
    assert reg[0] >= 47 and n[0, reg[0] - 11:reg[0] + 1].sum() >= 200 and reg[1] == -1 and reg[2] == -1
    for t in (50, reg[0], 70):  # 미래 데이터 방지
        cut = g0b.register_growth(n[:, :t + 1], total[:t + 1])
        assert all((c == r) if r <= t else (c == -1) for c, r in zip(cut, reg))


def test_section_business_skips_toc_and_risk_factors():
    body = "Item 1. Business " + "We design lidar sensors. lidar sensors " + "x " * 1200
    text = ("TABLE OF CONTENTS Item 1. Business 3 Item 1A. Risk Factors 9 Item 2. Properties 20 " + body +
            " Item 1A. Risk Factors " + "lidar sensors risk " * 200 + " Item 2. Properties ...")
    s = g0b.section(text, "10-K")
    assert s is not None and s.startswith("Item 1. Business We design") and "Risk Factors" not in s
    assert el.mentions(s, "lidar sensor")[0] == 2
    q = "Item 2. Management’s Discussion and Analysis " + "edge computing " * 3 + "y " * 1200 + " Item 3. Quantitative"
    assert el.mentions(g0b.section(q, "10-Q"), "edge computing")[0] == 3
    assert g0b.section("no items here", "10-K") is None and g0b.section(text, "8-K") is None
