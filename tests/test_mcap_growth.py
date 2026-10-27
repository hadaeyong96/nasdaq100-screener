"""실적 동반 시총 성장 전략 순수 함수 (core/mcap_growth.py) — 합성 데이터, 네트워크 없음."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import mcap_growth as mg


def _sh(rows):
    return pd.DataFrame(rows, columns=["tag", "start", "end", "filed", "val", "accn"])


def test_shares_dei_sums_classes_of_latest_filing_before_x():
    sh = _sh([
        (mg.SHARES_DEI, None, "2015-01-20", "2015-02-10", 100.0, "a1"),
        (mg.SHARES_DEI, None, "2015-04-20", "2015-05-01", 60.0, "a2"),  # 같은 접수 두 종류 주식
        (mg.SHARES_DEI, None, "2015-04-20", "2015-05-01", 50.0, "a2"),
        (mg.SHARES_DEI, None, "2015-07-20", "2015-08-01", 999.0, "a3"),  # 판단일 뒤 접수 — 쓰면 안 됨
    ])
    assert mg.shares_asof(sh, "2015-07-01") == 110.0
    assert mg.shares_asof(sh, "2015-05-01") == 100.0  # 접수 당일은 아직 모름(filed < x)
    assert mg.shares_asof(sh, "2015-02-10") is None


def test_shares_split_adjust_and_fallback_and_staleness():
    splits = pd.Series([2.0], index=pd.DatetimeIndex(["2015-06-01"]))
    sh = _sh([(mg.SHARES_DEI, None, "2015-04-20", "2015-05-01", 100.0, "a")])
    assert mg.shares_asof(sh, "2015-07-01", splits) == 200.0  # 기준일 뒤 2:1 분할
    assert mg.shares_asof(sh, "2015-05-20", splits) == 100.0
    assert mg.shares_asof(sh, "2016-06-01") is None  # 400일 넘음
    wa = _sh([(mg.SHARES_WA, "2015-01-01", "2015-03-31", "2015-05-01", 80.0, "b"),
              (mg.SHARES_WA, "2014-10-01", "2015-03-31", "2015-05-01", 79.0, "b")])
    assert mg.shares_asof(wa, "2015-06-01") == 80.0  # 3개월 기간 우선


def _rev(rows):
    df = pd.DataFrame(rows, columns=["start", "end", "filed", "val"])
    df["tag"] = "Revenues"
    return df


def test_revenue_growth_ytd_method_and_annual():
    rv = _rev([
        ("2013-01-01", "2013-12-31", "2014-02-20", 1000.0),
        ("2014-01-01", "2014-12-31", "2015-02-20", 1100.0),
        ("2013-01-01", "2013-06-30", "2013-08-01", 480.0),
        ("2014-01-01", "2014-06-30", "2014-08-01", 520.0),
        ("2015-01-01", "2015-06-30", "2015-08-01", 600.0),
        ("2012-01-01", "2012-12-31", "2013-02-20", 900.0),
    ])
    # 2015-09-01: TTM(2015-06-30) = 1100 + 600 − 520 = 1180, TTM(2014-06-30) = 1000 + 520 − 480 = 1040
    assert mg.revenue_growth(rv, "2015-09-01") == pytest.approx(1180 / 1040 - 1)
    # 2015-03-02: 최근 = 연간 2014 → 1100 / 1000
    assert mg.revenue_growth(rv, "2015-03-02") == pytest.approx(0.1)
    # 2015-07-15: 2015 상반기는 아직 모름 → 연간 비교(기간 끝 196일 전), 2015-08-01 당일은 상반기도 모르고 연간은 200일 넘어 없음
    assert mg.revenue_growth(rv, "2015-07-15") == pytest.approx(0.1)
    assert mg.revenue_growth(rv, "2015-08-01") is None
    assert mg.revenue_growth(rv, "2016-06-01") is None  # 최근 기간 끝이 200일 넘게 오래됨


def test_revenue_tag_priority_and_restatement():
    rv = _rev([("2013-01-01", "2013-12-31", "2014-02-20", 1000.0), ("2014-01-01", "2014-12-31", "2015-02-20", 1200.0),
               ("2013-01-01", "2013-12-31", "2015-02-20", 1100.0)])  # 2013 값 다시 보고(나중 접수본)
    assert mg.revenue_growth(rv, "2015-03-02") == pytest.approx(1200 / 1100 - 1)
    assert mg.revenue_growth(rv, "2015-02-01") is None  # 2014 연간 모름, 2013 연간 비교 상대(2012) 없음
    lower = rv.copy()
    lower.loc[0, "tag"] = "SalesRevenueNet"
    lower = pd.concat([lower, pd.DataFrame([{"start": "2013-01-01", "end": "2013-12-31", "filed": "2014-02-21", "val": 5.0,
                                             "tag": "SalesRevenueNet"}])])
    assert mg.revenue_growth(lower, "2015-03-02") == pytest.approx(1200 / 1100 - 1)  # 우선순위 높은 Revenues 값


def test_score_and_passes():
    assert mg.score(120.0, 100.0) == pytest.approx(0.2) and mg.score(None, 1.0) is None and mg.score(1.0, 0.0) is None
    assert mg.passes(0.2, 0.1, 0.05) and not mg.passes(0.2, 0.1, 0.051) and not mg.passes(0.1, 0.1, 0.0)
    assert not mg.passes(None, 0.1, 0.0) and mg.passes(None, None, 0.0, use_rev=False) and not mg.passes(0.5, 0.1, None)
    assert mg.sector_medians({"A": 0.1, "B": 0.3, "C": None, "D": 0.5}, {"A": "X", "B": "X", "C": "X", "D": "Y"}) == {"X": 0.2, "Y": 0.5}


def test_select_sector_champions_cap_and_fill():
    p = mg.Params(n=5, sector_cap=2, keep_rank=3)
    scores = {"A1": 0.9, "A2": 0.8, "A3": 0.7, "A4": 0.6, "B1": 0.1, "C1": 0.05, "D1": 0.5}
    sector = {"A1": "A", "A2": "A", "A3": "A", "A4": "A", "B1": "B", "C1": "C", "D1": "D"}
    ok = {t: True for t in scores} | {"D1": False}
    out = mg.select(scores, ok, sector, [], p)
    assert out == ["A1", "B1", "C1", "A2"]  # 섹터 1등(A1·B1·C1) → 점수 순, A는 2개까지, D1은 조건 탈락 → 4개뿐
    p2 = mg.Params(n=3, sector_cap=3, keep_rank=3)
    assert mg.select(scores, ok, sector, [], p2) == ["A1", "B1", "C1"]  # 빈자리보다 섹터 1등이 많으면 점수 큰 섹터부터


def test_select_buffer_keeps_holdings_in_top_rank():
    p = mg.Params(n=2, sector_cap=3, keep_rank=3)
    scores = {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6}
    sector = {t: "S" for t in scores}
    ok = {t: True for t in scores}
    assert mg.select(scores, ok, sector, ["C"], p) == ["C", "A"]  # C는 순위 3 → 유지
    assert mg.select(scores, ok, sector, ["D"], p) == ["A", "B"]  # D는 순위 4 → 탈락
    assert mg.select(scores, ok | {"C": False}, sector, ["C"], p) == ["A", "B"]  # 조건 탈락이면 유지 안 함


def test_select_random_respects_cap_and_seed():
    pool = [f"T{i}" for i in range(30)]
    sector = {t: f"S{i % 4}" for i, t in enumerate(pool)}
    a = mg.select_random(pool, sector, np.random.default_rng(1))
    b = mg.select_random(pool, sector, np.random.default_rng(1))
    assert a == b and len(a) == 12  # 섹터 4개 × 최대 3
    assert max(pd.Series([sector[t] for t in a]).value_counts()) <= 3


def test_mcap_ranks():
    assert mg.mcap_ranks({"A": 1.0, "B": 3.0, "C": 3.0}) == {"B": 1, "C": 2, "A": 3}
