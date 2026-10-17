"""core/value_score — 가치평가 점수 70점 (docs/prereg/value_score.md). 합성 데이터만, 네트워크 없음."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import value_score as vs


def _fy_facts(years: list[int], vals: dict[str, list[float]], filed_lag_days: int = 60, end_md: str = "12-31") -> pd.DataFrame:
    """회계연도마다 한 번 10-K(그해 끝 + filed_lag_days 접수)로 그해 값을 내는 합성 사실 표."""
    rows = []
    for i, y in enumerate(years):
        end = f"{y}-{end_md}"
        start = (pd.Timestamp(end) - pd.Timedelta(days=364)).date().isoformat()
        filed = (pd.Timestamp(end) + pd.Timedelta(days=filed_lag_days)).date().isoformat()
        for key, arr in vals.items():
            tag = vs.TAGS[key][0]
            unit = vs.UNITS.get(key, "USD")
            flow = key in vs.FLOW_KEYS
            rows.append({"tag": tag, "start": start if flow else None, "end": end, "val": arr[i], "filed": filed,
                         "form": "10-K", "unit": unit})
    return pd.DataFrame(rows)


BASE = {
    "rev": [100, 115, 132, 152, 175], "ni": [10, 12, 14, 16, 19], "eps": [1.0, 1.2, 1.4, 1.6, 1.9],
    "opinc": [15, 17, 20, 23, 26], "ocf": [14, 16, 19, 22, 25], "capex": [3, 3, 4, 4, 5], "gp": [50, 57, 66, 76, 88],
    "da": [5, 5, 6, 6, 7], "shares": [10, 10, 10, 10, 10], "equity": [60, 65, 70, 75, 80], "cash": [10, 10, 10, 10, 10],
    "assets": [150, 160, 170, 180, 190], "ltd_total": [20, 20, 20, 20, 20],
}
YEARS = [2010, 2011, 2012, 2013, 2014]


def test_annual_table_uses_only_filed_before_asof():
    f = _fy_facts(YEARS, BASE)
    t1 = vs.annual_table(f, pd.Timestamp("2015-02-28"))  # 2014 10-K 접수(2015-03-01) 전
    t2 = vs.annual_table(f, pd.Timestamp("2015-03-02"))  # 접수 다음 날부터
    assert t1.index[0] == "2013-12-31" and len(t1) == 4
    assert t2.index[0] == "2014-12-31" and len(t2) == 5
    # 접수 당일은 아직 쓰지 않는다 (접수 다음 거래일부터)
    assert vs.annual_table(f, pd.Timestamp("2015-03-01")).index[0] == "2013-12-31"


def test_amendment_by_filed_date():
    f = _fy_facts(YEARS, BASE)
    amend = f[(f["tag"] == "NetIncomeLoss") & (f["end"] == "2014-12-31")].copy()
    amend["val"], amend["filed"], amend["form"] = 99.0, "2015-06-01", "10-K/A"
    f2 = pd.concat([f, amend], ignore_index=True)
    assert vs.annual_table(f2, pd.Timestamp("2015-05-29")).loc["2014-12-31", "ni"] == 19
    assert vs.annual_table(f2, pd.Timestamp("2015-06-02")).loc["2014-12-31", "ni"] == 99


def test_quarter_facts_ignored_and_chain_breaks_on_gap():
    f = _fy_facts(YEARS, BASE)
    q = f[f["tag"] == "NetIncomeLoss"].head(1).copy()
    q["start"], q["end"], q["val"] = "2014-10-01", "2014-12-31", 5.0  # 분기 기간 → 연간 아님
    t = vs.annual_table(pd.concat([f, q], ignore_index=True), pd.Timestamp("2016-01-01"))
    assert t.loc["2014-12-31", "ni"] == 19
    g = _fy_facts([2008, 2011, 2012, 2013, 2014], BASE)  # 2008 → 2011 사이 공백: 체인이 2011에서 끊김
    assert list(vs.annual_table(g, pd.Timestamp("2016-01-01")).index) == ["2014-12-31", "2013-12-31", "2012-12-31", "2011-12-31"]


def test_metrics_points_base_case():
    t = vs.annual_table(_fy_facts(YEARS, BASE), pd.Timestamp("2015-03-02"))
    m = vs.metrics(t, price=19.0, splits=None, asof=pd.Timestamp("2015-03-02"))
    g = (1.9 / 1.2) ** (1 / 3) - 1  # EPS 3년 성장 ≈ 16.5%
    peg = (19.0 / 1.9) / (g * 100)  # PER 10 → PEG ≈ 0.61
    assert m.peg == pytest.approx(peg)
    assert m.pts["peg"] == 20.0
    assert m.pts["rev_growth"] == 10.0  # (175/115)^(1/3)-1 ≈ 15%
    assert m.pts["eps_up"] == 5.0  # 4번 모두 증가
    assert m.pts["ocf_ni"] == 5.0  # (25+22+19)/(19+16+14) > 1
    assert m.pts["fcf"] == 5.0  # FCF 모두 + , 20/175 ≥ 10%
    assert m.pts["shares"] == 5.0
    # ROIC = 영업이익×0.79 ÷ (자본+부채−현금), 3년 평균
    r = np.mean([26 * 0.79 / 90, 23 * 0.79 / 85, 20 * 0.79 / 80])
    assert m.pts["roic"] == (10.0 if r >= 0.15 else 5.0)
    # 순부채/EBITDA = (20−10)/(26+7) < 1
    assert m.pts["net_debt"] == 5.0
    assert m.raw["gpa"] == pytest.approx(88 / 190)
    assert m.raw["fcf_yield"] == pytest.approx(20 / (19.0 * 10))
    assert m.profitable and m.years == 5


@pytest.mark.parametrize("price,expected", [(19.0 * 2.0, 13.3), (19.0 * 3.0, 6.7), (19.0 * 4.0, 0.0)])
def test_peg_bands(price, expected):
    t = vs.annual_table(_fy_facts(YEARS, BASE), pd.Timestamp("2015-03-02"))
    m = vs.metrics(t, price=price, splits=None, asof=pd.Timestamp("2015-03-02"))
    lo = {13.3: 1.0, 6.7: 1.5, 0.0: 2.0}[expected]
    assert m.peg > lo
    assert m.pts["peg"] == expected


def test_peg_growth_capped_at_50pct():
    v = dict(BASE, eps=[0.1, 0.2, 0.5, 1.0, 2.0])  # 3년 성장 (2/0.2)^(1/3)−1 ≈ 115% → 50%로 자름
    t = vs.annual_table(_fy_facts(YEARS, v), pd.Timestamp("2015-03-02"))
    m = vs.metrics(t, price=60.0, splits=None, asof=pd.Timestamp("2015-03-02"))
    assert m.peg == pytest.approx((60.0 / 2.0) / 50.0)


def test_peg_zero_and_na_cases():
    asof = pd.Timestamp("2015-03-02")
    neg = dict(BASE, eps=[1.0, 1.2, 1.4, 1.6, -0.5])
    assert vs.metrics(vs.annual_table(_fy_facts(YEARS, neg), asof), 20.0, None, asof).pts["peg"] == 0.0  # EPS ≤ 0 → 0점
    down = dict(BASE, eps=[1.0, 2.0, 1.4, 1.6, 1.5])
    assert vs.metrics(vs.annual_table(_fy_facts(YEARS, down), asof), 20.0, None, asof).pts["peg"] == 0.0  # 성장률 ≤ 0 → 0점
    base_neg = dict(BASE, eps=[1.0, -0.2, 1.4, 1.6, 1.9])  # 3년 전 EPS ≤ 0 → N/A (구현 해석)
    assert vs.metrics(vs.annual_table(_fy_facts(YEARS, base_neg), asof), 20.0, None, asof).pts["peg"] is None
    three = {k: v[2:] for k, v in BASE.items()}  # 연간 3년뿐 → 3년 전 EPS 없음 → N/A
    m = vs.metrics(vs.annual_table(_fy_facts(YEARS[2:], three), asof), 20.0, None, asof)
    assert m.pts["peg"] is None and m.years == 3 and m.pts["eps_up"] is None


def test_eps_up_with_four_years():
    asof = pd.Timestamp("2015-03-02")
    four = {k: v[1:] for k, v in BASE.items()}
    m = vs.metrics(vs.annual_table(_fy_facts(YEARS[1:], four), asof), 20.0, None, asof)
    assert m.pts["eps_up"] == 5.0  # 3/3 만점
    v = dict(four, eps=[1.2, 1.4, 1.3, 1.9])
    assert vs.metrics(vs.annual_table(_fy_facts(YEARS[1:], v), asof), 20.0, None, asof).pts["eps_up"] == 2.5  # 2/3 절반


def test_split_adjusts_eps_and_shares_e1_way():
    """2014 10-K 접수(2015-03-01) 뒤 2015-06-01 2:1 분할 → 판단일 2015-07-01에는 EPS ÷2, 주식 수 ×2."""
    asof = pd.Timestamp("2015-07-01")
    t = vs.annual_table(_fy_facts(YEARS, BASE), asof)
    splits = pd.Series([2.0], index=[pd.Timestamp("2015-06-01")])
    m0 = vs.metrics(t, price=19.0, splits=None, asof=asof)
    m1 = vs.metrics(t, price=9.5, splits=splits, asof=asof)  # 분할 뒤 실제 가격도 절반
    assert m1.peg == pytest.approx(m0.peg)
    assert m1.raw["fcf_yield"] == pytest.approx(m0.raw["fcf_yield"])
    # 판단일 뒤 분할은 쓰지 않는다
    later = pd.Series([2.0], index=[pd.Timestamp("2015-08-01")])
    assert vs.metrics(t, 19.0, later, asof).peg == pytest.approx(m0.peg)


def test_profitable_requires_three_positive_years():
    asof = pd.Timestamp("2015-03-02")
    v = dict(BASE, ni=[10, 12, 14, -1, 19])
    assert not vs.metrics(vs.annual_table(_fy_facts(YEARS, v), asof), 20.0, None, asof).profitable


def test_roic_na_when_denominator_nonpositive():
    asof = pd.Timestamp("2015-03-02")
    v = dict(BASE, equity=[60, 65, 70, 75, -40])
    assert vs.metrics(vs.annual_table(_fy_facts(YEARS, v), asof), 20.0, None, asof).pts["roic"] is None


def test_tercile_points():
    pool = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    assert vs.tercile_points(9, pool) == 5.0
    assert vs.tercile_points(7, pool) == 5.0
    assert vs.tercile_points(6, pool) == 2.5
    assert vs.tercile_points(3, pool) == 0.0
    assert vs.tercile_points(None, pool) is None


def _m(pts: dict, raw=None, peg=1.0) -> vs.Metrics:
    full = {k: None for k in vs.ITEMS}
    full.update(pts)
    return vs.Metrics(5, True, full, raw or {"gpa": None, "fcf_yield": None}, peg)


def test_score_day_prorates_na_and_drops_over_two():
    all_pts = {"peg": 20.0, "rev_growth": 10.0, "roic": 10.0, "eps_up": 5.0, "ocf_ni": 5.0, "fcf": 5.0,
               "net_debt": 5.0, "shares": 5.0}
    raw = {"gpa": 1.0, "fcf_yield": 1.0}
    ms = {"A": _m(all_pts, raw), "B": _m({**all_pts, "roic": None}, raw),
          "C": _m({**all_pts, "roic": None, "eps_up": None}, {"gpa": None, "fcf_yield": 1.0})}
    sc = vs.score_day(ms, ["A", "B", "C"])
    assert sc.at["A", "total"] == pytest.approx(70.0)  # 배점 합 75 → 70점 만점 환산
    assert sc.at["B", "total"] == pytest.approx(70.0)  # N/A 1개: 비례 환산, 0점 아님
    assert np.isnan(sc.at["C", "total"])  # N/A 3개 → 순위 제외
    assert sc.at["C", "n_na"] == 3


def test_total_from_points_matches_score_day_and_v1_weights():
    rng = np.random.default_rng(0)
    ms = {}
    for i in range(30):
        pts = {k: float(rng.choice([0.0, vs.MAX_PTS[k] / 2, vs.MAX_PTS[k]])) for k in vs.ITEMS if k not in vs.CROSS}
        if i % 7 == 0:
            pts["roic"] = None
        ms[f"T{i}"] = _m(pts, {"gpa": float(rng.random()), "fcf_yield": float(rng.random())})
    sc = vs.score_day(ms, list(ms))
    again = vs.total_from_points(sc, vs.MAX_PTS)
    assert np.allclose(sc["total"].to_numpy(), again.to_numpy(), equal_nan=True)
    v1 = dict(vs.MAX_PTS, peg=0.0, fcf_yield=15.0, gpa=10.0)
    direct = vs.score_day(ms, list(ms), v1)["total"]
    # 순위 항목은 기본 배점 점수를 비율로 늘린 것과 같다
    assert np.allclose(direct.to_numpy(), vs.total_from_points(sc, v1).to_numpy(), equal_nan=True)


def test_rank_order_ties_by_peg():
    df = pd.DataFrame({"total": [60.0, 60.0, 50.0, np.nan], "peg_value": [1.5, 0.8, 0.1, 0.1]}, index=["A", "B", "C", "D"])
    assert vs.rank_order(df) == ["B", "A", "C"]


def test_pick_sector_cap_counts_holdings():
    sic2 = {"A": "28", "B": "28", "C": "28", "D": "35", "E": "28"}
    assert vs.pick(["A", "B", "C", "D", "E"], 10, sic2, held=[]) == ["A", "B", "C", "D"]
    assert vs.pick(["B", "C", "D"], 10, sic2, held=["A", "E"]) == ["B", "D"]


def test_swap_choice_same_sector_first_then_overall():
    sic2 = {"H1": "28", "H2": "35", "C1": "36", "C2": "28"}
    held = {"H1": 40.0, "H2": 38.0}
    # C2(같은 업종 28) 51점: 그 업종 최저 H1 40 → 차 11 → H1 매도. 단 56점 미만이라 후보 아님
    assert vs.swap_choice(["C1", "C2"], {"C1": 50.0, "C2": 51.0}, held, sic2) is None
    # C2 60점: 같은 업종 우선 → (H1, C2). C1이 더 높아도 같은 업종 비교가 먼저
    assert vs.swap_choice(["C1", "C2"], {"C1": 65.0, "C2": 60.0}, held, sic2) == ("H1", "C2")
    # 같은 업종 비교 실패(차 < 10) → 전체 최저 H2(38)와 비교
    assert vs.swap_choice(["C1", "C2"], {"C1": 58.0, "C2": 57.0}, {"H1": 50.0, "H2": 38.0}, sic2) == ("H2", "C1")
    # 점수 차 10 미만이면 없음
    assert vs.swap_choice(["C1"], {"C1": 57.0}, {"H1": 50.0, "H2": 48.0}, sic2) is None


def test_swap_qty_rules():
    assert vs.swap_qty(1_000_000, 10_000, 1000) == 100  # 공제 ÷ 1주당 이익
    assert vs.swap_qty(10_000_000, 10_000, 1000) == 500  # 절반이 최대
    assert vs.swap_qty(1_000_000, -5_000, 1000) == 500  # 손실이면 절반
    assert vs.swap_qty(0, 10_000, 1000) == 0  # 남은 공제 ≤ 0 → 건너뜀


def test_no_lookahead_truncation():
    """판단일 t까지 자른 사실·분할로 계산한 점수 = 전체 데이터로 계산한 t일 점수 (CLAUDE.md 미래 데이터 방지)."""
    f = _fy_facts([2008, 2009] + YEARS, {k: [v[0] * 0.8, v[0] * 0.9] + v for k, v in BASE.items()})
    splits = pd.Series([2.0, 3.0], index=[pd.Timestamp("2013-05-01"), pd.Timestamp("2015-05-01")])
    for asof in pd.to_datetime(["2012-06-29", "2013-06-28", "2014-03-31", "2015-06-30"]):
        cut = f[f["filed"] < asof.date().isoformat()]
        a = vs.metrics(vs.annual_table(f, asof), 25.0, splits, asof)
        b = vs.metrics(vs.annual_table(cut, asof), 25.0, splits[splits.index <= asof], asof)
        assert a.pts == b.pts and a.raw == b.raw and (a.peg == b.peg)
