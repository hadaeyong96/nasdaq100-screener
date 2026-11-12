"""core/pead_filters.py — 실적급등 보완 필터 순수 함수 (합성 데이터, 네트워크 없음)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import pead_filters as pf
from data import pead_filters_data as fd

DAYS = pd.bdate_range("2015-01-01", periods=400)


def _close(seed=1, n=400):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n)), index=DAYS[:n])


# ── F1 ──────────────────────────────────────────────────────────────────────


def test_f1_close_location():
    assert pf.f1_hold(110, 100, 107.5) == (True, "")  # 0.75 경계 포함
    assert pf.f1_hold(110, 100, 107.4) == (False, "")
    assert pf.f1_hold(100, 100, 100)[0] is None  # 고가 = 저가
    assert pf.f1_hold(np.nan, 100, 105)[0] is None
    assert pf.f1_hold(110, 100, 111)[0] is None  # 같은 출처 불일치


# ── F2 · F3 · F6 ────────────────────────────────────────────────────────────


def test_f2_uses_prior_252_closes_excluding_d0():
    c = pd.Series(100.0, index=DAYS[:300])
    c.iloc[100] = 200.0  # 직전 252개 안의 최고 200
    d0 = DAYS[299]
    c.iloc[299] = 190.0  # 190 ≥ 200 × 0.95
    assert pf.f2_near_high(c, d0) == (True, "")
    c.iloc[299] = 189.0
    assert pf.f2_near_high(c, d0) == (False, "")
    c.iloc[30] = 1000.0  # 252개 밖(299 − 252 = 47 전)은 안 본다
    assert pf.f2_near_high(c, d0) == (False, "")
    c.iloc[30] = 100.0
    c.iloc[299] = 500.0  # d0 자신은 최고값에 안 들어감 → 통과
    assert pf.f2_near_high(c, d0) == (True, "")


def test_f2_needs_252_prior_rows():
    c = pd.Series(100.0, index=DAYS[:252])
    assert pf.f2_near_high(c, DAYS[251])[0] is None  # 직전 251개
    c2 = pd.Series(100.0, index=DAYS[:253])
    assert pf.f2_near_high(c2, DAYS[252])[0] is True


def test_above_sma_includes_d0_and_needs_200():
    c = pd.Series(100.0, index=DAYS[:200])
    c.iloc[-1] = 101.0
    assert pf.above_sma(c, DAYS[199]) == (True, "")
    assert pf.above_sma(c.iloc[:199], DAYS[198])[0] is None
    assert pf.above_sma(c, DAYS[300])[0] is None  # d0 행 없음


def test_missing_d0_row_is_na():
    c = _close().drop(index=[DAYS[350]])
    assert pf.f2_near_high(c, DAYS[350])[0] is None
    assert pf.above_sma(c, DAYS[350])[0] is None


@pytest.mark.parametrize("fn", [pf.f2_near_high, pf.above_sma])
def test_price_filters_no_future_data(fn):
    """t일까지 자른 종가와 전체 종가의 t일 판정이 같다."""
    c = _close(seed=7)
    for i in range(255, 400, 7):
        d0 = DAYS[i]
        assert fn(c.loc[:d0], d0) == fn(c, d0)
        fut = c.copy()
        fut.loc[fut.index > d0] *= 3.0  # 미래를 바꿔도 같아야 한다
        assert fn(fut, d0) == fn(c, d0)


# ── XBRL ────────────────────────────────────────────────────────────────────


def _q(start, end, val, filed, accn="A"):
    return {"start": start, "end": end, "val": val, "filed": filed, "accn": accn}


def _rev_facts():
    """분기 매출: 2019Q1~Q4 100,100,100,100 / 2020Q1 110 Q2 130. 2020Q2 값은 10-Q(Q2) 2020-07-30 제출."""
    e = [
        _q("2019-01-01", "2019-03-31", 100, "2019-04-30", "q119"),
        _q("2019-04-01", "2019-06-30", 100, "2019-07-30", "q219"),
        _q("2020-01-01", "2020-03-31", 110, "2020-04-30", "q120"),
        _q("2019-01-01", "2019-03-31", 100, "2020-04-30", "q120"),
        _q("2020-04-01", "2020-06-30", 130, "2020-07-30", "q220"),
        _q("2019-04-01", "2019-06-30", 100, "2020-07-30", "q220"),
    ]
    return pf.fact_frame(e)


def _op_facts(prev_q2=10.0):
    e = [
        _q("2019-04-01", "2019-06-30", prev_q2, "2019-07-30", "q219"),
        _q("2020-04-01", "2020-06-30", 20, "2020-07-30", "q220"),
        _q("2019-04-01", "2019-06-30", prev_q2, "2020-07-30", "q220"),
    ]
    return pf.fact_frame(e)


def test_known_cutoff_rules():
    df = _rev_facts()
    d0 = pd.Timestamp("2020-07-30")
    assert "q220" not in set(pf.known(df, d0, None, None)["accn"])  # 제출일 = d0, 시각 모름 → 모름
    k = pf.known(df, d0, "q220", pd.Timestamp("2020-07-30 15:59"))
    assert "q220" in set(k["accn"])
    assert "q220" not in set(pf.known(df, d0, "q220", pd.Timestamp("2020-07-30 16:00"))["accn"])
    assert "q220" not in set(pf.known(df, d0, "other", pd.Timestamp("2020-07-30 08:00"))["accn"])
    assert "q220" in set(pf.known(df, pd.Timestamp("2020-07-31"), None, None)["accn"])


def test_f4_accel_and_late_report():
    tags = {"Revenues": _rev_facts()}
    end = pd.Timestamp("2020-06-30")
    d0 = pd.Timestamp("2020-07-31")
    kn = {t: pf.known(df, d0, None, None) for t, df in tags.items()}
    assert pf.f4_rev_accel(kn, end, False) == (True, "")  # +30% > +10%
    d0e = pd.Timestamp("2020-07-29")  # 10-Q가 반응일 뒤
    kn = {t: pf.known(df, d0e, None, None) for t, df in tags.items()}
    assert pf.f4_rev_accel(kn, end, True) == (None, pf.NA_LATE)


def test_f5_margin_and_op_base():
    end, d0 = pd.Timestamp("2020-06-30"), pd.Timestamp("2020-07-31")
    tags = {"Revenues": _rev_facts(), "OperatingIncomeLoss": _op_facts()}
    kn = {t: pf.known(df, d0, None, None) for t, df in tags.items()}
    assert pf.f5_margin_up(kn, end, False) == (True, "")  # 영업이익 +100% > 매출 +30%
    tags["OperatingIncomeLoss"] = _op_facts(prev_q2=-5.0)
    kn = {t: pf.known(df, d0, None, None) for t, df in tags.items()}
    assert pf.f5_margin_up(kn, end, False) == (None, pf.NA_OPBASE)
    assert pf.f5_margin_up({"Revenues": kn["Revenues"]}, end, False) == (None, pf.NA_NOVAL)


def test_q4_derived_from_annual_minus_nine_months():
    e = [
        _q("2018-01-01", "2018-12-31", 400, "2019-02-20", "k18"),
        _q("2018-01-01", "2018-09-30", 300, "2018-10-30", "q318"),
        _q("2019-01-01", "2019-12-31", 500, "2020-02-20", "k19"),
        _q("2019-01-01", "2019-09-30", 340, "2019-10-30", "q319"),
        _q("2019-07-01", "2019-09-30", 120, "2019-10-30", "q319"),
        _q("2018-07-01", "2018-09-30", 100, "2019-10-30", "q319"),
    ]
    df = pf.fact_frame(e)
    r = pf.quarter_value(df, pd.Timestamp("2019-12-26"), pd.Timestamp("2020-01-05"))
    assert r[0] == pd.Timestamp("2019-12-31") and r[1] == pytest.approx(160)
    kn = {"Revenues": pf.known(df, pd.Timestamp("2020-02-21"), None, None)}
    # Q4 +60% (160/100) > Q3 +20% (120/100)
    assert pf.f4_rev_accel(kn, pd.Timestamp("2019-12-31"), False) == (True, "")


def test_revenue_pair_uses_same_tag():
    """이번 값은 새 태그에만, 전년 동기는 옛 태그에만 있으면 그 태그로는 비교하지 않는다."""
    new = pf.fact_frame([_q("2020-04-01", "2020-06-30", 130, "2020-07-30")])
    old = pf.fact_frame([_q("2019-04-01", "2019-06-30", 100, "2019-07-30")])
    tags = {"RevenueFromContractWithCustomerExcludingAssessedTax": new, "SalesRevenueNet": old}
    why, _, _ = pf.revenue_yoy(tags, pd.Timestamp("2020-06-25"), pd.Timestamp("2020-07-05"))
    assert why == pf.NA_NOCMP


def test_latest_known_restatement_wins_but_not_future():
    e = [
        _q("2019-04-01", "2019-06-30", 100, "2019-07-30", "a"),
        _q("2019-04-01", "2019-06-30", 120, "2020-01-15", "b"),  # 정정
        _q("2019-04-01", "2019-06-30", 50, "2021-01-15", "c"),  # 미래 정정
    ]
    df = pf.fact_frame(e)
    k = pf.known(df, pd.Timestamp("2020-07-31"), None, None)
    assert pf.quarter_value(k, pd.Timestamp("2019-06-25"), pd.Timestamp("2019-07-05"))[1] == 120


def test_xbrl_filters_no_future_data():
    """d0 뒤 제출 사실을 더해도(정정·다음 분기) 판정이 같다."""
    end, d0 = pd.Timestamp("2020-06-30"), pd.Timestamp("2020-07-31")
    base = {"Revenues": _rev_facts(), "OperatingIncomeLoss": _op_facts()}
    extra_rev = pf.fact_frame([_q("2020-04-01", "2020-06-30", 50, "2021-07-30", "z"),
                               _q("2019-04-01", "2019-06-30", 400, "2020-08-01", "y"),
                               _q("2020-07-01", "2020-09-30", 999, "2020-10-30", "x")])
    extra_op = pf.fact_frame([_q("2019-04-01", "2019-06-30", -100, "2020-08-02", "w")])
    fut = {"Revenues": pd.concat([base["Revenues"], extra_rev]),
           "OperatingIncomeLoss": pd.concat([base["OperatingIncomeLoss"], extra_op])}
    for fn in (pf.f4_rev_accel, pf.f5_margin_up):
        a = fn({t: pf.known(df, d0, None, None) for t, df in base.items()}, end, False)
        b = fn({t: pf.known(df, d0, None, None) for t, df in fut.items()}, end, False)
        assert a == b and a[0] is not None


# ── 고르는 규칙 ─────────────────────────────────────────────────────────────

BASE = {"n": 100, "mean": 0.67, "p1": 1.08, "p2": 0.23}


def _r(name, mean, p1, p2, n):
    return {"name": name, "mean": mean, "p1": p1, "p2": p2, "n": n}


def test_pick_pair_needs_both_periods_and_40pct():
    s = [_r("F1", 2.0, 2.0, 0.1, 80),  # 2017~21 미달
         _r("F2", 1.5, 1.5, 0.5, 39),  # 40% 미달
         _r("F3", 1.0, 1.2, 0.3, 40),  # 통과(경계)
         _r("F4", 1.2, 1.1, 0.9, 60),  # 통과
         _r("F5", 0.9, 1.1, 0.24, 70)]  # 통과
    ok, pair = pf.pick_pair(s, BASE)
    assert [r["name"] for r in ok] == ["F3", "F4", "F5"]
    assert pair == ("F3", "F4")


def test_final_choice_with_and_without_combo():
    passed = [_r("F3", 1.0, 1.2, 0.3, 40), _r("F4", 1.2, 1.1, 0.9, 60)]
    combo = _r("F3+F4", 1.5, 1.3, 0.5, 25)
    ch, ok = pf.final_choice(passed, combo, BASE)
    assert ok and [c["name"] for c in ch] == ["F3+F4", "F4"]
    small = _r("F3+F4", 3.0, 3.0, 3.0, 24)  # 25% 미달
    ch, ok = pf.final_choice(passed, small, BASE)
    assert not ok and [c["name"] for c in ch] == ["F4", "F3"]
    weak = _r("F3+F4", 3.0, 3.0, 0.1, 50)  # 두 구간 조건 미달
    assert pf.final_choice(passed, weak, BASE)[1] is False


def test_no_pass_keeps_no_filter():
    ok, pair = pf.pick_pair([_r("F1", 0.9, 2.0, 0.2, 90), _r("F2", 0.9, 1.0, 0.5, 90)], BASE)
    assert ok == [] and pair is None
    assert pf.final_choice(ok, None, BASE) == ([], False)


# ── 입출력 도우미 (순수 부분) ────────────────────────────────────────────────


def test_q_accn_of_matches_same_quarter_report():
    ev = pd.DataFrame({"form": ["10-Q", "8-K", "10-Q"], "filing_date": ["2020-07-30", "2020-07-29", "2020-10-30"],
                       "report_date": ["2020-06-30", "2020-06-30", "2020-09-30"], "accn": ["q2", "k", "q3"],
                       "cik": [1, 1, 1]})
    assert fd.q_accn_of(ev, 1, "2020-06-30", "2020-07-30") == "q2"
    assert fd.q_accn_of(ev, 2, "2020-06-30", "2020-07-30") is None
