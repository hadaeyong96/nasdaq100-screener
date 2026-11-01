"""core/leverage_timing.py 테스트 (합성 데이터, 네트워크 없음) — 미래 데이터 방지·공개 시점·무작위 타이밍."""

import numpy as np
import pandas as pd

from core import leverage_timing as lt


def _close(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n))), index=pd.bdate_range("2000-01-03", periods=n))


def test_regime_targets_do_not_change_when_future_prices_change():
    c = _close()
    a = lt.regime_targets(c, lt.W_15X)
    c2 = c.copy()
    c2.iloc[400:] *= np.linspace(0.3, 1.8, len(c2) - 400)
    b = lt.regime_targets(c2, lt.W_15X)
    pd.testing.assert_series_equal(a.iloc[:400], b.iloc[:400])
    ev_a = [e for e in lt.events_from_daily_targets(a.to_numpy(), 250) if e[0] <= 400]
    ev_b = [e for e in lt.events_from_daily_targets(b.to_numpy(), 250) if e[0] <= 400]
    assert ev_a == ev_b and all(w in (lt.W_1X, lt.W_15X) for _, w in ev_a)


def test_events_execute_next_day_only_on_change():
    t = np.array([1, 1, 0.5, 0.5, 1, 1.0])
    assert lt.events_from_daily_targets(t, 0) == [(3, 0.5), (5, 1.0)]
    assert lt.events_from_monthly([10, 30, 50], [0.5, 0.5, 1.0]) == [(11, 0.5), (51, 1.0)]
    d = pd.bdate_range("2020-01-27", periods=30)
    assert [d[p].date().isoformat() for p in lt.month_end_positions(d, 0, len(d))] == ["2020-01-31", "2020-02-28"]


def test_m1_uses_previous_day_and_monthly_fallback_with_lag():
    t = pd.Series([0.5, -0.2], index=pd.to_datetime(["1990-01-30", "1990-01-31"]))
    assert lt.m1_decision(pd.Timestamp("1990-01-31"), t, pd.Series(dtype=float), pd.Series(dtype=float)) == (lt.W_15X, "T10Y2Y")
    gs10 = pd.Series([5.0, 4.0], index=pd.to_datetime(["1976-07-01", "1976-08-01"]))
    gs2 = pd.Series([4.0, 4.5], index=pd.to_datetime(["1976-07-01", "1976-08-01"]))
    empty = pd.Series(dtype=float)
    # 8월 값(역전)은 9월 1일부터 — 8/31 판단은 7월 값(+1.0)
    assert lt.m1_decision(pd.Timestamp("1976-08-31"), empty, gs10, gs2) == (lt.W_15X, "GS10−GS2")
    assert lt.m1_decision(pd.Timestamp("1976-09-30"), empty, gs10, gs2) == (lt.W_1X, "GS10−GS2")


def test_m3_never_uses_unreleased_month():
    months = pd.date_range("2000-01-01", periods=30, freq="MS")
    m2 = pd.Series(np.linspace(100, 130, 30), index=months)
    day = pd.Timestamp("2002-03-29")  # 알려진 마지막 달 = 2002-01 (m+2월 1일 규칙)
    base = lt.m3_decision(day, m2)
    m2b = m2.copy()
    m2b[m2b.index >= "2002-02-01"] = 1e9  # 아직 공개 안 된 달을 바꿔도
    assert lt.m3_decision(day, m2b) == base
    assert lt.nfci_current_asof(pd.Series([-0.3, 0.4], index=pd.to_datetime(["2005-01-07", "2005-01-14"])),
                                pd.Timestamp("2005-01-18")) == -0.3  # 1/14 값은 1/19부터


def test_random_timing_keeps_spell_lengths_and_ratio():
    rng = np.random.default_rng(1)
    lev = [False] * 5 + [True] * 10 + [False] * 3 + [True] * 2 + [False] * 7
    for _ in range(100):
        seq = lt.shuffle_spells(lt.spells(lev), rng)
        assert len(seq) == len(lev) and sum(seq) == sum(lev)
        assert sorted(n for x, n in lt.spells(seq) if x) == [2, 10]
    ev = lt.random_daily_events(lev, 100, lt.W_2X, rng)
    w = lt.exposure(ev, 100, 100 + len(lev))
    assert int((w[100:] != lt.W_1X).sum()) == sum(lev)
