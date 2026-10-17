"""core/shock_dip.py 테스트 (합성 데이터, 네트워크 없음) — 관문 2 실시간 신호."""

import numpy as np
import pandas as pd

from core import shock_dip as sd


def _days(n, start="2000-01-03"):
    return pd.bdate_range(start, periods=n)


def test_correct_with_reference_replaces_outlier_and_records_dates():
    d = _days(5)
    ref = pd.Series([100, 101, 102, 103, 104.0], index=d)
    q = pd.Series([10, 10.1, 9.0, 10.3, 10.4], index=d)  # 3번째 날 이상치
    out, bad = sd.correct_with_reference(q, ref, 0.02)
    assert bad == [d[2], d[3]]
    assert np.allclose(out.to_numpy(), (ref / 10).to_numpy())


def test_asof_daily_never_uses_unreleased_values():
    months = pd.to_datetime(["2020-01-01", "2020-02-01"])
    vals = pd.Series([1.0, 2.0], index=months)
    rel = pd.Series(pd.to_datetime(["2020-02-07", "2020-03-06"]), index=months)
    days = pd.DatetimeIndex(pd.to_datetime(["2020-02-06", "2020-02-07", "2020-03-05", "2020-03-06", "2020-03-09"]))
    us = sd.asof_daily(vals, rel, days, strict=False)
    fx = sd.asof_daily(vals, rel, days, strict=True)
    assert np.isnan(us.iloc[0]) and us.iloc[1] == 1.0 and us.iloc[2] == 1.0 and us.iloc[3] == 2.0
    assert np.isnan(fx.iloc[1]) and fx.iloc[3] == 1.0 and fx.iloc[4] == 2.0  # 해외: 발표 다음 거래일부터


def test_month_start_available_and_percentile_known_only():
    months = pd.date_range("2000-01-01", periods=4, freq="MS")
    vals = pd.Series([1.0, 3.0, 2.0, 0.5], index=months)
    avail = sd.month_start_available(months, 1)
    assert avail.iloc[0] == pd.Timestamp("2000-02-01")
    days = pd.DatetimeIndex(pd.to_datetime(["2000-02-01", "2000-03-01", "2000-04-03", "2000-05-01"]))
    p = sd.asof_percentile(vals, avail, days, strict=False)
    assert list(p.round(4)) == [1.0, 1.0, round(2 / 3, 4), 0.25]
    # 나중 값을 바꿔도 앞 날짜 백분위는 그대로
    vals2 = vals.copy()
    vals2.iloc[3] = 99.0
    p2 = sd.asof_percentile(vals2, avail, days, strict=False)
    assert list(p2.iloc[:3]) == list(p.iloc[:3])


def _synthetic(seed=1, n=1600):
    rng = np.random.default_rng(seed)
    d = _days(n, "1990-01-02")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n))), index=d)
    months = pd.date_range("1985-01-01", d[-1], freq="MS")
    macro = pd.Series(rng.uniform(-0.2, 0.8, len(months)), index=months)
    fear = pd.Series(rng.uniform(10, 60, n), index=d)
    return close, macro, fear


def _signals(close, macro, fear):
    d = close.index
    f = sd.price_features(close)
    sahm = sd.asof_daily(macro, sd.month_start_available(macro.index, 1), d, strict=False)
    pct = sd.asof_percentile(macro, sd.month_start_available(macro.index, 2), d, strict=False)
    conds = sd.variant_conditions(f, sahm, pct, pct, fear)
    allowed = pd.Series(True, index=d)
    return {v: sd.simulate(v, f, conds[v], sahm, allowed) for v in sd.VARIANTS}


def test_signals_before_cutoff_unchanged_when_future_data_changes():
    close, macro, fear = _synthetic()
    base = _signals(close, macro, fear)
    cut = close.index[1000]
    close2, macro2, fear2 = close.copy(), macro.copy(), fear.copy()
    close2[close2.index > cut] *= np.linspace(0.3, 2.0, int((close2.index > cut).sum()))
    macro2[macro2.index > cut - pd.DateOffset(months=1)] = 5.0  # 아직 발표 안 된 달 포함
    fear2[fear2.index > cut] = 99.0
    alt = _signals(close2, macro2, fear2)
    assert sum(len(v) for v in base.values()) > 0
    for v in sd.VARIANTS:
        a = [t.signal_pos for t in base[v] if t.signal_pos <= 1000]
        b = [t.signal_pos for t in alt[v] if t.signal_pos <= 1000]
        assert a == b, v
        done_a = [t for t in base[v] if t.exit_pos is not None and t.exit_pos <= 1000]
        done_b = [t for t in alt[v] if t.exit_pos is not None and t.exit_pos <= 1000]
        assert done_a == done_b, v


def test_b3_fixed_20_days_and_ignores_signals_while_holding():
    n = 80
    d = _days(n)
    close = pd.Series(100.0, index=d)
    f = sd.price_features(close)
    fear = pd.Series(10.0, index=d)
    fear.iloc[[5, 10, 30]] = 45
    nan = pd.Series(np.nan, index=d)
    trades = sd.simulate("B3", f, sd.variant_conditions(f, nan, nan, nan, fear)["B3"], nan, pd.Series(True, index=d))
    assert [(t.signal_pos, t.entry_pos, t.exit_pos) for t in trades] == [(5, 6, 26), (30, 31, 51)]


def test_b4_cancels_without_confirmation_and_d_exits_below_200():
    n = 400
    d = _days(n)
    # 300일 상승 뒤 꾸준히 하락 → 20일선 위로 못 올라옴
    close = pd.Series(np.r_[np.linspace(100, 200, 300), np.linspace(200, 80, 100)], index=d)
    f = sd.price_features(close)
    sahm = pd.Series(0.0, index=d)
    cond = sd.base_condition(f, sahm)
    assert cond.any()
    assert sd.simulate("B4", f, cond, sahm, pd.Series(True, index=d)) == []
    dc = sd.variant_conditions(f, sahm, sahm, sahm, sahm)["D"]
    trades = sd.simulate("D", f, dc, sahm, pd.Series(True, index=d))
    assert trades and trades[0].reason == "below200"


def test_time_exit_is_exactly_n_trading_days_after_entry():
    # v3-1: 체결일 다음 거래일부터 60번째 거래일 종가에 청산(판단일 = 체결일)
    n = 400
    d = _days(n)
    close = pd.Series(np.r_[np.linspace(100, 200, 300), np.linspace(200, 160, 25), np.full(75, 160.0)], index=d)
    f = sd.price_features(close)
    sahm = pd.Series(0.0, index=d)
    trades = sd.simulate("A", f, sd.base_condition(f, sahm), sahm, pd.Series(True, index=d))
    t = trades[0]
    assert t.reason == "time" and t.exit_pos - t.entry_pos == 60 and t.exit_signal_pos == t.exit_pos
