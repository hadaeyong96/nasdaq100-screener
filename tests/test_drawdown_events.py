"""core/drawdown_events.py 테스트 (합성 데이터, 네트워크 없음)."""

import numpy as np
import pandas as pd

from core import drawdown_events as de


def _series(values, start="2000-01-03"):
    return pd.Series(np.asarray(values, dtype=float), index=pd.bdate_range(start, periods=len(values)))


def test_single_event_fields():
    # 100 → 80(−20%) → 105 회복
    s = _series([100, 95, 90, 85, 80, 90, 100, 105])
    ev = de.find_drawdown_events(s, -0.15)
    assert len(ev) == 1
    e = ev.iloc[0]
    assert e["peak_date"] == s.index[0] and e["peak"] == 100
    assert e["hit_date"] == s.index[3]  # 85 = −15% 첫날
    assert e["trough_date"] == s.index[4] and abs(e["max_dd"] + 0.20) < 1e-12
    assert e["peak_to_trough_days"] == 4
    assert e["recovery_date"] == s.index[6] and e["peak_to_recovery_days"] == 6
    assert e["fast20"]  # 4일 만에 −10% 이하


def test_redrop_before_recovery_is_one_event_and_unrecovered():
    s = _series([100, 80, 95, 70, 90])  # 회복 전 재하락, 끝까지 미회복
    ev = de.find_drawdown_events(s, -0.15)
    assert len(ev) == 1
    e = ev.iloc[0]
    assert e["trough"] == 70 and pd.isna(e["recovery_date"]) and np.isnan(e["peak_to_recovery_days"])


def test_shallow_dip_is_not_event_and_slow_flag():
    s = _series([100, 90, 101])
    assert de.find_drawdown_events(s, -0.15).empty
    slow = _series(list(np.linspace(100, 80, 60)) + [101])  # 20거래일 안에 −10% 못 감
    e = de.find_drawdown_events(slow, -0.15).iloc[0]
    assert not e["fast20"]


def test_touch_sma_forward_and_further():
    s = _series([100] * 60 + [90, 70, 60, 50, 55, 120] + [120] * 10)
    e = de.find_drawdown_events(s, -0.15).iloc[0]
    p30 = de.first_touch_pos(s, e, -0.30)
    assert s.iloc[p30] == 70
    assert de.first_touch_pos(s, e, -0.60) is None
    assert de.further_drawdown(s, p30, e["end_pos"]) == 50 / 70 - 1
    assert de.forward_return(s, p30, 3) == 55 / 70 - 1
    assert np.isnan(de.forward_return(s, p30, 1000))
    assert s.iloc[de.first_above_sma_pos(s, p30, 50)] == 120


def test_asof_and_percentile_use_only_known_values():
    vals = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]))
    pub = pd.Series(pd.to_datetime(["2020-02-07", "2020-03-06", "2020-04-03"]), index=vals.index)
    assert np.isnan(de.asof_value(vals, pub, pd.Timestamp("2020-02-06")))
    assert de.asof_value(vals, pub, pd.Timestamp("2020-03-06")) == 2.0
    assert de.expanding_percentile(pd.Series([1, 2, 3, 4.0]), 3.0, 4) == 0.75
    assert np.isnan(de.expanding_percentile(pd.Series([1.0]), 1.0, 4))


def test_events_up_to_t_match_full_history_for_completed_events():
    # 미래 데이터 방지: t까지 자른 데이터에서 이미 회복된 사건은 전체 데이터와 같아야 한다
    rng = np.random.default_rng(0)
    s = _series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 1500))))
    full = de.find_drawdown_events(s)
    cut = de.find_drawdown_events(s.iloc[:1000])
    done = cut[cut["recovery_date"].notna()]
    cols = ["peak_date", "trough_date", "recovery_date", "max_dd"]
    pd.testing.assert_frame_equal(done[cols].reset_index(drop=True), full[cols].iloc[: len(done)].reset_index(drop=True))


def test_provisional_type():
    nan = np.nan
    assert de.provisional_type(0.1, 0.5, 0.5, 0.0, True) == "충격"
    assert de.provisional_type(0.1, 0.5, 0.5, 0.0, False) == "UNKNOWN"
    assert de.provisional_type(0.1, nan, 0.5, 0.0, True) == "UNKNOWN"
    assert de.provisional_type(0.6, nan, 0.5, 0.0, True) == "침체"
    assert de.provisional_type(0.1, 0.5, 0.95, 1.5, True) == "혼합(거품·긴축)"
