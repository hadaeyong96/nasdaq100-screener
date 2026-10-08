"""L1d 레버리지 일반화·금리 연결·봉인·신호 테스트 (네트워크 없음, 합성 데이터)."""

from __future__ import annotations

import inspect
import re

import numpy as np
import pandas as pd
import pytest

from core import l1b, l1d
from data import l1d_history as hist


def _days(start="1990-01-02", periods=700):
    return pd.bdate_range(start, periods=periods)


def _close(td, seed=1, vol=0.015):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, vol, len(td)))), index=td)


# ── 레버리지 일반화 (k=2 = L1b 회귀) ────────────────────────────────────────


def test_k2_equals_l1b_formula_exactly():
    td = _days(periods=300)
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 0.02, len(td)), index=td)
    rf = pd.Series(rng.uniform(-0.0001, 0.0003, len(td)), index=td)
    old = (2.0 * r - rf - 0.0095 / 252).clip(lower=-1.0)  # L1b 원래 식
    pd.testing.assert_series_equal(l1b.synthetic_leveraged_returns(r, rf, 2.0), old)
    pd.testing.assert_series_equal(l1b.synthetic_2x_returns(r, rf), old)


def test_k2_full_simulation_equals_l1b_path():
    """L1d G1 경로(core.l1d) = L1b B1 경로(core.l1b)를 같은 입력으로 돌린 결과."""
    td = _days(periods=900)
    close = _close(td, vol=0.02)
    rf = pd.Series(0.0001, index=td)
    rets_d = l1d.market_returns(close, rf)
    r = close.pct_change().fillna(0.0)
    rets_b = pd.DataFrame({"1x": r, "2x": l1b.synthetic_2x_returns(r, rf), "cash": rf})
    held_d = l1d.held_after_close(l1d.target("G1", close)).loc[td[210]:]
    held_b = l1b.held_after_close_daily(l1b.target_b1(close, 200)).loc[td[210]:]
    pd.testing.assert_series_equal(held_d, held_b)
    a, b = l1b.simulate(held_d, rets_d), l1b.simulate(held_b, rets_b)
    pd.testing.assert_series_equal(a["values"], b["values"])
    assert a["cagr_pct"] == b["cagr_pct"] and a["taxes"] == b["taxes"]


def test_k3_synthetic():
    td = _days(periods=3)
    r = pd.Series([0.01, -0.02, -0.40], index=td)
    rf = pd.Series([0.0002, -0.0001, 0.0], index=td)
    out = l1b.synthetic_leveraged_returns(r, rf, 3.0)
    fee = 0.0095 / 252
    assert out.iloc[0] == pytest.approx(0.03 - 2 * 0.0002 - fee)
    assert out.iloc[1] == pytest.approx(-0.06 + 2 * 0.0001 - fee)  # 음수 금리 = 차입 비용 음수
    assert out.iloc[2] == -1.0  # −120% → −100%로 자름


def test_wipeout_stays_zero():
    td = _days(periods=6)
    rets = pd.DataFrame({"1x": [0, 0.01, -0.34, 0.5, 0.5, 0.1], "cash": 0.0}, index=td)
    rets["3x"] = l1b.synthetic_leveraged_returns(rets["1x"], rets["cash"], 3.0)
    held = pd.Series(["3x", "3x", "3x", "cash", "3x", "3x"], index=td)
    sim = l1b.simulate(held, rets)
    assert (sim["values"].iloc[2:] == 0).all()
    assert sim["after_tax_final"] == 0 and sim["cagr_pct"] == -100.0
    assert sim["switches"] == 0  # 0이 된 뒤에는 거래도 없다


def test_targets_g0_g1_g2():
    td = _days(periods=5)
    lv = pd.Series([1.0, 1.0, 1.0, 2.0, 1.0], index=td)
    assert list(l1d.target("G0", lv, 3)) == ["1x"] * 5
    g2 = l1d.target("G2", lv, 3)
    assert g2.iloc[:2].isna().all() and list(g2.iloc[2:]) == ["cash", "3x", "cash"]
    assert list(l1d.target("G1", lv, 3).iloc[2:]) == ["cash", "2x", "cash"]


# ── 미래 데이터 방지 ───────────────────────────────────────────────────────


@pytest.mark.parametrize("cut", [250, 400, 650])
@pytest.mark.parametrize("cand", ["G1", "G2"])
def test_truncated_equals_full(cand, cut):
    td = _days()
    close = _close(td, vol=0.03)
    t = td[cut]
    full = l1d.held_after_close(l1d.target(cand, close, 190))
    trunc = l1d.held_after_close(l1d.target(cand, close.loc[:t], 190))
    pd.testing.assert_series_equal(full.loc[:t], trunc)
    rf = pd.Series(0.0001, index=td)
    pd.testing.assert_frame_equal(l1d.market_returns(close, rf).loc[:t], l1d.market_returns(close.loc[:t], rf.loc[:t]))


def test_no_negative_shift_in_sources():
    from scripts import l1d_experiments

    for mod in (l1d, hist, l1d_experiments):
        assert not re.search(r"shift\(\s*-", inspect.getsource(mod)), mod.__name__


# ── 금리 연결·대체 표시 ─────────────────────────────────────────────────────


def test_rate_chain_priority_and_fallback_marked():
    months = pd.period_range("2000-01", "2000-06", freq="M")
    a = pd.Series([1.0, 1.1], index=pd.period_range("2000-05", "2000-06", freq="M"))   # 1순위: 뒤 두 달만
    b = pd.Series([2.0, 2.1, 2.2], index=pd.period_range("2000-02", "2000-04", freq="M"))  # 2순위
    dtb3 = pd.Series(5.0, index=pd.period_range("1999-01", "2001-12", freq="M"))
    chain = l1d.chain_monthly_rates({"A": a, "B": b, "DTB3": dtb3}, ["A", "B", "DTB3"], months)
    assert list(chain["source"]) == ["DTB3", "B", "B", "B", "A", "A"]
    assert list(chain["rate_pct"]) == [5.0, 2.0, 2.1, 2.2, 1.0, 1.1]
    periods = l1d.source_periods(chain)
    assert [(p["source"], p["start"], p["end"], p["months"]) for p in periods] == [
        ("DTB3", "2000-01", "2000-01", 1), ("B", "2000-02", "2000-04", 3), ("A", "2000-05", "2000-06", 2)]
    missing = l1d.chain_monthly_rates({"A": a}, ["A"], months)
    assert missing["rate_pct"].isna().sum() == 4 and missing["source"].isna().sum() == 4


def test_daily_rate_uses_same_month_value_divided_by_252():
    chain = pd.DataFrame({"rate_pct": [2.52, -0.252], "source": ["X", "X"]}, index=pd.period_range("2000-01", "2000-02", freq="M"))
    days = pd.to_datetime(["2000-01-03", "2000-01-31", "2000-02-01"])
    out = l1d.daily_rate(days, chain)
    assert list(out.round(10)) == [0.0001, 0.0001, -0.00001]


# ── 봉인·정리 ────────────────────────────────────────────────────────────


def test_seal_2022_rows():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2021-12-31", "2022-01-03"]))
    with pytest.raises(l1d.SealError):
        l1d.seal(s, "x")
    assert len(l1d.cut_and_seal(s, "x")) == 1
    assert l1d.SEAL_DATE == pd.Timestamp("2021-12-31")


def test_clean_index():
    idx = pd.to_datetime(["2000-01-03", "2000-01-04", "2000-01-04", "2000-01-05", "2000-01-06"])
    s = pd.Series([100.0, 101.0, 102.0, np.nan, 0.0], index=idx)
    out, info = l1d.clean_index(s)
    assert list(out) == [100.0, 102.0] and info == {"duplicates": 1, "nan": 1, "nonpositive": 1}
