"""pead-smallmid(실적급등 다음날 매수·3달 보유 — 중소형 재현) 순수 함수 테스트. 네트워크·캐시 없음 (합성 데이터)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core import sw1  # noqa: E402
from scripts import pead_smallmid_experiments as pe  # noqa: E402
from scripts import sw1_data_check as sw  # noqa: E402
import smallmid_prices_check as pc  # noqa: E402


def _px(n=120, seed=0, start="2017-01-03"):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n)
    close = 10 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    vol = rng.integers(200_000, 2_000_000, n).astype(float)
    return pd.DataFrame({"open": close * (1 + rng.normal(0, 0.005, n)), "close": close, "volume": vol}, index=days)


def test_liquid_flags_equals_liquid_before():
    px = _px()
    f = pe.liquid_flags(px)
    for d in px.index[:: 7]:
        assert bool(f[d]) == pc.liquid_before(px, d)[0]


def test_liquid_flags_no_future_data():
    """미래 데이터 방지: t일까지 자른 결과와 전체의 t일 값이 같다."""
    px = _px(seed=3)
    full = pe.liquid_flags(px)
    for t in px.index[20::9]:
        assert bool(pe.liquid_flags(px.loc[:t]).iloc[-1]) == bool(full[t])


def test_liquid_flags_ignores_same_day():
    """반응일 당일 거래량·가격은 유동성 판정에 안 쓴다."""
    px = _px(seed=5)
    d = px.index[60]
    px2 = px.copy()
    px2.loc[d, "volume"] = 1e12
    px2.loc[d, "close"] = 0.01
    assert pe.liquid_flags(px)[d] == pe.liquid_flags(px2)[d]


def test_signal_no_future_data():
    """신호 판정(sw.signal_on)은 반응일 뒤 가격을 바꿔도 같다."""
    px = _px(seed=7)
    q = _px(seed=8)["close"]
    d0 = px.index[70]
    a = sw.signal_on(px[["close", "volume"]], q, d0)
    px2 = px.copy()
    px2.loc[px.index > d0, ["close", "volume"]] *= 3
    assert a == sw.signal_on(px2[["close", "volume"]], q, d0)


def test_trade_exit_no_future_data():
    """매매 결과는 예정 매도일 뒤 가격을 바꿔도 같다."""
    px = _px(n=200, seed=9)
    days = px.index
    sig = pd.DataFrame({"ticker": [1], "d0": [days[30]]})
    a, _ = sw1.make_trades(sig, {1: px}, days, 60)
    px2 = px.copy()
    px2.loc[days > a["plan_sell_day"].iloc[0], ["open", "close"]] *= 5
    b, _ = sw1.make_trades(sig, {1: px2}, days, 60)
    pd.testing.assert_frame_equal(a, b)


def test_slippage_cost():
    c0 = sw1.Costs()
    c3 = sw1.Costs(slippage_pct=0.3)
    base = sw1.krw_after_cost(0.10, 1000.0, 1000.0, c0)
    with_s = sw1.krw_after_cost(0.10, 1000.0, 1000.0, c3)
    assert with_s == pytest.approx((1 + base) * 0.997 / 1.003 - 1)
    # 슬리피지 0이면 SW1과 같다
    assert sw1.krw_after_cost(0.10, 1000.0, 1100.0, sw1.Costs(slippage_pct=0.0)) == sw1.krw_after_cost(0.10, 1000.0, 1100.0)


def test_mean_without_top():
    x = pd.Series([0.5, 0.4, 0.3, 0.2, 0.1, -0.01, -0.02])
    assert pe.mean_without_top(x, 5) == pytest.approx(-0.015)
    assert np.isnan(pe.mean_without_top(pd.Series([1.0, 2.0]), 5))


def test_missed_trades_mean():
    x = pd.Series(np.arange(-9, 11) / 100.0)  # 20건, 평균 0.005
    r = pe.missed_trades_mean(x, undecided=30, signal_rate=0.1)
    assert r["k"] == 3 and r["n_low"] == 2
    assert r["fill"] == pytest.approx(-0.085)
    assert r["mean"] == pytest.approx((x.sum() + 3 * -0.085) / 23)


def _crit():
    return {"periods": [(2017, 2019), (2020, 2021)], "random_pct_min": 99.0, "min_pos_years": 3, "min_trades": 300,
            "drop_top": 5, "max_undecided": 0.10}


def _trades(mean, n=400):
    years = pd.Series(np.repeat([2017, 2018, 2019, 2020, 2021], n // 5))
    return pd.Series(np.full(n, mean)), years


def test_verdict_pass_and_fail_and_inconclusive():
    ex, yr = _trades(0.01)
    assert pe.verdict(ex, yr, 99.5, 0.05, 0.005, _crit())["result"] == "PASS"
    # ② 미달 → INCONCLUSIVE
    assert pe.verdict(ex, yr, 98.9, 0.05, 0.005, _crit())["result"] == "INCONCLUSIVE"
    # ⑦ 미달 → INCONCLUSIVE
    assert pe.verdict(ex, yr, 99.5, 0.05, -0.001, _crit())["result"] == "INCONCLUSIVE"
    # ⑧ 미달(판정 불가 > 10%) → INCONCLUSIVE
    assert pe.verdict(ex, yr, 99.5, 0.11, 0.005, _crit())["result"] == "INCONCLUSIVE"
    # ⑤ 300건 미만 → INCONCLUSIVE
    ex2, yr2 = _trades(0.01, n=250)
    assert pe.verdict(ex2, yr2, 99.5, 0.05, 0.005, _crit())["result"] == "INCONCLUSIVE"
    # ① 미달 → FAIL, ⑧과 겹쳐도 FAIL (사용자 결정)
    exn, yrn = _trades(-0.01)
    assert pe.verdict(exn, yrn, 99.5, 0.05, -0.02, _crit())["result"] == "FAIL"
    assert pe.verdict(exn, yrn, 99.5, 0.20, -0.02, _crit())["result"] == "FAIL"


def test_verdict_years_and_periods():
    ex = pd.Series([0.02] * 100 + [0.03] * 100 + [-0.01] * 100 + [0.01] * 100 + [-0.001] * 100)
    yr = pd.Series(np.repeat([2017, 2018, 2019, 2020, 2021], 100))
    v = pe.verdict(ex, yr, 99.5, 0.05, 0.001, _crit())
    assert v["pos_years"] == 3 and v["c4"]
    assert v["period_means"][0] == pytest.approx(0.04 / 3)
    assert v["period_means"][1] == pytest.approx(0.0045)
    assert v["c3"]


def test_in_zone_excludes_single_report_and_big_overlap():
    a, b = pd.Timestamp("2017-01-31"), pd.Timestamp("2019-12-31")
    ivs = [(a, b, "sp600", 4)]
    d = pd.Timestamp("2018-06-01")
    assert pe.in_zone(ivs, None, d) == (True, "sp600")
    assert pe.in_zone([(a, b, "sp600", 1)], None, d)[0] is False
    big = (pd.Timestamp("2018-09-01"), pd.Timestamp("2020-01-01"))  # 92일 안
    assert pe.in_zone(ivs, big, d)[0] is False
    assert pe.in_zone(ivs, (pd.Timestamp("2019-01-01"), pd.Timestamp("2020-01-01")), d)[0] is True


def test_seal_check_stops():
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.to_datetime(["2021-12-31", "2022-01-03"]))
    with pytest.raises(SystemExit):
        pe.seal_check(df, "x")
    pe.seal_check(df.iloc[:1], "x")


def test_verify_cache_stops_on_mismatch(tmp_path, monkeypatch):
    f = tmp_path / "a.csv"
    f.write_text("x\n1\n", encoding="utf-8")
    monkeypatch.setattr(pe, "ROOT", tmp_path)
    good = {"files": [{"path": "a.csv", "sha256": pe.file_sha256(f)}], "prices": []}
    assert pe.verify_cache(good) == []
    with pytest.raises(SystemExit):
        pe.verify_cache({"files": [{"path": "a.csv", "sha256": "0" * 64}], "prices": []})
    with pytest.raises(SystemExit):
        pe.verify_cache({"files": [], "prices": [{"path": "none.csv", "sha256": "0" * 64}]})
