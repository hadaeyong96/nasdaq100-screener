"""구역 ③ 판정 실행기 — 합성 데이터만 (구역 ③ 가격·수익을 쓰지 않는다).

태그 없으면 멈춤, 신호일(월말 2021-11까지·20거래일 주기), 장부 시뮬레이션(같은 비중·끝 청산 세금·가격 끝난 종목 처분),
QQQ 보유, 곡선 자르기·연도별·낙폭, 합격 기준 6개.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import rs26_zone3_run as run


def test_refuses_without_tag(monkeypatch):
    monkeypatch.setitem(run.SPECS, "rs26", {**run.SPECS["rs26"], "tag": "no-such-tag-for-test"})
    with pytest.raises(SystemExit, match="태그"):
        run.verify_prereg("rs26")


def test_signal_days():
    m = run.monthly_signals()
    assert m[0] == pd.Timestamp("2012-01-31") and m[-1] == pd.Timestamp("2021-11-30") and len(m) == 119
    c = run.cycle_signals()
    assert c[0] == pd.Timestamp("2012-01-03") and c[-1] < run.END
    days = run.nyse_days()
    pos = [days.get_loc(d) for d in c]
    assert all(b - a == 20 for a, b in zip(pos, pos[1:]))


class FakeMat:
    def __init__(self, days, prices: dict[str, np.ndarray], divs: dict[str, np.ndarray] | None = None):
        self.days = days
        self.di = {d: i for i, d in enumerate(days)}
        self.tickers = sorted(prices)
        self.ti = {t: j for j, t in enumerate(self.tickers)}
        self.close = np.column_stack([prices[t] for t in self.tickers])
        self.div = np.column_stack([(divs or {}).get(t, np.zeros(len(days))) for t in self.tickers])
        self.last, self.last_close = {}, {}
        for t in self.tickers:
            ok = ~np.isnan(prices[t])
            self.last[t] = days[np.where(ok)[0][-1]]
            self.last_close[t] = float(prices[t][ok][-1])


DAYS = pd.bdate_range("2021-01-04", "2021-12-31")


def _flat(n=len(DAYS), px=100.0):
    return np.full(n, px)


def test_simulate_equal_weight_flat_prices_loses_only_costs():
    prices = {f"S{i}": _flat() for i in range(10)}
    sig = {DAYS[0]: [{"ticker": t, "score_pp": 1.0} for t in prices]}
    r = run.simulate(FakeMat(DAYS, prices), sig, cost_pct=0.1)
    assert r["curve"].iloc[0] == pytest.approx(run.INITIAL)
    # 매수·끝 청산 수수료 0.1%씩, 이익 없음 → 세금 0
    assert r["final"] == pytest.approx(run.INITIAL * (1 - 0.001) * (1 - 0.001) / (1 + 0.001) * (1 + 0.001), rel=2e-3)
    assert r["tax"] == 0 and r["rebalances"] == 1


def test_simulate_taxes_gain_at_end_and_year_end():
    up = np.linspace(100, 200, len(DAYS))
    prices = {f"S{i}": up.copy() for i in range(10)}
    sig = {DAYS[0]: [{"ticker": t, "score_pp": 1.0} for t in prices]}
    r = run.simulate(FakeMat(DAYS, prices), sig, cost_pct=0.0)
    gross = r["curve"].iloc[-1]
    assert r["final"] == pytest.approx(gross - (gross - run.INITIAL) * 0.22, rel=1e-3)


def test_ended_series_is_sold_at_last_close_on_next_rebalance():
    p = {f"S{i}": _flat() for i in range(10)}
    p["S0"] = _flat()
    p["S0"][50:] = np.nan  # 50일째부터 가격 없음(인수)
    p["S0"][:50] = 120.0
    p["N0"] = _flat()
    sig = {DAYS[0]: [{"ticker": f"S{i}", "score_pp": 1.0} for i in range(10)],
           DAYS[100]: [{"ticker": t, "score_pp": 1.0} for t in [f"S{i}" for i in range(1, 10)] + ["N0"]]}
    r = run.simulate(FakeMat(DAYS, p), sig, cost_pct=0.0)
    assert r["curve"].iloc[80] == pytest.approx(r["curve"].iloc[49])  # 끝난 뒤 마지막 종가로 평가
    assert r["rebalances"] == 2


def test_qqq_hold_buys_on_fill_day_and_collects_dividends():
    qc = pd.Series(np.full(len(DAYS), 100.0), index=DAYS)
    qd = pd.Series(0.0, index=DAYS)
    qd.iloc[10] = 1.0
    r = run.simulate_qqq(qc, qd, DAYS[0], cost_pct=0.0)
    assert r["curve"].iloc[0] == run.INITIAL
    assert r["final"] == pytest.approx(run.INITIAL + 1000 * 1.0 * 0.85)


def test_slice_yearly_mdd():
    idx = pd.bdate_range("2012-01-02", "2021-12-31")
    curve = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    assert run.slice_cagr(curve, idx[0], idx[-1]) == pytest.approx(run.cagr(100, 200, idx[0], idx[-1]))
    y = run.yearly(curve, idx[0])
    assert set(y) == set(range(2012, 2022)) and all(v > 0 for v in y.values())
    dd = curve.copy()
    dd.iloc[100:] = dd.iloc[100:] * 0.5
    assert run.mdd(dd) == pytest.approx(-50.0, abs=0.5)


def test_judge_criteria_and_percentile():
    idx = pd.bdate_range("2012-01-31", "2021-12-31")
    good = {"curve": pd.Series(np.geomspace(1e5, 4e5, len(idx)), index=idx), "final": 4e5}
    q = {"curve": pd.Series(np.geomspace(1e5, 3e5, len(idx)), index=idx), "final": 3e5}
    ew = {"curve": q["curve"], "final": 2.5e5}
    j = run.judge(good, q, ew, list(np.linspace(0, 10, 1000)), idx[0])
    assert j["pass"] and j["random_pct"] == 100.0 and j["years_beat"] == 10
    j2 = run.judge(good, q, ew, list(np.linspace(0, 30, 1000)), idx[0])
    assert not j2["pass"] and not j2["criteria"][2]["ok"]
