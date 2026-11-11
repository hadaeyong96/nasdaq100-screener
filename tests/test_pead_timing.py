"""core/pead_timing.py — 실적급등 타이밍 탐색 순수 함수 (합성 데이터, 네트워크 없음)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import pead_timing as pt
from core import sw1

DAYS = pd.bdate_range("2020-01-01", periods=80)


def _px(close, open_=None, drop=()):
    c = pd.Series(close, index=DAYS[: len(close)], dtype=float)
    o = c * 0.99 if open_ is None else pd.Series(open_, index=c.index, dtype=float)
    d = pd.DataFrame({"open": o, "close": c})
    return d.drop(index=[DAYS[i] for i in drop])


def _rand_px(seed, n=80, gaps=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.02, n))
    o = c * (1 + rng.normal(0, 0.005, n))
    drop = sorted(rng.choice(np.arange(5, n), size=gaps, replace=False)) if gaps else ()
    return _px(c, o, drop)


def test_align_lastv_points_to_last_close_at_or_before():
    a = pt.align(_px([10, 11, 12, 13, 14], drop=(2, 3)), DAYS[:6])
    assert list(a.lastv) == [0, 1, 1, 1, 4, 4]
    assert np.isnan(a.close[2]) and a.close[4] == 14


@pytest.mark.parametrize("mode,ib,use_open", [("B1", 6, True), ("B2", 6, False), ("B3", 8, False), ("B4", 10, False)])
def test_fixed_buy_modes(mode, ib, use_open):
    a = pt.align(_px(np.arange(100, 130)), DAYS[:30])
    got, why = pt.buy_point(a, 5, mode)
    assert (got, why) == (ib, "")
    assert pt.uses_open(mode) is use_open


def test_b5_first_close_three_percent_below_reaction_close():
    close = [100] * 5 + [100, 99, 97.5, 96.9, 95, 90] + [95] * 20  # i0 = 5, 반응일 종가 100
    a = pt.align(_px(close), DAYS[: len(close)])
    assert pt.buy_point(a, 5, "B5") == (8, "")  # 96.9 ≤ 97 (97.5는 아님)


def test_b5_no_pullback_buys_tenth_day_and_skips_missing_rows():
    close = [100] * 5 + [100] + [99] * 10 + [99] * 10
    a = pt.align(_px(close, drop=(7,)), DAYS[: len(close)])  # 빠진 날은 안 닿음
    assert pt.buy_point(a, 5, "B5") == (15, "")
    b = pt.align(_px(close, drop=(15,)), DAYS[: len(close)])  # 10거래일째 행 없음
    assert pt.buy_point(b, 5, "B5") == (None, "가격 공백")


def test_buy_point_calendar_end_and_missing_price():
    a = pt.align(_px([100] * 8), DAYS[:8])
    assert pt.buy_point(a, 5, "B4") == (None, "봉인 경계")
    assert pt.buy_point(a, 5, "B5") == (None, "봉인 경계")
    b = pt.align(_px([100] * 10, drop=(6,)), DAYS[:10])
    assert pt.buy_point(b, 5, "B2") == (None, "가격 공백")


def test_buy_point_does_not_use_prices_after_buy_day():
    """미래 데이터 방지: 매수일 뒤 가격을 지워도 매수 위치가 같다."""
    full = _rand_px(3)
    a_full = pt.align(full, DAYS)
    for mode in pt.BUY_MODES:
        ib, _ = pt.buy_point(a_full, 10, mode)
        cut = pt.align(full.loc[: DAYS[ib]], DAYS)
        assert pt.buy_point(cut, 10, mode)[0] == ib


@pytest.mark.parametrize("seed", range(5))
def test_exit_point_matches_sw1_realized_exit(seed):
    px = _rand_px(seed, gaps=15)
    a = pt.align(px, DAYS)
    for ib in range(0, 60, 7):
        if not np.isfinite(a.close[ib]):
            continue
        for j in (ib + 5, ib + 19):
            ex = sw1.realized_exit(px["close"], DAYS[ib], DAYS[j])
            e = pt.exit_point(a, ib, j)
            assert (DAYS[e], a.close[e]) == (ex[0], ex[1])


def test_b1_trades_match_sw1_make_trades():
    pxs = {f"T{i}": _rand_px(10 + i, gaps=4) for i in range(4)}
    sig = pd.DataFrame({"ticker": ["T0", "T1", "T0", "T2", "T3", "T1"],
                        "d0": [DAYS[3], DAYS[5], DAYS[12], DAYS[20], DAYS[40], DAYS[70]]})
    ref, ref_sk = sw1.make_trades(sig, pxs, DAYS, 15)
    got, got_sk = pt.make_trades(sig, {t: pt.align(p, DAYS) for t, p in pxs.items()}, DAYS, "B1", 15)
    assert list(got["buy_day"]) == list(ref["buy_day"])
    assert list(got["sell_day"]) == list(ref["sell_day"])
    assert np.allclose(got["buy_px"], ref["buy_open"]) and np.allclose(got["sell_px"], ref["sell_close"])
    assert sorted(got_sk["skip"]) == sorted(ref_sk["skip"])


def test_overlap_ignores_signal_while_waiting_to_buy():
    px = {"A": pt.align(_px(np.full(60, 100.0)), DAYS[:60])}
    sig = pd.DataFrame({"ticker": ["A", "A", "A"], "d0": [DAYS[5], DAYS[8], DAYS[40]]})
    tr, sk = pt.make_trades(sig, px, DAYS[:60], "B4", 20)  # 첫 매수 위치 10, 매도 30 → d0 8은 대기 중
    assert list(tr["d0"]) == [DAYS[5]]
    assert list(sk["skip"]) == ["보유 중", "봉인 경계"]


def test_price_trades_b1_matches_sw1_open_legs_and_close_legs_for_others():
    pxs = {"A": _rand_px(1), "Q": _rand_px(2)}
    al = {t: pt.align(p, DAYS) for t, p in pxs.items()}
    sig = pd.DataFrame({"ticker": ["A", "A"], "d0": [DAYS[2], DAYS[40]]})
    fx = np.full(len(DAYS), 1200.0)
    for mode in ("B1", "B3"):
        tr, _ = pt.make_trades(sig, al, DAYS, mode, 20)
        out = pt.price_trades(tr, al["Q"], fx, mode, sw1.Costs(), 0.22)
        for _, r in out.iterrows():
            q = pxs["Q"]
            b = q.at[r["buy_day"], "open" if mode == "B1" else "close"]
            assert r["bench_usd"] == pytest.approx(q.at[r["sell_day"], "close"] / b - 1)
        exp = out["ret_krw"] - sw1.ledger_tax(out["ret_krw"].to_numpy(), pd.DatetimeIndex(out["sell_day"]).year.to_numpy())
        assert np.allclose(out["ret_at"], exp)


def test_candidates_match_brute_force():
    pxs = {f"S{i}": _rand_px(30 + i, gaps=10) for i in range(6)}
    al = {t: pt.align(p, DAYS) for t, p in pxs.items()}
    names = list(al)
    O = np.vstack([al[t].open for t in names])
    C = np.vstack([al[t].close for t in names])
    L = np.vstack([al[t].lastv for t in names])
    bench = pt.align(_rand_px(99), DAYS)
    fx = np.linspace(1100, 1300, len(DAYS))
    years = DAYS.year.to_numpy()
    tr = pd.DataFrame({"ib": [10, 30], "plan": [40, 70]})
    pools = [np.arange(6), np.array([0, 2, 4])]
    for mode in ("B1", "B4"):
        got = pt.candidates(tr, pools, O, C, L, bench, fx, years, mode, sw1.Costs())
        for k, (ib, j) in enumerate(zip(tr["ib"], tr["plan"])):
            exp = []
            for i in pools[k]:
                a = al[names[i]]
                bp = (a.open if mode == "B1" else a.close)[ib]
                e = pt.exit_point(a, ib, j)
                if np.isfinite(bp) and bp > 0 and e is not None:
                    exp.append(sw1.krw_after_cost(a.close[e] / bp - 1, fx[ib], fx[e]))
            assert np.allclose(got[k]["ret"], exp)


def test_cumulative_excess_simple_and_carries_last_close():
    n = 20
    days = DAYS[:n]
    q = np.full(n, 100.0)
    a = pt.align(_px(np.r_[100, 100, np.full(n - 2, 110.0)], drop=(10, 11)), days)
    curve, cnt = pt.cumulative_excess(pd.DataFrame({"ticker": ["A"], "d0": [days[1]]}), {"A": a}, q, days, horizon=15)
    assert cnt == 1 and np.allclose(curve, 0.10)
    _, cnt2 = pt.cumulative_excess(pd.DataFrame({"ticker": ["A"], "d0": [days[10]]}), {"A": a}, q, days, horizon=15)
    assert cnt2 == 0  # i0 + 15가 달력 밖


def _grid(fn):
    return pd.DataFrame([{"mode": m, "hold": h, **fn(m, h)} for m in pt.BUY_MODES for h in pt.HOLDS])


def test_select_cell_highest_percentile_with_positive_periods_and_neighbors():
    def f(m, h):
        base = {"mean": 0.5, "p1": 0.5, "p2": 0.5, "pct": 80.0}
        if (m, h) == ("B3", 90):
            base["pct"] = 99.0
        if (m, h) == ("B2", 40):
            base["pct"] = 99.5
        if (m, h) == ("B2", 60):
            base["mean"] = -0.1  # B2·40의 이웃이 음수 → B2·40 탈락
        return base

    cell, notes = pt.select_cell(_grid(f))
    assert cell == ("B3", 90)
    assert next(x for x in notes if (x["mode"], x["hold"]) == ("B2", 40))["neighbors"] is False


def test_select_cell_none_when_no_cell_qualifies():
    cell, _ = pt.select_cell(_grid(lambda m, h: {"mean": 0.3, "p1": 0.4, "p2": -0.1, "pct": 90.0}))
    assert cell is None


def test_select_cell_tie_breaks_on_mean_then_shorter_hold():
    def f(m, h):
        return {"mean": 0.9 if (m, h) in (("B1", 40), ("B1", 90)) else 0.2, "p1": 1, "p2": 1, "pct": 95.0}

    assert pt.select_cell(_grid(f))[0] == ("B1", 40)
