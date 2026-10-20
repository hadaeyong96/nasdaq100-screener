"""3단 거르기 탐색 — 지표·국면·장부 규칙 (합성 데이터, 네트워크 없음)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import tax_harvest as th
from core import triple_screen as ts


def _px(n=400, seed=1):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2011-01-03", periods=n)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n))), index=days)
    h = c * (1 + rng.uniform(0, 0.02, n))
    lo = c * (1 - rng.uniform(0, 0.02, n))
    return days, c, h, lo


# ── 지표 ────────────────────────────────────────────────────────────────


def test_indicators_no_future_data():
    days, c, h, lo = _px()
    full = {"rsi": ts.rsi_wilder(c), "stk": ts.slow_stoch_k(h, lo, c), "s200": ts.sma(c, 200)}
    wl, ws = ts.weekly_macd(c, days)
    cw = ts.completed_week_index(days)
    for t in (250, 300, 399):
        cut = {"rsi": ts.rsi_wilder(c[:t + 1]), "stk": ts.slow_stoch_k(h[:t + 1], lo[:t + 1], c[:t + 1]),
               "s200": ts.sma(c[:t + 1], 200)}
        for k in full:
            assert cut[k].iloc[-1] == pytest.approx(full[k].iloc[t])
        # 거래 달력은 미리 정해진 것(전체 days) — t일 뒤 가격만 지운다. 그날의 '직전 완료 주' 값은 같아야 한다
        wl2, ws2 = ts.weekly_macd(c[:t + 1], days)
        w = cw[t]
        assert wl2[w] == pytest.approx(wl[w]) and ws2[w] == pytest.approx(ws[w])


def test_week_ends_with_friday_holiday():
    days = pd.DatetimeIndex(["2021-03-29", "2021-03-30", "2021-03-31", "2021-04-01",  # 4/2 성금요일 휴장
                             "2021-04-05", "2021-04-06", "2021-04-09"])
    assert list(ts.week_ends(days)) == [False, False, False, True, False, False, True]
    assert list(ts.completed_week_index(days)) == [-1, -1, -1, 0, 0, 0, 1]


def test_weekly_macd_uses_week_last_close():
    days = pd.bdate_range("2020-01-06", periods=10)  # 2주
    c = pd.Series(np.arange(10, dtype=float) + 1, index=days)
    wl, _ = ts.weekly_macd(c, days)
    exp_line, _ = ts.macd(pd.Series([5.0, 10.0]))
    assert wl == pytest.approx(exp_line.to_numpy())


def test_cross_up_and_stoch_range():
    x = pd.Series([45, 39, 38, 40, 41, 39, 40.0])
    assert list(ts.cross_up(x, 40)) == [False, False, False, True, False, False, True]
    days, c, h, lo = _px()
    k = ts.slow_stoch_k(h, lo, c).dropna()
    assert (k >= 0).all() and (k <= 100).all()


def test_rsi_matches_core_indicators_formula():
    from core.indicators import compute_indicators

    days, c, h, lo = _px(300)
    df = pd.DataFrame({"open": c, "high": h, "low": lo, "close": c, "volume": 1.0})
    cfg = {"indicators": {"macd": {"fast": 12, "slow": 26, "signal": 9}, "rsi": {"period": 14}, "ichimoku": {"tenkan": 9, "kijun": 26, "senkou_b": 52},
                          "ichimoku_shift": 26, "volume_ma": {"period": 20}, "swing_low": {"period": 20},
                          "bollinger": {"period": 20, "std_mult": 2, "percentile_period": 120}}}
    ref = compute_indicators(df, cfg)
    assert ts.rsi_wilder(c).to_numpy() == pytest.approx(ref["rsi"].to_numpy(), nan_ok=True)
    line, sig = ts.macd(c)
    assert line.to_numpy() == pytest.approx(ref["macd"].to_numpy()) and sig.to_numpy() == pytest.approx(ref["signal"].to_numpy())


# ── 국면 ────────────────────────────────────────────────────────────────


def test_classify_priority_and_recovery():
    n = 400
    days = pd.bdate_range("2010-01-04", periods=n)
    q = pd.Series(np.r_[np.linspace(100, 200, 300), np.linspace(200, 150, 60), np.linspace(150, 185, 40)], index=days)
    xly = pd.Series(np.r_[np.full(360, 1.0), np.linspace(1.0, 1.3, 40)], index=days)
    xlp = pd.Series(1.0, index=days)
    r = ts.classify(q, xly, xlp, mid=100)
    assert (r.iloc[:199] == ts.UNKNOWN).all()
    assert r.iloc[250] == ts.UP
    assert ts.CRASH in set(r.iloc[300:360])  # 고점 대비 −20% 이하 → 폭락장이 먼저
    r0 = ts.classify(q, xly, xlp, mid=100, use_ratio=False)
    assert set(r0) >= {ts.UP, ts.CRASH}
    for t in (250, 330, 399):  # 미래 데이터 방지
        assert ts.classify(q[:t + 1], xly[:t + 1], xlp[:t + 1]).iloc[-1] == r.iloc[t]


def test_recovery_rule_needs_all_conditions():
    days = pd.bdate_range("2010-01-04", periods=330)
    # 길게 오르다(200일선 위) 천천히 내려 200일선 아래·100일선 위·100일선 상승을 만든다(폭락장 조건은 피함)
    q = pd.Series(np.r_[np.linspace(100, 300, 300), np.linspace(300, 262, 30)], index=days)
    xly = pd.Series(np.r_[np.full(300, 1.0), np.linspace(1.0, 1.2, 30)], index=days)
    xlp = pd.Series(1.0, index=days)
    s200, s100 = ts.sma(q, 200), ts.sma(q, 100)
    r = ts.classify(q, xly, xlp)
    want = (q < s200) & (q > s100) & (s100 > s100.shift(20)) & ((xly / xlp) > ts.sma(xly / xlp, 50))
    base = ts.rg.classify(q)
    want &= base != ts.rg.CRASH
    assert ((r == ts.REC) == want).all()
    flat = pd.Series(1.0, index=days)
    assert not (ts.classify(q, flat, xlp) == ts.REC).any()  # 비율이 평균보다 높지 않으면 회복장 아님


def test_recovery_episodes():
    r = pd.Series([ts.DOWN, ts.REC, ts.REC, ts.UP, ts.DOWN, ts.REC, ts.DOWN, ts.REC])
    e = ts.recovery_episodes(r)
    assert e == {"episodes": 3, "to_up": 1, "undecided": 1, "to_up_pct": 50.0}


# ── 장부 ────────────────────────────────────────────────────────────────


def _market(D=60, names=("AAA", "BBB")):
    days = pd.bdate_range("2013-01-07", periods=D)  # 월요일 시작
    C = len(names) + 1
    op = np.full((D, C), 100.0)
    cl = np.full((D, C), 100.0)
    mk = ts.Market(days, list(names) + [ts.IDLE], op, cl.copy(), cl.copy(), cl, np.ones((D, C)), np.zeros((D, C)),
                   np.c_[np.ones((D, C - 1), bool), np.zeros((D, 1), bool)],
                   pd.DataFrame({"open": 100.0, "close": 100.0, "div": 0.0}, index=days), np.zeros(D), np.full(D, 1000.0))
    z = lambda v=False: np.full((D, C), v)  # noqa: E731
    mk.ind = {"s200": np.full((D, C), 50.0), "s100": np.full((D, C), 50.0), "adv20": np.full((D, C), 1.0),
              "trig_rsi": z(), "trig_stoch": z(), "tide": z(True), "below_now": z(), "above_now": z(True),
              "cross_down": z(), "week_end": ts.week_ends(days)}
    mk.reg = {(100, True): np.full(D, ts.UP, dtype=object)}
    return mk


def _cal(mk):
    return ts.calendar(mk.days, th.deadline)


def _run(mk, v=ts.A1, **kw):
    return ts.run_strategy(mk, v, ts.ZERO, _cal(mk), **kw)


def test_buy_next_open_ten_percent_and_idle_qqq():
    mk = _market()
    mk.ind["trig_rsi"][5, 0] = True
    mk.open[6, 0] = 80.0  # 다음 날 시가로 체결
    r = _run(mk)
    b = r["book"]
    h = b.hold.get(0)
    assert h is None or h["fill"] == 6
    t = [x for x in b.trades if x["col"] == 0] or [{"fill": h["fill"]}]
    assert t[0]["fill"] == 6
    # 첫날 종가 국면 상승장 → 둘째 날 시가에 QQQ, 종목 10%만큼 QQQ를 팔아 마련
    assert r["buys_per_day"] == {5: 1}
    assert b.curve_stock[6] == pytest.approx(0.10 * b.curve_pre[5] / 1000 * 100 / 80, rel=1e-6)


def test_exit_cross_down_only_after_buy():
    mk = _market()
    mk.ind["trig_rsi"][5, 0] = True  # 월요일 기준 5 = 다음 주 월
    mk.ind["cross_down"][4, 0] = True  # 매수 전 주의 하향 교차는 사유 아님
    fri = [i for i in range(len(mk.days)) if mk.ind["week_end"][i] and i >= 6]
    mk.ind["cross_down"][fri[1], 0] = True
    b = _run(mk)["book"]
    t = [x for x in b.trades if x["col"] == 0][0]
    assert t["exit"] == fri[1] + 1 and t["why"].startswith("①")


def test_exit_supplement_four_weeks_without_recovery():
    mk = _market()
    mk.ind["trig_rsi"][5, 0] = True
    mk.ind["below_now"][5, 0] = True
    mk.ind["above_now"][:, 0] = False
    fri = [i for i in range(len(mk.days)) if mk.ind["week_end"][i] and i >= 6]
    b = _run(mk)["book"]
    t = [x for x in b.trades if x["col"] == 0][0]
    assert t["exit"] == fri[3] + 1 and t["why"].startswith("①-보완")


def test_supplement_cleared_when_line_recovers():
    mk = _market()
    mk.ind["trig_rsi"][5, 0] = True
    mk.ind["below_now"][5, 0] = True
    mk.ind["above_now"][:, 0] = False
    fri = [i for i in range(len(mk.days)) if mk.ind["week_end"][i] and i >= 6]
    mk.ind["above_now"][fri[1], 0] = True  # 2번째 완료 주에 MACD선 ≥ 시그널선
    b = _run(mk)["book"]
    t = [x for x in b.trades if x["col"] == 0][0]
    assert t["why"] == "끝 청산"


def test_exit_below_200_and_regime():
    mk = _market()
    mk.ind["trig_rsi"][5, 0] = True
    mk.close[10, 0] = 40.0  # 종가 < 200일선(50)
    b = _run(mk)["book"]
    t = [x for x in b.trades if x["col"] == 0][0]
    assert t["exit"] == 11 and t["why"].startswith("②")
    mk = _market()
    mk.ind["trig_rsi"][5, 0] = True
    mk.reg[(100, True)][20:] = ts.DOWN
    r = _run(mk)
    b = r["book"]
    t = [x for x in b.trades if x["col"] == 0][0]
    assert t["exit"] == 21 and t["why"] == "③ 국면"
    q = len(mk.names) - 1
    assert q not in b.hold and b.curve_stock[30] == 0.0  # 상승장이 아니면 빈자리 돈은 BIL(현금)


def test_rank_by_dollar_volume_when_slots_short():
    mk = _market(names=("AAA", "BBB", "CCC"))
    mk.ind["trig_rsi"][5, :3] = True
    mk.ind["adv20"][5, :3] = [1.0, 3.0, 2.0]
    v = ts.Variant("t", slots=2)
    r = _run(mk, v)
    held = {mk.names[x["col"]] for x in r["book"].trades}
    assert held == {"BBB", "CCC"} and r["stats"]["no_slot"] == 1


def test_random_same_count_from_stage2_pool():
    mk = _market(names=("AAA", "BBB", "CCC", "DDD"))
    mk.ind["tide"][:, 3] = False  # DDD는 2단 탈락 → 무작위 후보 아님
    for s in range(20):
        r = _run(mk, rng=np.random.default_rng(s), buys_per_day={5: 2})
        held = {mk.names[x["col"]] for x in r["book"].trades}
        assert len(held) == 2 and "DDD" not in held


def test_engine_no_future_data():
    mk = _market()
    mk.ind["trig_rsi"][5, 0] = True
    mk.ind["trig_rsi"][30, 1] = True
    a = _run(mk)["book"]
    mk2 = _market()
    mk2.ind["trig_rsi"][5, 0] = True
    mk2.ind["trig_rsi"][30, 1] = True
    mk2.close[41:, :] = 10.0  # 41일 뒤 가격을 바꿔도
    mk2.open[41:, :] = 10.0
    b = _run(mk2)["book"]
    assert a.curve_pre[:41] == pytest.approx(b.curve_pre[:41])


def test_costs_and_tax_reduce_value():
    mk = _market()
    mk.ind["trig_rsi"][5, 0] = True
    mk.close[20:, 0] = 150.0
    mk.open[20:, 0] = 150.0
    mk.ind["cross_down"][[i for i in range(len(mk.days)) if mk.ind["week_end"][i] and i >= 25][0], 0] = True
    z = ts.xirr(ts.run_strategy(mk, ts.A1, ts.ZERO, _cal(mk))["book"].flows)
    c = ts.xirr(ts.run_strategy(mk, ts.A1, ts.NO_TAX, _cal(mk))["book"].flows)
    f = ts.xirr(ts.run_strategy(mk, ts.A1, ts.FULL, _cal(mk))["book"].flows)
    assert z > c > f


def test_timing_compare_follows_regime():
    mk = _market()
    mk.reg[(100, True)][30:] = ts.DOWN
    r = ts.run_timing(mk, {ts.UP: 1.0}, mk.reg[(100, True)], ts.ZERO, _cal(mk))
    b = r["book"]
    assert b.curve_pre[10] == pytest.approx(b.curve_pre[0])  # 가격 고정 → 가치 같음
    assert 0 not in b.hold  # 하락장 → BIL


def test_compare_books_run_and_keep_value_on_flat_prices():
    mk = _market()
    cal = _cal(mk)
    for res in (ts.run_qqq_hold(mk, ts.ZERO, cal), ts.run_equal_weight(mk, ts.ZERO, cal),
                ts.run_timing(mk, {ts.UP: 1.0, ts.REC: 0.3}, mk.reg[(100, True)], ts.ZERO, cal)):
        assert res["final"] == pytest.approx(ts.INITIAL_KRW)  # 가격·환율 고정, 비용 0 → 원금 그대로
    ew = ts.run_equal_weight(mk, ts.ZERO, cal)["book"]
    assert len(mk.names) - 1 not in ew.hold  # QQQ 열은 대상 아님
