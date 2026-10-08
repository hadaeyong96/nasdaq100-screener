"""L1b 신호·봉인·합성 2배·장부(세금) 테스트 (네트워크 없음, 합성 데이터)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import l1b
from data import l1b_history as hist


def _days(start="1960-01-04", periods=900):
    return pd.bdate_range(start, periods=periods)


def _level(td, seed=1, vol=0.015):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, vol, len(td)))), index=td)


# ── 미래 데이터 방지 ───────────────────────────────────────────────────────


@pytest.mark.parametrize("cut", [260, 450, 700])
@pytest.mark.parametrize("cand,params", [
    ("B1", {"sma_days": 200}), ("B2", {"sma_days": 200, "crash_pct": -20.0}),
    ("B3", {"sma_days": 190, "confirm_days": 3}), ("B0", {}),
])
def test_daily_signal_truncated_equals_full(cand, params, cut):
    td = _days()
    level = _level(td, vol=0.03)  # 변동을 키워 B2 표시가 실제로 켜지게
    t = td[cut]
    full = l1b.daily_target(cand, level, **params)
    trunc = l1b.daily_target(cand, level.loc[:t], **params)
    pd.testing.assert_series_equal(full.loc[:t], trunc)
    pd.testing.assert_series_equal(l1b.held_after_close_daily(full).loc[:t], l1b.held_after_close_daily(trunc))


@pytest.mark.parametrize("cut", [12, 30, 55])
def test_monthly_signal_truncated_equals_full(cut):
    me = l1b.month_end_levels(_level(_days(periods=1500), seed=3))
    t = me.index[cut]
    for n in (9, 10, 11):
        pd.testing.assert_series_equal(l1b.target_monthly(me, n).loc[:t], l1b.target_monthly(me.loc[:t], n))


def test_no_negative_shift_in_l1b_sources():
    import inspect
    import re

    from scripts import l1b_experiments

    for mod in (l1b, hist, l1b_experiments):
        assert not re.search(r"shift\(\s*-", inspect.getsource(mod)), mod.__name__


# ── 신호 규칙 ─────────────────────────────────────────────────────────────


def test_above_sma_equal_is_below_and_warmup_nan():
    td = _days(periods=5)
    out = l1b.above_sma(pd.Series([1.0, 1.0, 1.0, 2.0, 1.0], index=td), 3)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    assert list(out.iloc[2:]) == [0.0, 1.0, 0.0]


def test_crash_flag_on_and_off():
    td = _days(periods=8)
    # 고점 100 → 75(−25%) → 69(−31%, 켬) → 90 → 99.9(아직) → 100(신고가 회복 = 끔) → 80(−20%, 꺼진 채)
    lv = pd.Series([90, 100, 75, 69, 90, 99.9, 100, 80], index=td, dtype=float)
    flag = l1b.crash_flag(lv, -30.0)
    assert list(flag) == [False, False, False, True, True, True, False, False]
    assert list(l1b.crash_flag(lv, -25.0))[:3] == [False, False, True]  # 같으면 포함


def test_b2_targets_follow_flag_and_sma():
    td = _days(periods=8)
    lv = pd.Series([100, 100, 60, 65, 70, 69, 120, 110], index=td, dtype=float)
    out = l1b.target_b2(lv, sma_days=2, crash_pct=-30.0)
    # 표시: 60에서 켬, 120(신고가)에서 끔. SMA2 위: 65>62.5, 70>67.5 → 2배, 69<69.5 → 1배
    assert list(out) == ["1x", "1x", "1x", "2x", "2x", "1x", "1x", "1x"]


def test_b3_confirmation_days():
    td = _days(periods=10)
    above = [1, 1, 0, 0, 1, 0, 0, 0, 1, 1]
    # sma_days=1이면 above_sma는 항상 "같음=아래"라 쓸 수 없어, 직접 위/아래가 되는 지수를 만든다
    lv = pd.Series(1.0, index=td)
    vals, prev = [], 100.0
    for a in above:
        prev = prev * (1.10 if a else 0.90)
        vals.append(prev)
    lv = pd.Series(vals, index=td)
    side = l1b.above_sma(lv, 2)
    assert list(side.iloc[1:]) == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    st = l1b.confirmed_side(lv, 2, 3)
    # 첫 산출일(1번 행) 위 → 2·3번 아래 2행뿐(유지) → 5~7번 아래 3행 연속 → 7번에서 아래로
    assert list(st.iloc[1:]) == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    pd.testing.assert_series_equal(l1b.target_b3(lv, 2, 1), l1b.target_b1(lv, 2))


def test_monthly_signal_rule():
    me = pd.Series([1, 2, 3, 2, 1.5, 3], index=pd.date_range("2000-01-31", periods=6, freq="ME"), dtype=float)
    out = l1b.target_monthly(me, 3, defense="hidiv")
    assert out.iloc[:2].isna().all()
    assert list(out.iloc[2:]) == ["2x", "hidiv", "hidiv", "2x"]


# ── 봉인 ─────────────────────────────────────────────────────────────────


def test_seal_raises_on_1999_rows():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["1998-12-31", "1999-01-04"]))
    with pytest.raises(l1b.SealError):
        l1b.seal(s, "x")
    assert len(l1b.cut_and_seal(s, "x")) == 1
    df = pd.DataFrame({"a": [1.0]}, index=pd.to_datetime(["1999-01-01"]))
    with pytest.raises(l1b.SealError):
        l1b.seal(df, "df")


def test_french_parsers_cut_and_missing():
    text = "header\n\n,Mkt-RF,SMB,HML,RF\n19981231,    1.00,0,0,    0.02\n19990104,    2.00,0,0,    0.02\n\nCopyright\n"
    df = hist.parse_french_daily(text)
    assert list(df.index.year) == [1998, 1999]
    assert df["mkt_rf"].iloc[0] == pytest.approx(0.01)
    with pytest.raises(l1b.SealError):
        l1b.seal(df)
    dp = ("x\n  Value Weight Returns -- Monthly\n,<= 0,Lo 30,Med 40,Hi 30\n"
          "192707, 1, 2, 3, 5.94\n192708, 1, 2, 3, -99.99\n\n  Equal Weight Returns -- Monthly\n")
    hi = hist.parse_french_dp_hi30(dp)
    assert hi.iloc[0] == pytest.approx(0.0594) and np.isnan(hi.iloc[1])


# ── 합성 2배·정렬 ───────────────────────────────────────────────────────────


def test_synthetic_2x():
    td = _days(periods=3)
    r = pd.Series([0.01, -0.02, -0.60], index=td)
    rf = pd.Series([0.0001, 0.0002, 0.0], index=td)
    out = l1b.synthetic_2x_returns(r, rf)
    fee = 0.0095 / 252
    assert out.iloc[0] == pytest.approx(0.02 - 0.0001 - fee)
    assert out.iloc[1] == pytest.approx(-0.04 - 0.0002 - fee)
    assert out.iloc[2] == -1.0  # −100% 아래는 자른다
    m = l1b.compound_by_month(pd.Series([0.1, 0.1, 0.05], index=pd.to_datetime(["2000-01-03", "2000-01-31", "2000-02-01"])))
    assert m.iloc[0] == pytest.approx(0.21) and m.index[0] == pd.Timestamp("2000-01-31")


def test_align_rf_compounds_skipped_days():
    rf = pd.Series([0.01, 0.02, 0.03], index=pd.to_datetime(["2000-01-03", "2000-01-04", "2000-01-05"]))
    out = l1b.align_rf(rf, pd.to_datetime(["2000-01-03", "2000-01-05"]))
    assert out.iloc[0] == pytest.approx(0.01)
    assert out.iloc[1] == pytest.approx(1.02 * 1.03 - 1)


# ── 장부·세금 ─────────────────────────────────────────────────────────────


def _rets(idx, r1, cash=0.0):
    return pd.DataFrame({"1x": r1, "2x": [2 * x for x in r1], "cash": [cash] * len(idx)}, index=idx)


def test_buy_and_hold_tax_only_at_liquidation():
    idx = pd.to_datetime(["1990-12-28", "1990-12-31", "1991-12-31"])
    held = pd.Series("1x", index=idx)
    sim = l1b.simulate(held, _rets(idx, [0.0, 0.10, 0.10]))
    c = 0.001
    v0 = 100_000 / (1 + c)
    v_end = v0 * 1.1 * 1.1
    sale = v_end * (1 - c)
    assert sim["taxes"][1990] == 0.0  # 실현 없음 → 연말 세금 0
    assert sim["after_tax_final"] == pytest.approx(sale - 0.22 * (sale - 100_000))
    assert sim["switches"] == 0


def test_yearend_tax_no_carryforward_and_next_year_booking():
    # 1990: 2배에서 +50% 뒤 국채로 전환(실현이익) → 연말 22% 납부
    # 1991: 1배로 사서 −50% 손실 실현 → 세금 0, 손실은 1992로 넘어가지 않는다
    idx = pd.to_datetime(["1990-01-02", "1990-06-01", "1990-12-31", "1991-03-01", "1991-12-31", "1992-06-01", "1992-12-31"])
    held = pd.Series(["2x", "cash", "cash", "1x", "cash", "1x", "1x"], index=idx)
    r1 = [0.0, 0.25, 0.0, 0.0, -0.5, 0.0, 0.2]
    sim = l1b.simulate(held, _rets(idx, r1))
    c = 0.001
    v = 100_000 / (1 + c) * 1.5
    sale = v * (1 - c)
    gain90 = sale - 100_000
    assert sim["taxes"][1990] == pytest.approx(0.22 * gain90)
    assert sim["taxes"][1991] == 0.0
    # 1992 마지막 청산 이익은 1991 손실과 상계하지 않고 그대로 과세된다
    assert sim["taxes"][1992] > 0
    assert sim["switches"] == 4


def test_cash_interest_taxed_and_pretax():
    idx = pd.to_datetime(["1990-12-28", "1990-12-31", "1991-12-31"])
    held = pd.Series("cash", index=idx)
    sim = l1b.simulate(held, _rets(idx, [0, 0, 0], cash=0.05))
    v0 = 100_000 / 1.001
    assert sim["taxes"][1990] == pytest.approx(0.22 * v0 * 0.05)
    pre = l1b.simulate(held, _rets(idx, [0, 0, 0], cash=0.05), apply_tax=False)
    assert pre["pretax_final"] == pytest.approx(v0 * 1.05 * 1.05)
    assert pre["taxes"] == {}


def test_daily_execution_is_next_day_close():
    td = _days(periods=5)
    target = pd.Series(["1x", "2x", "2x", "cash", "cash"], index=td)
    held = l1b.held_after_close_daily(target)
    assert list(held.iloc[1:]) == ["1x", "2x", "2x", "cash"]
    sim = l1b.simulate(held.iloc[1:], pd.DataFrame({"1x": 0.0, "2x": [0, 0.1, 0.1, 0.1, 0.1], "cash": 0.0}, index=td), apply_tax=False)
    # td1 종가 2배 신호 → td2 종가에 2배로 바꿈 → td3·td4 수익만 받음(td2 수익은 1배)
    # td3 종가 국채 신호 → td4 종가에 국채로 바꿈(td4 수익은 아직 2배)
    assert sim["values"].iloc[-1] == pytest.approx(100_000 / 1.001 * 0.999 / 1.001 * 1.1 * 1.1 * 0.999 / 1.001)


def test_metrics():
    v = pd.Series([100, 120, 60, 90, 130, 125], index=_days(periods=6), dtype=float)
    assert l1b.max_drawdown(v)["mdd_pct"] == pytest.approx(-50.0)
    rec = l1b.longest_recovery(v)
    assert rec["rows"] == 3 and not rec["open_ended"]
