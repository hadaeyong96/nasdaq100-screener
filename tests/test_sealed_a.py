"""봉인 ⓐ 개봉(core/sealed_a.py) 테스트 — 네트워크 없음, 합성 데이터."""

import pandas as pd
import pytest

from core import l1b
from core import sealed_a as sa


def _days():
    return pd.DatetimeIndex(["2022-01-03", "2022-01-04", "2022-01-07", "2022-01-10"])


def test_dtb3_daily_rf_uses_previous_day_rate_and_calendar_days():
    dtb3 = pd.Series([3.6, 7.2, 36.0], index=pd.DatetimeIndex(["2021-12-31", "2022-01-04", "2022-01-07"]))
    rf = sa.dtb3_daily_rf(dtb3, _days())
    assert rf.iloc[0] == 0.0
    assert rf.iloc[1] == pytest.approx(0.036 / 360 * 1)  # 01-03까지 알려진 값 3.6 (01-04 값은 아직 안 씀)
    assert rf.iloc[2] == pytest.approx(0.072 / 360 * 3)  # 01-04 값, 3일
    assert rf.iloc[3] == pytest.approx(0.36 / 360 * 3)   # 01-07 값, 주말 포함 3일


def test_dtb3_daily_rf_no_future():
    dtb3 = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.DatetimeIndex(["2021-12-31", "2022-01-04", "2022-01-07", "2022-01-10"]))
    days = _days()
    full = sa.dtb3_daily_rf(dtb3, days)
    for k in range(2, len(days) + 1):
        cut = days[:k]
        part = sa.dtb3_daily_rf(dtb3[dtb3.index <= cut[-1]], cut)
        assert part.iloc[-1] == pytest.approx(full.loc[cut[-1]])


def test_check_seal_stops_without_truncating():
    s = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2026-09-30", "2026-10-01"]))
    with pytest.raises(sa.SealError):
        sa.check_seal(s, "x")
    ok = s.iloc[:1]
    assert sa.check_seal(ok, "x") is ok


def _r(p, m):
    return {"posttax": p, "mdd": m}


def test_b1_verdict_pass_and_each_failure():
    full = {"B0": _r(10.0, -35.0), "B1": _r(12.5, -36.0)}
    subs = {"a": {"B0": _r(-5.0, 0), "B1": _r(-4.0, 0)}, "b": {"B0": _r(20.0, 0), "B1": _r(21.0, 0)}}
    nb = {190: 12.0, 210: 13.4}
    v = sa.b1_verdict(full, subs, nb)
    assert v["pass"] and all(v["ok"].values())
    assert not sa.b1_verdict({"B0": _r(10.0, -35.0), "B1": _r(11.99, -36.0)}, subs, nb)["ok"][1]
    assert not sa.b1_verdict({"B0": _r(10.0, -35.0), "B1": _r(12.5, -37.01)}, subs, nb)["ok"][2]
    tie = {**subs, "b": {"B0": _r(20.0, 0), "B1": _r(20.0, 0)}}
    assert not sa.b1_verdict(full, tie, nb)["ok"][3]  # 같으면 아님
    assert not sa.b1_verdict(full, subs, {190: 11.4, 210: 12.5})["ok"][4]


def test_b1_signal_and_holding_no_future():
    """B1 목표·보유는 t일까지 자른 결과와 전체의 t일 값이 같다 (l1b 함수 재사용 확인)."""
    idx = pd.bdate_range("2021-01-01", periods=320)
    level = pd.Series([100 + (i % 37) - (i // 50) * 3 for i in range(320)], index=idx, dtype=float)
    full = l1b.held_after_close_daily(l1b.target_b1(level, 200))
    for t in (205, 250, 319):
        part = l1b.held_after_close_daily(l1b.target_b1(level.iloc[: t + 1], 200))
        assert part.iloc[-1] == full.iloc[t]


def test_window_report_loss_and_moves():
    idx = pd.bdate_range("2021-12-30", periods=6)
    v = pd.Series([100, 100, 90, 80, 85, 88], index=idx, dtype=float)
    held = pd.Series(["2x", "2x", "2x", "cash", "cash", "2x"], index=idx)
    b0 = pd.Series([100, 100, 95, 90, 92, 93], index=idx, dtype=float)
    w = sa.window_report(v, held, b0, idx[2], idx[5])
    assert w["loss_pct"] == pytest.approx(-12.0)
    assert w["mdd_pct"] == pytest.approx(-20.0)
    assert w["b0_loss_pct"] == pytest.approx(-7.0)
    assert [a for _, a in w["moves"]] == ["cash", "2x"]


# ── SW1 대형 (scripts/sealed_a_sw1.py, scripts/sealed_a_sw1_data.py) ──────────


def _S(ex=0.5, per=(0.1, 0.2), pos=3, n=250):
    return {"ex_after": ex, "per": list(per), "pos_years": pos, "n": n}


def test_sw1_verdict_names():
    from scripts import sealed_a_sw1 as s

    assert s.verdict(_S(), 97.5, 0.1, 0.10)["name"] == "PASS"
    assert s.verdict(_S(), 99.0, 0.1, 0.1001)["name"] == "WAIT"       # ⑦ 먼저
    assert s.verdict(_S(n=199), 99.0, 0.1, 0.05)["name"] == "판정 불가"  # ⑤
    assert s.verdict(_S(), 97.4, 0.1, 0.05)["name"] == "FAIL"           # ② 0.05/2
    assert s.verdict(_S(ex=-0.1), 99.0, 0.1, 0.05)["name"] == "FAIL"
    assert s.verdict(_S(per=(0.1, 0.0)), 99.0, 0.1, 0.05)["name"] == "FAIL"
    assert s.verdict(_S(pos=2), 99.0, 0.1, 0.05)["name"] == "FAIL"
    assert s.verdict(_S(), 99.0, 0.0, 0.05)["name"] == "FAIL"           # ⑥


def test_sw1_mean_without_top():
    from scripts import sealed_a_sw1 as s

    ex = pd.Series([10.0, 9.0, 8.0, 7.0, 6.0, 1.0, -1.0])
    assert s.mean_without_top(ex, 5) == pytest.approx(0.0)


def test_sw1_pools_use_previous_month_end_and_exclude_self():
    from scripts import sealed_a_sw1 as s

    me_prev, me_cur = pd.Timestamp("2022-03-31"), pd.Timestamp("2022-04-30")
    m = {"ndx": {me_prev: {"AAA", "BBB"}, me_cur: {"FUT"}}, "sp100": {me_prev: {"CCC", "NOPX"}}}
    cik_of = {"AAA": "1", "BBB": "2", "CCC": "3", "FUT": "9"}
    tr = pd.DataFrame({"buy_day": [pd.Timestamp("2022-04-14")], "ticker": ["1"]})
    assert s.pools_for(tr, m, cik_of) == [["2", "3"]]  # 이번 달 명단(FUT)·가격 없는 티커·자기 자신 제외


def test_sw1_price_judge_seal_window():
    from scripts import sealed_a_sw1_data as d

    f, l = pd.Timestamp("2021-12-31"), pd.Timestamp("2026-08-31")
    assert d.judge(f, l, pd.Timestamp("2021-01-04"), pd.Timestamp("2026-09-30")) == "맞음"
    assert d.judge(f, l, pd.Timestamp("2022-03-01"), pd.Timestamp("2026-09-30")) == "앞이 빔"
    assert d.judge(f, l, pd.Timestamp("2021-01-04"), pd.Timestamp("2025-01-02")) == "끝이 이름"
    assert d.judge(f, l, pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-30")) == "다른 회사 의심"
