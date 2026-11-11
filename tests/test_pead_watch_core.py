"""core/pead_watch.py — 실적급등 추천 장부 순수 함수 (합성 데이터, 네트워크 없음)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import pead_timing as pt
from core import pead_watch as pw
from scripts import sw1_data_check as dc

DAYS = pd.bdate_range("2026-01-01", periods=200)


def _px(seed, n=200, gaps=0, jump_at=None):
    rng = np.random.default_rng(seed)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.015, n))
    v = rng.uniform(1e6, 2e6, n)
    if jump_at is not None:
        c[jump_at:] *= 1.12
        v[jump_at] *= 4
    o = c * (1 + rng.normal(0, 0.004, n))
    d = pd.DataFrame({"open": o, "close": c, "volume": v}, index=DAYS[:n])
    if gaps:
        drop = rng.choice(np.arange(30, n), size=gaps, replace=False)
        d = d.drop(index=DAYS[sorted(drop)])
    return d


# ── 발표·반응일 ─────────────────────────────────────────────────────────────


def test_is_8k_202_excludes_amendments_and_other_items():
    assert pw.is_8k_202("8-K", "2.02,9.01")
    assert pw.is_8k_202("8-K", "7.01, 2.02")
    assert not pw.is_8k_202("8-K/A", "2.02")
    assert not pw.is_8k_202("8-K", "5.02,9.01")
    assert not pw.is_8k_202("10-Q", "")


def test_counted_announcements_60_day_gap_from_last_counted():
    f = [{"form": "8-K", "items": "2.02", "filing_date": d, "accn": a} for d, a in
         [("2026-01-28", "a"), ("2026-02-20", "b"), ("2026-03-29", "c"), ("2026-04-30", "d"), ("2026-07-30", "e")]]
    f.append({"form": "8-K/A", "items": "2.02", "filing_date": "2026-10-30", "accn": "x"})
    got = [x["accn"] for x in pw.counted_announcements(f, 60)]
    assert got == ["a", "c", "e"]  # b: 23일, c: 60일(a부터), d: 32일(c부터), e: 91일


@pytest.mark.parametrize("acc,expected", [
    ("2026-02-03 08:15:00", "2026-02-03"),  # 장전 → 그날
    ("2026-02-03 09:30:00", "2026-02-04"),  # 09:30 정각은 장중
    ("2026-02-03 16:05:00", "2026-02-04"),  # 장후 → 다음 거래일
    ("2026-02-07 07:00:00", "2026-02-09"),  # 토요일 장전 → 월요일
    ("2026-02-06 17:00:00", "2026-02-09"),  # 금요일 장후 → 월요일
])
def test_reaction_day_matches_sw1(acc, expected):
    t = pd.Timestamp(acc)
    assert pw.reaction_day(t, DAYS) == pd.Timestamp(expected)
    assert dc.reaction_day(t, DAYS) == pd.Timestamp(expected)


def test_membership_label_and_prev_month_end():
    assert pw.membership_label("AAPL", {"AAPL"}, {"AAPL"}) == "둘 다"
    assert pw.membership_label("ADBE", {"ADBE"}, set()) == "나스닥100"
    assert pw.membership_label("JPM", set(), {"JPM"}) == "S&P100"
    assert pw.membership_label("XYZ", set(), set()) is None
    assert pw.prev_month_end("2026-11-05") == pd.Timestamp("2026-10-31")
    assert pw.prev_month_end("2026-10-31") == pd.Timestamp("2026-09-30")


# ── 신호 ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", range(4))
def test_judge_matches_sw1_signal_on(seed):
    px, q = _px(seed, jump_at=60), _px(100 + seed)["close"]
    for d0 in (DAYS[10], DAYS[25], DAYS[60], DAYS[61], DAYS[120]):
        a, b = pw.judge(px, q, d0), dc.signal_on(px, q, d0)
        assert (a is None) == (b is None)
        if a:
            assert a == pytest.approx(b)
    assert pw.judge(px, q, DAYS[60])["signal"] is True


def test_judge_does_not_use_rows_after_reaction_day():
    """미래 데이터 방지: 반응일 뒤 행을 지워도 같은 판정."""
    px, q = _px(7, jump_at=80), _px(8)["close"]
    d0 = DAYS[80]
    assert pw.judge(px, q, d0) == pw.judge(px.loc[:d0], q.loc[:d0], d0)


def test_judge_none_without_reaction_day_row():
    px, q = _px(1, gaps=0), _px(2)["close"]
    assert pw.judge(px.drop(index=DAYS[50]), q, DAYS[50]) is None


# ── 장부 진행 ───────────────────────────────────────────────────────────────


def _simulate(signals: pd.DataFrame, pxs: dict, bench_df: pd.DataFrame, hold: int, upto: int | None = None):
    """반응일 순서대로 거래일마다 step_day — 엔진과 같은 순서."""
    al = {t: pt.align(p, DAYS) for t, p in pxs.items()}
    bench = pt.align(bench_df, DAYS)
    trades: list[dict] = []
    by_day = {pd.Timestamp(d): g.to_dict("records") for d, g in signals.groupby("d0")}
    last = len(DAYS) - 1 if upto is None else upto
    for i in range(last + 1):
        d = DAYS[i]
        trades, _ = pw.step_day(trades, d, by_day.get(d, []), al, bench, DAYS, "main", hold)
    return trades


def test_stepping_matches_pead_timing_make_trades_b1():
    pxs = {f"T{i}": _px(20 + i, gaps=6) for i in range(5)}
    q = _px(99)
    sig = pd.DataFrame({"ticker": ["T0", "T1", "T0", "T2", "T3", "T0", "T4", "T1"],
                        "d0": [DAYS[d] for d in (5, 9, 20, 30, 41, 70, 80, 150)]})
    ref, ref_sk = pt.make_trades(sig, {t: pt.align(p, DAYS) for t, p in pxs.items()}, DAYS, "B1", 30)
    got = _simulate(sig, pxs, q, 30)
    closed = sorted((t for t in got if t["status"] == "closed"), key=lambda t: (t["d0"], t["ticker"]))
    assert [(t["ticker"], t["buy_day"], t["sell_day"]) for t in closed] == \
           [(r.ticker, r.buy_day.date().isoformat(), r.sell_day.date().isoformat()) for r in ref.itertuples()]
    assert np.allclose([t["buy_px"] for t in closed], ref["buy_px"])
    assert np.allclose([t["sell_px"] for t in closed], ref["sell_px"])
    skipped = sorted(t["skip"] for t in got if t["status"] == "skipped")
    assert skipped == sorted(ref_sk["skip"])


def test_stepping_uses_no_future_prices():
    """미래 데이터 방지: 거래일 k까지 진행한 결과가 k 뒤 가격을 지운 결과와 같다."""
    pxs = {f"T{i}": _px(40 + i) for i in range(3)}
    q = _px(98)
    sig = pd.DataFrame({"ticker": ["T0", "T1", "T2"], "d0": [DAYS[10], DAYS[12], DAYS[50]]})
    k = 60
    full = _simulate(sig, pxs, q, 30, upto=k)
    cut = _simulate(sig, {t: p.loc[:DAYS[k]] for t, p in pxs.items()}, q.loc[:DAYS[k]], 30, upto=k)
    assert full == cut


def test_pending_open_closed_states_and_busy_rule():
    pxs = {"A": _px(5)}
    q = _px(6)
    sig = pd.DataFrame({"ticker": ["A", "A", "A"], "d0": [DAYS[10], DAYS[15], DAYS[60]]})
    t10 = _simulate(sig, pxs, q, 40, upto=10)
    assert [t["status"] for t in t10] == ["pending"] and t10[0]["buy_day"] == DAYS[11].date().isoformat()
    t20 = _simulate(sig, pxs, q, 40, upto=20)
    assert [t["status"] for t in t20] == ["open", "skipped"] and t20[1]["skip"] == "보유 중"
    assert t20[0]["plan_day"] == DAYS[51].date().isoformat()
    t70 = _simulate(sig, pxs, q, 40, upto=70)
    assert [t["status"] for t in t70] == ["closed", "skipped", "open"]


def test_missing_buy_open_is_skipped_as_price_gap():
    px = _px(3)
    px.loc[DAYS[11], "open"] = np.nan
    got = _simulate(pd.DataFrame({"ticker": ["A"], "d0": [DAYS[10]]}), {"A": px}, _px(4), 20, upto=12)
    assert got[0]["status"] == "skipped" and got[0]["skip"] == "가격 공백"


def test_mark_and_scorecard():
    pxs = {"A": _px(11)}
    q = _px(12)
    al, bench = pt.align(pxs["A"], DAYS), pt.align(q, DAYS)
    got = _simulate(pd.DataFrame({"ticker": ["A"], "d0": [DAYS[10]]}), pxs, q, 30, upto=25)
    m = pw.mark(got[0], al, bench, DAYS, DAYS[25])
    assert m["held"] == 14
    assert m["ret"] == pytest.approx(pxs["A"]["close"].iloc[25] / pxs["A"]["open"].iloc[11] - 1)
    qr = q["close"].iloc[25] / q["open"].iloc[11] - 1
    assert m["vs_qqq"] == pytest.approx(m["ret"] - qr)
    closed = {"status": "closed", "buy_px": 100, "sell_px": 110, "q_buy": 100, "q_sell": 104, "fx_buy": 1300, "fx_sell": 1300}
    lost = {**closed, "sell_px": 95}
    sc = pw.scorecard([closed, lost, {"status": "open"}])
    assert sc["n"] == 2 and sc["win"] == 50.0 and sc["basis"] == "cost"
    assert pw.scorecard([])["n"] == 0
