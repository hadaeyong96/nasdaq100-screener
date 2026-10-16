"""추천 따라하기 장부(core/picks_book.py) — 장부 규칙 테스트. 네트워크 없이 합성 데이터만.

신호는 열(x_a1·x_b·x_a2·x_a3·x_e3·x_score)로 직접 켜는 가짜 묶음(STUB)으로 통제한다 —
장부 틀(체결 시점·자리·매도 먼저·SGOV·세금·분할)만 시험하고, 신호 규칙 자체는 test_bollinger_signals.py가 본다.
손절(swing_low)·E1(dc)·E2(rsi 50 하향)·A1 만료는 core.state가 원래 함수로 판정한다.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from core import picks_book as pb
from core import state as st

CFG = {"assumptions": {"reentry_cooldown_days": 5, "a1_to_a2_expiry_days": 10}}
C = 0.0007

STUB = st.SignalRules(
    name="stub",
    a1=lambda prev, row, cfg: bool(row["x_a1"]),
    a2=lambda row, cfg: bool(row["x_a2"]),
    a3=lambda row, cfg: bool(row["x_a3"]),
    b=lambda row, cfg: bool(row["x_b"]),
    e3=lambda row: bool(row["x_e3"]),
    ban=lambda *a: [],
    score=lambda df, date, row, kind, cfg: (None, int(row["x_score"])),
    a3_stop=lambda row, old: old,
)
IDX = pd.bdate_range("2025-11-03", periods=60)


def frame(close=100.0, score=10, **flags) -> pd.DataFrame:
    """flags: x_a1=[날짜 위치...] 처럼 신호를 켤 위치."""
    n = len(IDX)
    df = pd.DataFrame({
        "close": np.asarray(close, dtype=float) if np.ndim(close) else np.full(n, float(close)),
        "rsi": 40.0, "swing_low": 1.0, "dc": False, "gc": False, "macd_norm": 0.0, "x_score": score,
        "x_a1": False, "x_b": False, "x_a2": False, "x_a3": False, "x_e3": False,
    }, index=IDX)
    for k, pos in flags.items():
        col = k if k in df.columns else f"x_{k}"
        for p in pos if isinstance(pos, (list, tuple)) else [pos]:
            df.iloc[p, df.columns.get_loc(col)] = True if df[col].dtype == bool else pos
    return df


def run(frames: dict, upto: int, *, book=None, params=None, year_end_pos=(), briefing=None, scan=None,
        splits=None, sgov=100.0, start=0, initial=100_000.0):
    book = book or pb.new_book("추천 종목 따라하기", IDX[start], initial)
    params = params or pb.PicksParams(name=book["name"])
    rows, trades, sigs = [], [], []
    for i in range(start if book["last_date"] is None else IDX.get_loc(book["last_date"]) + 1, upto + 1):
        d = IDX[i]
        closes = {t: float(df.at[d, "close"]) for t, df in frames.items()}
        book, row, tr, sg = pb.process_day(
            book, d, frames=frames, closes=closes, sgov_close=sgov, dividends={}, splits=(splits or {}).get(i, {}),
            scan_tickers=scan or list(frames), earnings_map={}, rules=STUB, params=params, cfg=CFG,
            year_end=i in year_end_pos, briefing=(briefing or {}).get(i))
        rows.append(row)
        trades.extend(tr)
        sigs.extend(sg)
    return book, rows, trades, sigs


def test_start_day_puts_all_cash_in_sgov_with_commission():
    book, rows, trades, _ = run({"AAA": frame()}, 0)
    n = math.floor(100_000 / (100 * (1 + C)))
    assert book["sgov_qty"] == n and trades == []  # SGOV 매매는 매매기록에 쓰지 않는다
    assert book["cash"] == pytest.approx(100_000 - n * 100 * (1 + C))
    assert rows[0]["value"] == pytest.approx(100_000 - n * 100 * C)
    assert rows[0]["sgov_value"] == pytest.approx(n * 100)


def test_buy_fills_at_next_day_close_with_equity_over_ten():
    close = np.linspace(100, 130, len(IDX))
    book, rows, trades, _ = run({"AAA": frame(close=close, a1=[3])}, 4)
    t = [x for x in trades if x["side"] == pb.SIDE_BUY]
    assert len(t) == 1 and t[0]["signal_date"] == IDX[3] and t[0]["fill_date"] == IDX[4]
    assert t[0]["price"] == pytest.approx(close[4])
    equity_before = rows[3]["value"]  # SGOV 가격이 고정이라 4일 매수 전 총자산 = 3일 평가와 같다
    assert t[0]["qty"] == math.floor(equity_before / 10 / (close[4] * (1 + C)))
    assert book["positions"]["AAA"]["entry_date"] == IDX[4]
    assert book["signal_states"]["AAA"]["state"] == "정찰"


def test_ten_slots_by_score_and_no_slot_rows():
    frames = {f"T{i:02d}": frame(score=i, a1=[2]) for i in range(12)}
    book, _, trades, _ = run(frames, 3)
    bought = sorted(t["ticker"] for t in trades if t["side"] == pb.SIDE_BUY)
    skipped = [t for t in trades if t["side"] == pb.SIDE_SKIP]
    assert bought == [f"T{i:02d}" for i in range(2, 12)]  # 점수 높은 10종목
    assert sorted(t["ticker"] for t in skipped) == ["T00", "T01"]
    assert all(t["reason"].startswith("자리 없음") and t["qty"] == 0 for t in skipped)
    assert len(book["positions"]) == 10
    # 자리 없어 못 산 종목은 신호 상태가 '체결 안 됨'으로 대기에 돌아간다
    assert book["signal_states"]["T00"]["state"] == "대기"


def test_no_add_while_held_but_state_progresses():
    df = frame(a1=[2], a2=[5], a3=[8])
    book, _, trades, sigs = run({"AAA": df}, 10)
    assert [t["side"] for t in trades] == [pb.SIDE_BUY]  # A2·A3에 추가 매수 없음
    assert {s["kind"] for s in sigs} >= {"A1", "A2", "A3"}
    assert book["signal_states"]["AAA"]["state"] == "확정"  # 원래 전략처럼 단계는 넘어간다
    assert book["positions"]["AAA"]["qty"] == trades[0]["qty"]

    book2, _, _, _ = run({"AAA": df}, 10, params=pb.PicksParams(name="x", progress_on_skipped_adds=False))
    assert book2["signal_states"]["AAA"]["state"] == "정찰"  # 끄면 A2가 '체결 안 됨'으로 되돌아간다


def test_full_sell_on_e1_even_though_strategy_sells_part():
    df = frame(a1=[2], a2=[4], dc=[7])
    book, _, trades, _ = run({"AAA": df}, 8)
    buy = next(t for t in trades if t["side"] == pb.SIDE_BUY)
    sell = next(t for t in trades if t["side"] == pb.SIDE_SELL)
    assert sell["kind"] == "E1" and sell["signal_date"] == IDX[7] and sell["fill_date"] == IDX[8]
    assert sell["qty"] == buy["qty"] and "AAA" not in book["positions"]
    assert book["signal_states"]["AAA"]["state"] == "대기" and book["signal_states"]["AAA"]["cooldown_until"] is not None


def test_a1_expire_sells_and_rebuy_after_sell_on_new_entry():
    df = frame(a1=[2, 13], b=[20])  # 13은 재진입 대기(5영업일) 안이라 막힘, 20의 B는 다시 산다
    book, _, trades, _ = run({"AAA": df}, 21)
    kinds = [(t["side"], t["kind"], IDX.get_loc(t["fill_date"])) for t in trades]
    assert kinds == [(pb.SIDE_BUY, "A1", 3), (pb.SIDE_SELL, "A1_EXPIRE", 13), (pb.SIDE_BUY, "B", 21)]
    assert book["positions"]["AAA"]["kind"] == "B"


def test_same_day_sell_first_frees_slot_for_buy():
    frames = {f"T{i}": frame(score=50, a1=[2]) for i in range(10)}
    frames["T0"] = frame(score=50, a1=[2], close=[100.0] * 6 + [40.0] * 54)
    frames["T0"].iloc[2, frames["T0"].columns.get_loc("swing_low")] = 50.0  # 진입일 손절가 50
    frames["NEW"] = frame(score=99, a1=[6])
    book, _, trades, _ = run(frames, 7)
    day7 = [t for t in trades if t["fill_date"] == IDX[7]]
    assert [(t["side"], t["ticker"]) for t in day7] == [(pb.SIDE_SELL, "T0"), (pb.SIDE_BUY, "NEW")]
    assert day7[0]["kind"] == "STOP" and len(book["positions"]) == 10


def _sgov_round_trip(book_before: dict, need: float, px: float = 100.0):
    """현금이 need보다 모자라면 SGOV n주를 팔고, 남는 현금으로 m주를 다시 산다 — 장부 규칙 그대로 손으로 계산."""
    cash = book_before["cash"]
    n = max(0, math.ceil((need - cash) / (px * (1 - C)))) if need > cash else 0
    cash = cash + n * px * (1 - C) - need
    m = math.floor(cash / (px * (1 + C)))
    return n, m


def test_buy_sells_sgov_with_commission_and_value_moves_only_by_fees():
    close = np.full(len(IDX), 50.0)
    frames = {"AAA": frame(close=close, a1=[2])}
    book2, rows2, _, _ = run(frames, 2)
    book3, rows3, trades, _ = run(frames, 3, book=book2)
    buy = trades[0]
    n, m = _sgov_round_trip(book2, buy["amount"])
    assert n > 0  # 첫날 전액 SGOV라 사려면 SGOV를 판다
    assert book3["sgov_qty"] == book2["sgov_qty"] - n + m
    fees = buy["commission"] + n * 100 * C + m * 100 * C  # 종목 매수 + SGOV 매도 + 남는 현금 SGOV 재매수 수수료
    assert rows3[-1]["value"] == pytest.approx(rows2[-1]["value"] - fees)


def test_year_end_tax_deducted_from_cash_and_rolls_sale_pnl():
    close = np.r_[np.full(13, 100.0), np.full(len(IDX) - 13, 150.0)]
    frames = {"AAA": frame(close=close, a1=[2])}
    book13, rows13, trades, _ = run(frames, 13)  # 13일 A1 만료 매도(100→150 이익)
    sell = next(t for t in trades if t["side"] == pb.SIDE_SELL)
    assert sell["fill_date"] == IDX[13] and sell["realized_pnl"] > 0
    realized = book13["realized_ytd"]  # 종목 이익 + SGOV 매매 손익
    tax = 0.22 * realized
    n, m = _sgov_round_trip(book13, tax)
    book14, rows14, _, _ = run(frames, 14, book=book13, year_end_pos=(14,))
    assert book14["tax_cum"] == pytest.approx(tax)
    assert book14["sgov_qty"] == book13["sgov_qty"] - n + m
    assert rows14[-1]["value"] == pytest.approx(rows13[-1]["value"] - tax - (n + m) * 100 * C)
    cost_out = book13["sgov_basis"] * n / book13["sgov_qty"]
    assert book14["realized_ytd"] == pytest.approx(n * 100 * (1 - C) - cost_out)  # 납부용 매도 손익은 다음 해로


def test_held_ticker_out_of_scan_list_still_gets_sell_signal():
    df = frame(a1=[2], dc=[8], a2=[4])
    frames = {"GONE": df, "AAA": frame()}
    book, _, _, _ = run(frames, 5)
    # 6일부터 명단에서 빠져도(scan에는 엔진이 보유 종목을 더해 준다) 보유 종목이라 끝까지 처리
    book, _, trades, _ = run(frames, 9, book=book, scan=["AAA", "GONE"])
    assert any(t["side"] == pb.SIDE_SELL and t["ticker"] == "GONE" and t["kind"] == "E1" for t in trades)


def test_briefing_flag_yes_no_unknown():
    frames = {"AAA": frame(a1=[2]), "BBB": frame(a1=[2]), "CCC": frame(a1=[3])}
    briefing = {2: {"buy": {"AAA"}, "sell": set()}}  # 3일은 브리핑 기록 없음
    _, _, trades, _ = run(frames, 4, briefing=briefing)
    flag = {t["ticker"]: t["in_briefing"] for t in trades if t["side"] == pb.SIDE_BUY}
    assert flag == {"AAA": True, "BBB": False, "CCC": None}


def test_split_scales_qty_and_signal_prices():
    close = np.r_[np.full(6, 100.0), np.full(len(IDX) - 6, 50.0)]
    df = frame(close=close, a1=[2])
    df.iloc[2, df.columns.get_loc("swing_low")] = 90.0
    book, rows, trades, _ = run({"AAA": df}, 6, splits={6: {"AAA": 2.0}})
    q = trades[0]["qty"]
    assert book["positions"]["AAA"]["qty"] == 2 * q
    assert book["signal_states"]["AAA"]["stop"] == pytest.approx(45.0)
    assert not any(t["side"] == pb.SIDE_SELL for t in trades)  # 분할로 손절이 잘못 나지 않는다
    assert rows[-1]["value"] == pytest.approx(rows[-2]["value"], rel=1e-9)


def test_dividend_credited_after_withholding():
    book, rows, trades, _ = run({"AAA": frame(a1=[2])}, 3)
    q = book["positions"]["AAA"]["qty"]
    b2, row, _, _ = pb.process_day(book, IDX[4], frames={"AAA": frame(a1=[2])}, closes={"AAA": 100.0}, sgov_close=100.0,
                                   dividends={"AAA": 1.0, "SGOV": 0.5}, splits={}, scan_tickers=["AAA"], earnings_map={},
                                   rules=STUB, params=pb.PicksParams(name="x"), cfg=CFG, year_end=False, briefing=None)
    assert b2["div_cum"] == pytest.approx((q * 1.0 + book["sgov_qty"] * 0.5) * 0.85)


def test_no_lookahead_book_cut_equals_full():
    close = 100 + np.cumsum(np.sin(np.arange(len(IDX))))
    frames = {"AAA": frame(close=close, a1=[2], a2=[6], dc=[15]), "BBB": frame(close=close[::-1].copy(), b=[9])}
    _, full_rows, full_trades, _ = run(frames, 30)
    for cut in (5, 12, 16, 25):
        cut_frames = {t: df.iloc[: cut + 1] for t, df in frames.items()}
        _, rows, trades, _ = run(cut_frames, cut)
        assert [r["value"] for r in rows] == pytest.approx([r["value"] for r in full_rows[: cut + 1]])
        assert [(t["ticker"], t["side"], t["fill_date"]) for t in trades] == \
            [(t["ticker"], t["side"], t["fill_date"]) for t in full_trades if t["fill_date"] <= IDX[cut]]


def test_trade_stats():
    trades = [
        {"side": pb.SIDE_SELL, "kind": "E1", "realized_pnl": 10.0, "fill_date": IDX[10], "entry_date": IDX[3]},
        {"side": pb.SIDE_SELL, "kind": "STOP", "realized_pnl": -5.0, "fill_date": IDX[12], "entry_date": IDX[8]},
        {"side": pb.SIDE_SELL, "kind": "TAX", "realized_pnl": 1.0, "fill_date": IDX[14], "entry_date": IDX[8]},
        {"side": pb.SIDE_SKIP, "kind": "A1", "realized_pnl": None, "fill_date": IDX[14], "entry_date": None},
    ]
    s = pb.trade_stats(trades)
    assert s["completed"] == 2 and s["win_rate_pct"] == 50.0
    assert s["avg_hold_days"] == pytest.approx(((IDX[10] - IDX[3]).days + (IDX[12] - IDX[8]).days) / 2)
