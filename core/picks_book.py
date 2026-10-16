"""가상 장부 「추천 종목 따라하기」·「볼린저·RSI·MACD 따라하기」 — 하루 처리 순수 함수.

네트워크·파일·DB·현재 시각에 접근하지 않는다. engine/virtual_books.py가 DB에서 장부 상태를 읽어
아직 처리하지 않은 거래일마다 process_day를 부르고, 결과 상태를 다시 DB에 쌓는다.
실제 체결·구글 시트와 무관하다 — 장부가 종목마다 core.state.process_day를 직접 돌려 자기 신호 상태를 가진다.

장부 규칙 (사용자 확정 2026-10-14, docs/design/virtual_books.md):
- 매수: 신규 진입 신호(A1·B)만. t일 신호 → t+1 거래일 종가 체결. 자리 최대 10종목,
  한 종목 = 그날(매도 체결 뒤) 장부 총자산 ÷ 10, 정수 주식. 후보가 남은 자리보다 많으면 점수순,
  자리가 없으면 매매기록에 "자리 없음". 보유 중 추가 매수 없음, 판 뒤 새 진입 신호면 다시 산다.
- A2·A3(2·3차 추가 매수)는 사지 않는다. [해석] 신호 상태는 원래 전략이 그 추천을 따랐을 때처럼
  다음 단계(확인·확정)로 넘어간다 — 그래야 손절·E1·E2·E3가 원래 전략과 같은 때 나온다
  (config virtual_books.picks.progress_on_skipped_adds, 사용자 확인 필요).
- 매도: 손절·E1·E2·E3·A1 만료 중 하나라도 나오면 t+1 거래일 종가에 전량. 원래 전략의 E1·E2 부분
  매도와 다르다. 매도한 종목의 신호 상태는 신호일에 대기 + 재진입 대기(core.state.force_flat).
- 같은 날 매도·매수가 겹치면 매도 먼저.
- 남는 현금은 SGOV 정수 주식으로 들고 있다. SGOV 매매에도 수수료를 물리지만 매매기록에는 쓰지 않는다.
- 수수료 편도·분배금 원천징수·연말 실현 순이익 세금(장부 현금 차감, 모자라면 SGOV → 보유 종목 순으로
  팔아 내고 그 매도 손익은 다음 해)은 기존 가상 장부와 같다.
- 분할: 보유 종목이 분할하면 그날 수량 × 비율, 신호 상태의 손절가·진입가 ÷ 비율 (가격이 분할 반영
  종가라 과거 값이 소급 조정되기 때문).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import pandas as pd

from core import state as st

SELL_KINDS = ("STOP", "A1_EXPIRE", "E3", "E1", "E2")
ENTRY_KINDS = ("A1", "B")
ADD_KINDS = ("A2", "A3")
KIND_LABEL = {
    "A1": "1차 정찰(A1)", "B": "재진입(B)", "STOP": "손절", "A1_EXPIRE": "A1 만료",
    "E1": "E1 모멘텀 약화", "E2": "E2 추세 약화", "E3": "E3 구조 붕괴",
}
SIDE_BUY, SIDE_SELL, SIDE_SKIP = "매수", "매도", "건너뜀"


@dataclass(frozen=True)
class PicksParams:
    """장부 설정 (퍼센트 값은 % 단위)."""

    name: str
    slots: int = 10
    commission_pct: float = 0.07
    slippage_pct: float = 0.0
    tax_rate_pct: float = 22.0
    dividend_withholding_pct: float = 15.0
    cash_ticker: str = "SGOV"
    progress_on_skipped_adds: bool = True


def new_book(name: str, start_date, initial_usd: float) -> dict:
    """빈 장부 상태 (전액 현금 — 시작일 처리에서 SGOV로 옮긴다)."""
    return {
        "name": name, "start_date": pd.Timestamp(start_date), "initial_usd": float(initial_usd),
        "cash": float(initial_usd), "sgov_qty": 0, "sgov_basis": 0.0, "positions": {},
        "realized_ytd": 0.0, "tax_cum": 0.0, "div_cum": 0.0, "peak": float(initial_usd),
        "pending_sells": [], "pending_buys": [], "signal_states": {}, "last_date": None,
    }


def _trade(book: dict, d, side, ticker, qty, price, fee, amount, pnl, reason, signal_date, kind, in_briefing,
           entry_date=None) -> dict:
    return {
        "book": book["name"], "signal_date": pd.Timestamp(signal_date) if signal_date is not None else None,
        "fill_date": pd.Timestamp(d), "side": side, "ticker": ticker, "qty": int(qty), "price": float(price),
        "commission": float(fee), "amount": float(amount), "realized_pnl": pnl, "reason": reason, "kind": kind,
        "in_briefing": in_briefing, "entry_date": pd.Timestamp(entry_date) if entry_date is not None else None,
    }


def _prune_alerts(state: dict, d: pd.Timestamp, keep_days: int = 40) -> None:
    """중복 알림 키(KIND:YYYY-MM-DD)는 같은 날짜에만 쓰이므로 오래된 키를 버려 DB 크기를 묶는다."""
    cut = (pd.Timestamp(d) - pd.Timedelta(days=keep_days)).date().isoformat()
    state["sent_alerts"] = [k for k in state.get("sent_alerts", []) if k.rsplit(":", 1)[-1] >= cut]


def _sell_cash_ticker(book: dict, need: float, px: float, c: float) -> float:
    """현금이 need만큼 모자라면 SGOV를 정수 주식으로 팔아 채운다. 판 금액(수수료 뒤)을 돌려준다."""
    if need <= 1e-9 or book["sgov_qty"] <= 0 or px <= 0:
        return 0.0
    n = min(book["sgov_qty"], int(math.ceil(need / (px * (1.0 - c)) - 1e-9)))
    gross = n * px
    proceeds = gross * (1.0 - c)
    cost_out = book["sgov_basis"] * n / book["sgov_qty"]
    book["realized_ytd"] += proceeds - cost_out
    book["sgov_basis"] -= cost_out
    book["sgov_qty"] -= n
    book["cash"] += proceeds
    return proceeds


def _buy_cash_ticker(book: dict, px: float, c: float) -> None:
    """남는 현금으로 SGOV를 정수 주식만큼 산다."""
    if px <= 0:
        return
    n = int(math.floor(book["cash"] / (px * (1.0 + c)) + 1e-9))
    if n <= 0:
        return
    cost = n * px * (1.0 + c)
    book["cash"] -= cost
    book["sgov_qty"] += n
    book["sgov_basis"] += cost


def _equity(book: dict, closes: dict, sgov_px: float) -> float:
    return book["cash"] + book["sgov_qty"] * sgov_px + sum(
        p["qty"] * closes.get(t, p["last_close"]) for t, p in book["positions"].items())


def process_day(
    book: dict,
    d,
    *,
    frames: dict[str, pd.DataFrame],
    closes: dict[str, float],
    sgov_close: float,
    dividends: dict[str, float],
    splits: dict[str, float],
    scan_tickers: list[str],
    earnings_map: dict,
    rules: st.SignalRules,
    params: PicksParams,
    cfg: dict,
    year_end: bool,
    briefing: dict | None,
) -> tuple[dict, dict, list[dict], list[dict]]:
    """장부 하루 처리.

    입력: book(장부 상태 — 바꾸지 않는다), d(거래일), frames({티커: 지표 DataFrame}),
         closes({티커: d일 종가 — 데이터가 없는 티커는 빠짐}), sgov_close, dividends({티커 또는 SGOV:
         d일이 배당락일인 주당 분배금}), splits({티커: d일 분할 비율}), scan_tickers(오늘 신호를 볼 종목 —
         명단 + 보유 + 체결 대기), earnings_map({티커: 다음 실적일}), rules(신호 묶음), params, cfg,
         year_end(d가 그해 마지막 거래일), briefing({"buy": set, "sell": set} — d일 브리핑 추천, 모르면 None)
    출력: (새 장부 상태, 일별 행, 매매기록 행 목록, 신호 기록 행 목록)
    """
    b = copy.deepcopy(book)
    d = pd.Timestamp(d)
    c = params.commission_pct / 100.0
    s = params.slippage_pct / 100.0
    w = params.dividend_withholding_pct / 100.0
    tr = params.tax_rate_pct / 100.0
    trades: list[dict] = []
    signals: list[dict] = []

    # ⓪ 분할 — 수량 × 비율, 신호 상태의 가격 ÷ 비율
    for t, ratio in splits.items():
        if t in b["positions"] and ratio and ratio > 0 and ratio != 1:
            p = b["positions"][t]
            p["qty"] = int(round(p["qty"] * ratio))
            p["last_close"] = p["last_close"] / ratio
            sst = b["signal_states"].get(t)
            if sst:
                if sst.get("stop") is not None:
                    sst["stop"] = sst["stop"] / ratio
                sst["entries"] = {u: (v / ratio if v else v) for u, v in sst.get("entries", {}).items()}

    # ① 분배금 — 어제 종가까지 들고 있던 수량 기준
    for t, p in b["positions"].items():
        dps = dividends.get(t, 0.0)
        if dps > 0 and p["qty"] > 0:
            net = p["qty"] * dps * (1.0 - w)
            b["cash"] += net
            b["div_cum"] += net
    dps = dividends.get(params.cash_ticker, 0.0)
    if dps > 0 and b["sgov_qty"] > 0:
        net = b["sgov_qty"] * dps * (1.0 - w)
        b["cash"] += net
        b["div_cum"] += net

    # ② 매도 체결 (어제 신호) — 매도 먼저
    carry_sells = []
    for o in b["pending_sells"]:
        t = o["ticker"]
        if t not in b["positions"]:
            continue
        if t not in closes:
            carry_sells.append(o)  # 오늘 종가가 없으면(데이터 구멍) 다음 거래일 종가로 넘긴다
            continue
        p = b["positions"].pop(t)
        px = closes[t] * (1.0 - s)
        gross = p["qty"] * px
        fee = gross * c
        proceeds = gross - fee
        pnl = proceeds - p["basis"]
        b["realized_ytd"] += pnl
        b["cash"] += proceeds
        trades.append(_trade(b, d, SIDE_SELL, t, p["qty"], px, fee, proceeds, pnl,
                             f"{KIND_LABEL.get(o['kind'], o['kind'])} 신호 → 전량 매도", o["signal_date"], o["kind"],
                             o.get("in_briefing"), entry_date=p["entry_date"]))
    b["pending_sells"] = carry_sells

    # ③ 매수 체결 (어제 신호) — 점수순, 자리 10개, 한 종목 = 총자산 ÷ 10
    buys = sorted(b["pending_buys"], key=lambda o: (-o["score"], o["ticker"]))
    equity = _equity(b, closes, sgov_close)
    slot_value = equity / params.slots
    for o in buys:
        t = o["ticker"]
        label = f"{KIND_LABEL.get(o['kind'], o['kind'])}, 점수 {o['score']}"
        if t in b["positions"]:
            trades.append(_trade(b, d, SIDE_SKIP, t, 0, closes.get(t, 0.0), 0.0, 0.0, None,
                                 f"이미 보유 — 추가 매수 없음 ({label})", o["signal_date"], o["kind"], o.get("in_briefing")))
            continue
        if t not in closes:
            # 체결일 종가가 없으면(데이터 구멍) 사지 않는다 — 신호 상태는 오늘 처리에서 '체결 안 됨'으로 되돌아간다
            trades.append(_trade(b, d, SIDE_SKIP, t, 0, 0.0, 0.0, 0.0, None, f"체결일 종가 없음 ({label})",
                                 o["signal_date"], o["kind"], o.get("in_briefing")))
            continue
        if len(b["positions"]) >= params.slots:
            trades.append(_trade(b, d, SIDE_SKIP, t, 0, closes[t], 0.0, 0.0, None, f"자리 없음 ({label})",
                                 o["signal_date"], o["kind"], o.get("in_briefing")))
            continue
        px = closes[t] * (1.0 + s)
        qty = int(math.floor(slot_value / (px * (1.0 + c)) + 1e-9))
        need = qty * px * (1.0 + c)
        if need > b["cash"]:
            _sell_cash_ticker(b, need - b["cash"], sgov_close * (1.0 - s), c)
        if need > b["cash"] + 1e-9:
            qty = int(math.floor(max(b["cash"], 0.0) / (px * (1.0 + c)) + 1e-9))
            need = qty * px * (1.0 + c)
        if qty <= 0:
            trades.append(_trade(b, d, SIDE_SKIP, t, 0, closes[t], 0.0, 0.0, None, f"자금 부족 ({label})",
                                 o["signal_date"], o["kind"], o.get("in_briefing")))
            continue
        fee = qty * px * c
        b["cash"] -= need
        b["positions"][t] = {"qty": qty, "basis": need, "entry_date": d, "signal_date": pd.Timestamp(o["signal_date"]),
                             "kind": o["kind"], "last_close": closes[t]}
        trades.append(_trade(b, d, SIDE_BUY, t, qty, px, fee, need, None, f"{label} → 매수", o["signal_date"], o["kind"],
                             o.get("in_briefing")))
        sst = b["signal_states"].get(t)
        if sst is not None and sst.get("pending"):
            unit = sst["pending"]["unit"]
            b["signal_states"][t] = st.apply_fill(sst, {"unit": unit, "side": "buy", "price": px, "qty": qty}, cfg)
    b["pending_buys"] = []

    # ④ 연말 세금 — 장부 현금에서 뺀다. 모자라면 SGOV → 보유 종목(평가액 큰 순) 순으로 판다
    if year_end:
        tax = max(b["realized_ytd"], 0.0) * tr
        b["realized_ytd"] = 0.0
        b["tax_cum"] += tax
        if tax > b["cash"]:
            _sell_cash_ticker(b, tax - b["cash"], sgov_close * (1.0 - s), c)
        for t in sorted(b["positions"], key=lambda x: -b["positions"][x]["qty"] * closes.get(x, b["positions"][x]["last_close"])):
            short = tax - b["cash"]
            if short <= 1e-9:
                break
            p = b["positions"][t]
            px = closes.get(t, p["last_close"]) * (1.0 - s)
            n = min(p["qty"], int(math.ceil(short / (px * (1.0 - c)) - 1e-9)))
            gross = n * px
            fee = gross * c
            cost_out = p["basis"] * n / p["qty"]
            pnl = gross - fee - cost_out
            b["realized_ytd"] += pnl  # 납부용 매도 손익은 다음 해
            b["cash"] += gross - fee
            p["qty"] -= n
            p["basis"] -= cost_out
            trades.append(_trade(b, d, SIDE_SELL, t, n, px, fee, gross - fee, pnl, f"{d.year}년 세금 납부 매도 (세금 ${tax:,.2f})",
                                 d, "TAX", None, entry_date=p["entry_date"]))
            if p["qty"] <= 0:
                b["positions"].pop(t)
                b["signal_states"][t] = st.force_flat(b["signal_states"].get(t) or st.init_state(t), d, cfg)
        b["cash"] -= tax

    # ⑤ 남는 현금 → SGOV
    _buy_cash_ticker(b, sgov_close * (1.0 + s), c)

    # ⑥ 오늘 신호 (종목마다 core.state.process_day, 장부 자기 상태)
    for t in scan_tickers:
        df = frames.get(t)
        if df is None or d not in df.index:
            continue
        sst = b["signal_states"].get(t) or st.init_state(t)
        events, new = st.process_day(df, d, sst, cfg, earnings_date=earnings_map.get(t), new_entry_allowed=True, rules=rules)
        held = t in b["positions"]
        sell_events = [e for e in events if e["kind"] in SELL_KINDS]
        for e in events:
            k = e["kind"]
            if k in ENTRY_KINDS:
                ib = None if briefing is None else (t in briefing.get("buy", set()))
                b["pending_buys"].append({"ticker": t, "kind": k, "score": int(e.get("score", 0)), "signal_date": d,
                                          "in_briefing": ib})
            elif k in ADD_KINDS and held and params.progress_on_skipped_adds:
                # 추가 매수는 하지 않지만 원래 전략처럼 다음 단계로 넘어가게 체결 표시만 한다(수량 1, 장부 수량과 무관)
                new = st.apply_fill(new, {"unit": e["unit"], "side": "buy", "price": float(e.get("price", 0.0)), "qty": 1}, cfg)
            if k in ENTRY_KINDS + ADD_KINDS + SELL_KINDS + ("BLOCKED",):
                signals.append({"book": b["name"], "date": d, "ticker": t, "kind": k,
                                "stage": e.get("stage"), "score": e.get("score"),
                                "note": ", ".join(e.get("reasons", [])) if k == "BLOCKED" else ""})
        if held and sell_events and not any(o["ticker"] == t for o in b["pending_sells"]):
            k = sell_events[0]["kind"]
            ib = None if briefing is None else (t in briefing.get("sell", set()))
            b["pending_sells"].append({"ticker": t, "kind": k, "signal_date": d, "in_briefing": ib})
            new = st.force_flat(new, d, cfg)
        _prune_alerts(new, d)
        b["signal_states"][t] = new

    # ⑦ 평가
    for t, p in b["positions"].items():
        if t in closes:
            p["last_close"] = closes[t]
    value = _equity(b, closes, sgov_close)
    b["peak"] = max(b["peak"], value)
    stock_liq = sum(p["qty"] * p["last_close"] * (1.0 - s) * (1.0 - c) for p in b["positions"].values())
    sgov_liq = b["sgov_qty"] * sgov_close * (1.0 - s) * (1.0 - c)
    unreal = stock_liq - sum(p["basis"] for p in b["positions"].values()) + sgov_liq - b["sgov_basis"]
    liq = b["cash"] + stock_liq + sgov_liq - max(b["realized_ytd"] + unreal, 0.0) * tr
    b["last_date"] = d
    row = {
        "book": b["name"], "date": d, "holding": f"{len(b['positions'])}종목+{params.cash_ticker}",
        "qty": len(b["positions"]), "close": float("nan"), "cash": b["cash"], "value": value,
        "cum_return_pct": (value / b["initial_usd"] - 1.0) * 100.0,
        "drawdown_pct": (value / b["peak"] - 1.0) * 100.0,
        "realized_ytd": b["realized_ytd"], "tax_cum": b["tax_cum"], "value_after_tax_liq": liq,
        "dividends_cum": b["div_cum"], "sgov_value": b["sgov_qty"] * sgov_close,
    }
    return b, row, trades, signals


def trade_stats(trades: list[dict]) -> dict:
    """완료 거래(신호 매도로 끝난 거래) 수·승률·평균 보유일. 세금 납부 매도·건너뜀은 빼고 센다."""
    done = [t for t in trades if t["side"] == SIDE_SELL and t["kind"] in SELL_KINDS]
    if not done:
        return {"completed": 0, "win_rate_pct": None, "avg_hold_days": None}
    wins = sum(1 for t in done if (t["realized_pnl"] or 0) > 0)
    holds = [(pd.Timestamp(t["fill_date"]) - pd.Timestamp(t["entry_date"])).days for t in done if t.get("entry_date") is not None]
    return {"completed": len(done), "win_rate_pct": wins / len(done) * 100.0,
            "avg_hold_days": sum(holds) / len(holds) if holds else None}
