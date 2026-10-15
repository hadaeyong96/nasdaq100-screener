"""가상 장부 8번 「6개월 강세 따라하기」 — 순위 매기기·하루 처리 순수 함수 (작업 5-2).

네트워크·파일·DB·현재 시각에 접근하지 않는다. engine/picks_books.py가 DB에서 장부 상태를 읽어
아직 처리하지 않은 거래일마다 process_day를 부르고 결과를 다시 DB에 쌓는다. 실제 주문과 무관하다.

장부 규칙 (사전 등록, 2026-10-15 사용자 지시 — 임의 변경 금지, docs/design/virtual_books.md):
- 신호일: 매월 마지막 거래일 종가 (월말 판정은 engine이 NYSE 달력으로 해서 넘긴다).
- 점수: 최근 126거래일 수익률 − 같은 기간 QQQ 수익률(%p). 기간 = QQQ 거래일 기준 126거래일 전 → 신호일.
  대상: 그날 스크리너 명단(나스닥 100) 중 126거래일 전·신호일 종가가 모두 있고 신호일 종가 5달러 이상.
- 선택: 점수 상위 10. 체결: 신호 다음 거래일 종가.
- 매달 같은 비중(총자산 ÷ 10)으로 다시 맞춘다. 남은 종목은 비중 차이만 사고판다. 매도 먼저, 그다음 매수(점수순).
- 보유 중 명단에서 빠진 종목: 다음 교체일까지 들고 있다가 교체일에 판다(명단 밖이라 상위 10에 못 든다).
- 쉬는 돈 없음: SGOV를 쓰지 않는다. 정수 주식으로 남는 잔돈·분배금은 현금으로 둔다(다음 교체일에 다시 나눈다).
- 비용·세금·환율은 기존 가상 장부와 같다: 수수료 편도, 분배금 원천징수 뒤 현금, 연말 실현 순이익 × 세율을
  장부 현금에서 빼고 모자라면 보유 종목(평가액 큰 순)을 팔아 낸다(그 매도 손익은 다음 해). 분할은 수량 × 비율.
- 시작: start_date를 비워 두고, 장부가 처음 돈 날(armed_date) 이후 첫 월말 신호일이 시작일이다(그날은 현금).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import pandas as pd

SIDE_BUY, SIDE_SELL, SIDE_SKIP = "매수", "매도", "건너뜀"
KIND_IN, KIND_OUT, KIND_ADJ, KIND_LIST_OUT, KIND_TAX = "MOM_IN", "MOM_OUT", "MOM_ADJ", "MOM_LIST_OUT", "TAX"
WAITING, RUNNING = "waiting", "running"


@dataclass(frozen=True)
class MomentumParams:
    """장부 설정 (퍼센트 값은 % 단위)."""

    name: str
    top_n: int = 10
    lookback_days: int = 126
    min_price: float = 5.0
    commission_pct: float = 0.07
    slippage_pct: float = 0.0
    tax_rate_pct: float = 22.0
    dividend_withholding_pct: float = 15.0


def new_book(name: str, armed_date) -> dict:
    """시작 전(첫 월말 신호 대기) 장부 상태. 시작 자금은 시작일에 engine이 start_book으로 넣는다."""
    return {
        "name": name, "status": WAITING, "armed_date": pd.Timestamp(armed_date), "start_date": None,
        "initial_usd": None, "cash": 0.0, "positions": {}, "realized_ytd": 0.0, "tax_cum": 0.0, "div_cum": 0.0,
        "peak": 0.0, "pending": None, "carry_sells": [], "rebalances": 0, "last_date": None,
    }


def start_book(book: dict, d, initial_usd: float) -> dict:
    """첫 월말 신호일에 시작 자금(달러)을 넣는다."""
    b = copy.deepcopy(book)
    b.update(status=RUNNING, start_date=pd.Timestamp(d), initial_usd=float(initial_usd), cash=float(initial_usd),
             peak=float(initial_usd))
    return b


def rank(closes: dict[str, pd.Series], qqq_close: pd.Series, d, universe: list[str], params: MomentumParams) -> dict:
    """신호일 d의 6개월 강세 순위.

    입력: closes({티커: 종가 Series(날짜 인덱스)}), qqq_close(QQQ 종가 Series), d(신호일), universe(그날 명단), params
    출력: {"ok": bool, "window": (시작일, d) 또는 None, "qqq_return": 소수 또는 None,
           "ranked": [{ticker, score_pp, ret, close}] 점수 큰 순(전 대상), "targets": 상위 top_n,
           "excluded": {티커: 사유}}
    """
    d = pd.Timestamp(d)
    q = qqq_close[qqq_close.index <= d].dropna()
    n = int(params.lookback_days)
    out = {"ok": False, "window": None, "qqq_return": None, "ranked": [], "targets": [], "excluded": {}}
    if len(q) <= n or q.index[-1] != d:
        return out
    start = q.index[-1 - n]
    q_ret = float(q.iloc[-1] / q.iloc[-1 - n] - 1.0)
    out.update(ok=True, window=(start, d), qqq_return=q_ret)
    ranked = []
    for t in sorted(set(universe)):
        s = closes.get(t)
        c0 = None if s is None else s.get(start)
        c1 = None if s is None else s.get(d)
        if c0 is None or c1 is None or pd.isna(c0) or pd.isna(c1) or float(c0) <= 0:
            out["excluded"][t] = f"{n}거래일 가격 없음"
            continue
        if float(c1) < params.min_price:
            out["excluded"][t] = f"종가 {params.min_price:g}달러 미만"
            continue
        ret = float(c1) / float(c0) - 1.0
        ranked.append({"ticker": t, "score_pp": (ret - q_ret) * 100.0, "ret": ret, "close": float(c1)})
    ranked.sort(key=lambda r: (-r["score_pp"], r["ticker"]))
    out["ranked"] = ranked
    out["targets"] = ranked[: int(params.top_n)]
    return out


def _trade(b, d, side, ticker, qty, price, fee, amount, pnl, reason, signal_date, kind, entry_date=None) -> dict:
    return {
        "book": b["name"], "signal_date": pd.Timestamp(signal_date) if signal_date is not None else None,
        "fill_date": pd.Timestamp(d), "side": side, "ticker": ticker, "qty": int(qty), "price": float(price),
        "commission": float(fee), "amount": float(amount), "realized_pnl": pnl, "reason": reason, "kind": kind,
        "in_briefing": None, "entry_date": pd.Timestamp(entry_date) if entry_date is not None else None,
    }


def _equity(b: dict, closes: dict) -> float:
    return b["cash"] + sum(p["qty"] * closes.get(t, p["last_close"]) for t, p in b["positions"].items())


def _sell(b, d, t, n, close, s, c, reason, signal_date, kind, trades) -> None:
    p = b["positions"][t]
    n = min(int(n), p["qty"])
    if n <= 0:
        return
    px = close * (1.0 - s)
    gross = n * px
    fee = gross * c
    cost_out = p["basis"] * n / p["qty"]
    pnl = gross - fee - cost_out
    b["realized_ytd"] += pnl
    b["cash"] += gross - fee
    p["qty"] -= n
    p["basis"] -= cost_out
    trades.append(_trade(b, d, SIDE_SELL, t, n, px, fee, gross - fee, pnl, reason, signal_date, kind, p["entry_date"]))
    if p["qty"] <= 0:
        b["positions"].pop(t)


def _buy(b, d, t, n, close, s, c, reason, signal_date, kind, trades) -> None:
    px = close * (1.0 + s)
    n = min(int(n), int(math.floor(max(b["cash"], 0.0) / (px * (1.0 + c)) + 1e-9)))
    if n <= 0:
        trades.append(_trade(b, d, SIDE_SKIP, t, 0, close, 0.0, 0.0, None, f"자금 부족 ({reason})", signal_date, kind))
        return
    need = n * px * (1.0 + c)
    fee = n * px * c
    b["cash"] -= need
    p = b["positions"].get(t)
    if p is None:
        b["positions"][t] = {"qty": n, "basis": need, "entry_date": pd.Timestamp(d), "last_close": close}
    else:
        p["qty"] += n
        p["basis"] += need
        p["last_close"] = close
    trades.append(_trade(b, d, SIDE_BUY, t, n, px, fee, need, None, reason, signal_date, kind,
                         b["positions"][t]["entry_date"]))


def _rebalance(b, d, closes, params, trades) -> None:
    """어제(신호일) 순위로 오늘 종가에 다시 맞춘다. 매도 먼저, 그다음 매수(점수순)."""
    c = params.commission_pct / 100.0
    s = params.slippage_pct / 100.0
    pend = b["pending"]
    sig = pend["signal_date"]
    targets = [x["ticker"] for x in pend["targets"]]
    scores = {x["ticker"]: x["score_pp"] for x in pend["targets"]}
    off_list = set(pend.get("off_list", []))
    slot = _equity(b, closes) / params.top_n  # 체결 전 총자산(오늘 종가 평가) ÷ 10
    carry = []

    def target_qty(t):
        return int(math.floor(slot / (closes[t] * (1.0 + s) * (1.0 + c)) + 1e-9))

    # ① 상위 10 밖 보유 종목 전량 매도 (명단 이탈 종목 포함)
    for t in sorted(b["positions"]):
        if t in targets:
            continue
        if t not in closes:
            carry.append(t)  # 체결일 종가가 없으면 다음 거래일 종가로 넘긴다
            continue
        reason, kind = (("명단 이탈 → 교체일 전량 매도", KIND_LIST_OUT) if t in off_list
                        else ("상위 10 탈락 → 전량 매도", KIND_OUT))
        _sell(b, d, t, b["positions"][t]["qty"], closes[t], s, c, f"월간 교체: {reason}", sig, kind, trades)
    # ② 남은 종목 중 비중이 넘는 것 줄이기
    for t in targets:
        if t in b["positions"] and t in closes:
            diff = target_qty(t) - b["positions"][t]["qty"]
            if diff < 0:
                _sell(b, d, t, -diff, closes[t], s, c, "월간 교체: 비중 맞춤 매도", sig, KIND_ADJ, trades)
    # ③ 신규 편입·비중 모자란 종목 사기 (점수순)
    for t in targets:
        if t not in closes:
            trades.append(_trade(b, d, SIDE_SKIP, t, 0, 0.0, 0.0, 0.0, None,
                                 f"체결일 종가 없음 (점수 {scores[t]:+.1f}%p)", sig, KIND_IN))
            continue
        held = b["positions"].get(t, {}).get("qty", 0)
        diff = target_qty(t) - held
        if diff > 0:
            if held:
                _buy(b, d, t, diff, closes[t], s, c, "월간 교체: 비중 맞춤 매수", sig, KIND_ADJ, trades)
            else:
                _buy(b, d, t, diff, closes[t], s, c, f"월간 교체: 신규 편입 (점수 {scores[t]:+.1f}%p)", sig, KIND_IN, trades)
    b["carry_sells"] = [{"ticker": t, "signal_date": sig, "off_list": t in off_list} for t in carry]
    b["rebalances"] += 1
    b["pending"] = None


def process_day(book: dict, d, *, closes: dict[str, float], dividends: dict[str, float], splits: dict[str, float],
                signal: dict | None, universe: list[str], params: MomentumParams, year_end: bool) -> tuple[dict, dict | None, list[dict]]:
    """하루 처리.

    입력: book(상태 — 시작 전이면 그대로 돌려준다), d(거래일), closes({티커: d일 종가 — 없는 티커는 빠짐}),
         dividends({티커: d일이 배당락일인 주당 분배금}), splits({티커: d일 분할 비율}),
         signal(d가 월말 신호일이면 rank() 결과, 아니면 None), universe(그날 명단 — 명단 이탈 표시용),
         params, year_end(그해 마지막 거래일이면 True)
    출력: (새 상태, 일별 행(시작 전이면 None), 매매 행 목록)
    순서: 분할 → 분배금 → 넘어온 매도 → 교체 체결(어제 신호) → 연말 세금 → 오늘 신호 → 평가
    """
    b = copy.deepcopy(book)
    d = pd.Timestamp(d)
    trades: list[dict] = []
    if b["status"] != RUNNING:
        b["last_date"] = d
        return b, None, trades
    c = params.commission_pct / 100.0
    s = params.slippage_pct / 100.0
    w = params.dividend_withholding_pct / 100.0
    tr = params.tax_rate_pct / 100.0

    for t, ratio in splits.items():  # ⓪ 분할
        if t in b["positions"] and ratio and ratio > 0 and ratio != 1:
            p = b["positions"][t]
            p["qty"] = int(round(p["qty"] * ratio))
            p["last_close"] = p["last_close"] / ratio
    for t, p in b["positions"].items():  # ① 분배금 — 어제까지 들고 있던 수량, 원천징수 뒤 현금
        dps = dividends.get(t, 0.0)
        if dps > 0 and p["qty"] > 0:
            net = p["qty"] * dps * (1.0 - w)
            b["cash"] += net
            b["div_cum"] += net

    carry = []  # ② 지난 교체일에 종가가 없어 못 판 종목
    for o in b.get("carry_sells", []):
        t = o["ticker"]
        if t not in b["positions"]:
            continue
        if t not in closes:
            carry.append(o)
            continue
        reason, kind = (("명단 이탈 → 교체일 전량 매도", KIND_LIST_OUT) if o.get("off_list")
                        else ("상위 10 탈락 → 전량 매도", KIND_OUT))
        _sell(b, d, t, b["positions"][t]["qty"], closes[t], s, c, f"월간 교체: {reason} (종가 없어 넘김)",
              o["signal_date"], kind, trades)
    b["carry_sells"] = carry

    if b.get("pending"):  # ③ 교체 체결 (어제 신호, 오늘 종가)
        _rebalance(b, d, closes, params, trades)
        b["carry_sells"] = carry + b["carry_sells"]

    if year_end:  # ④ 연말 세금 — 장부 현금에서 뺀다. 모자라면 보유 종목(평가액 큰 순)을 판다
        tax = max(b["realized_ytd"], 0.0) * tr
        b["realized_ytd"] = 0.0
        b["tax_cum"] += tax
        for t in sorted(b["positions"], key=lambda x: -b["positions"][x]["qty"] * closes.get(x, b["positions"][x]["last_close"])):
            short = tax - b["cash"]
            if short <= 1e-9:
                break
            px_close = closes.get(t, b["positions"][t]["last_close"])
            n = int(math.ceil(short / (px_close * (1.0 - s) * (1.0 - c)) - 1e-9))
            _sell(b, d, t, n, px_close, s, c, f"{d.year}년 세금 납부 매도 (세금 ${tax:,.2f})", d, KIND_TAX, trades)
        b["cash"] -= tax

    if signal is not None and signal.get("ok"):  # ⑤ 오늘 월말 신호 → 내일 종가에 교체
        b["pending"] = {"signal_date": d, "targets": [dict(x) for x in signal["targets"]],
                        "off_list": sorted(t for t in b["positions"] if t not in set(universe))}

    for t, p in b["positions"].items():  # ⑥ 평가
        if t in closes:
            p["last_close"] = closes[t]
    value = _equity(b, closes)
    b["peak"] = max(b["peak"], value)
    liq_stock = sum(p["qty"] * p["last_close"] * (1.0 - s) * (1.0 - c) for p in b["positions"].values())
    unreal = liq_stock - sum(p["basis"] for p in b["positions"].values())
    liq = b["cash"] + liq_stock - max(b["realized_ytd"] + unreal, 0.0) * tr
    b["last_date"] = d
    row = {
        "book": b["name"], "date": d, "holding": f"{len(b['positions'])}종목", "qty": len(b["positions"]),
        "cash": b["cash"], "value": value, "cum_return_pct": (value / b["initial_usd"] - 1.0) * 100.0,
        "drawdown_pct": (value / b["peak"] - 1.0) * 100.0, "realized_ytd": b["realized_ytd"], "tax_cum": b["tax_cum"],
        "value_after_tax_liq": liq, "dividends_cum": b["div_cum"], "sgov_value": float("nan"),
    }
    return b, row, trades
