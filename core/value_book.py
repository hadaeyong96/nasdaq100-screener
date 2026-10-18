"""가상 장부 8번 「가치평가 따라하기」 — 하루 처리 순수 함수 (사용자 승인 2026-10-18, docs/prereg/value_score.md 규칙).

네트워크·파일·DB·현재 시각에 접근하지 않는다. engine/picks_books.run_value_score가 DB에서 장부 상태를 읽어
아직 처리하지 않은 거래일마다 process_day를 부르고, 점수(core.value_score로 계산한 월말 점수표)와 그날 종가·환율을 넘긴다.

규칙 (검증 ③ 판정과 같음 — scripts/value_score_zone3_run.py 4e76c43, 사전 등록 4-3):
- 시작: 시작일(설정 고정) 종가에, 그 전 마지막 월말 점수로 49점(100점 표시 70점) 이상 상위 10종목을 1억 원 ÷ 10씩. SIC 2자리 업종 최대 3.
  [사용자 결정 2026-10-18] 1억 원 일시금, 매달 납입 없음.
- 수시 매도: 분기 말 점수가 직전 분기 말과 이번 분기 말 모두 35점(100점 표시 50점) 미만이면 다음 거래일 종가 매도. 첫 매수부터 365일 전이면 안 함.
  점수가 없는 분기(판정 불가)는 연속을 끊는다. 판 돈은 현금(빈자리).
- 12월 실행일(그해 마지막 거래일 2거래일 전, core.tax_harvest.deadline) 종가, 11월 말 점수: ① 갈아타기 1건(56점·10점 차, 수량 = 남은 공제 ÷ 1주당 이익,
  최대 절반) ② 빈자리 채우기(49점 이상) ③ 남은 공제로 공제 소진(팔고 같은 날 다시 삼 — 첫 매수일은 유지).
- 세금(공용 규칙): 그해 원화 실현 순이익 − 250만 원 → 22%, 다음 해 5월 첫 거래일 납부(모자라면 평가액 큰 종목부터 판다).
- 비용: 수수료 편도 0.07%, 슬리피지 편도 0.1%, 환전 0.1%(시작 원화→달러, 세금 달러→원화). 분배금 85% 현금. 분할은 수량 × 비율.
- 상장폐지·인수: 보유 종목 종가가 5거래일 넘게 없으면 마지막 종가로 판다(라이브에서 가격 끝을 아는 방법 — 구현 해석).
- 가상 장부는 정수 주식(다른 가상 장부와 같음). 판정 장부는 소수 주였다(사전 등록 4-1-4) — 라이브 표시용 차이.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import pandas as pd

from core import value_score as vs

SIDE_BUY, SIDE_SELL = "매수", "매도"
K_INIT, K_FILL, K_SWAP_OUT, K_SWAP_IN, K_QTR, K_GONE, K_HARV_OUT, K_HARV_IN, K_TAX = (
    "VAL_INIT", "VAL_FILL", "VAL_SWAP_OUT", "VAL_SWAP_IN", "VAL_QTR_OUT", "VAL_GONE", "VAL_HARVEST_OUT", "VAL_HARVEST_IN", "TAX")
WAITING, RUNNING = "waiting", "running"
GONE_DAYS = 5
DISPLAY = 100.0 / vs.TOTAL  # 70점판 → 100점 표시


@dataclass(frozen=True)
class ValueParams:
    """장부 설정 (퍼센트 값은 % 단위, 점수 문턱은 70점판 — 표시만 100점)."""

    name: str
    slots: int = 10
    min_pick: float = 49.0
    min_keep: float = 35.0
    min_swap: float = 56.0
    swap_gap: float = 10.0
    sector_cap: int = 3
    start_krw: float = 100_000_000.0
    commission_pct: float = 0.07
    slippage_pct: float = 0.1
    fx_spread_pct: float = 0.1
    tax_rate_pct: float = 22.0
    deduction_krw: float = 2_500_000.0
    dividend_withholding_pct: float = 15.0


def display_score(total: float | None) -> float | None:
    """70점판 총점 → 100점 만점 표시 (판정은 70점판 그대로)."""
    return None if total is None else total * DISPLAY


def threshold_text(p: ValueParams) -> str:
    """문턱 표시(100점 만점): "매수 70 · 유지 50 · 갈아타기 80점 이상·약 14.3점 차"."""
    return (f"매수 {p.min_pick * DISPLAY:.0f} · 유지 {p.min_keep * DISPLAY:.0f} · 갈아타기 {p.min_swap * DISPLAY:.0f}점 이상·"
            f"약 {p.swap_gap * DISPLAY:.1f}점 차")


def new_book(name: str, start_date) -> dict:
    """시작 전 장부. 시작일(설정 고정)에 start_book으로 돈을 넣는다."""
    return {"name": name, "status": WAITING, "armed_date": pd.Timestamp(start_date), "start_date": None,
            "start_fx": None, "initial_usd": None, "cash": 0.0, "positions": {}, "realized_krw": {}, "pending_tax": None,
            "tax_cum": 0.0, "tax_cum_krw": 0.0, "div_cum": 0.0, "peak": 0.0, "sell_next": [], "swaps": [],
            "rebalances": 0, "last_date": None}


def start_book(book: dict, d, fx: float, p: ValueParams) -> dict:
    """시작일: 1억 원을 그날 원/달러(0.1% 불리하게)로 달러로."""
    b = copy.deepcopy(book)
    usd = p.start_krw / fx * (1.0 - p.fx_spread_pct / 100.0)
    b.update(status=RUNNING, start_date=pd.Timestamp(d), start_fx=float(fx), initial_usd=usd, cash=usd, peak=usd)
    return b


# ── 점수표 ────────────────────────────────────────────────────────────────


def board(scores: pd.DataFrame | None) -> dict:
    """월말 점수표(index=티커, 열 total·peg_value·universe·sic2) → {order(대상 중 점수 순), total{}, sic2{}}."""
    if scores is None or scores.empty:
        return {"order": [], "total": {}, "sic2": {}}
    u = scores[scores["universe"].astype(bool) & scores["total"].notna()]
    return {"order": vs.rank_order(u[["total", "peg_value"]]),
            "total": {t: (None if pd.isna(v) else float(v)) for t, v in scores["total"].items()},
            "sic2": {t: (None if pd.isna(v) else str(v)) for t, v in scores["sic2"].items()}}


# ── 매매 ─────────────────────────────────────────────────────────────────


def _trade(b, d, side, t, q, px, fee, amount, pnl, reason, kind, signal_date=None, entry=None) -> dict:
    return {"book": b["name"], "signal_date": pd.Timestamp(signal_date) if signal_date is not None else pd.Timestamp(d),
            "fill_date": pd.Timestamp(d), "side": side, "ticker": t, "qty": int(q), "price": float(px), "commission": float(fee),
            "amount": float(amount), "realized_pnl": pnl, "reason": reason, "kind": kind, "in_briefing": None,
            "entry_date": pd.Timestamp(entry) if entry is not None else None}


def _buy(b, d, t, q, close, fx, p: ValueParams, reason, kind, trades, signal_date=None, entry=None) -> None:
    q = int(q)
    if q <= 0:
        return
    px = close * (1 + p.slippage_pct / 100.0)
    fee = q * px * p.commission_pct / 100.0
    need = q * px + fee
    pos = b["positions"].get(t)
    if pos is None:
        pos = b["positions"][t] = {"qty": 0, "cost_krw": 0.0, "first_buy": pd.Timestamp(entry or d), "last_close": close,
                                   "flag": None, "missing": 0}
    pos["qty"] += q
    pos["cost_krw"] += need * fx
    pos["last_close"] = close
    b["cash"] -= need
    trades.append(_trade(b, d, SIDE_BUY, t, q, px, fee, need, None, reason, kind, signal_date, pos["first_buy"]))


def _sell(b, d, t, q, close, fx, p: ValueParams, reason, kind, trades, signal_date=None) -> float:
    pos = b["positions"][t]
    q = int(min(q, pos["qty"]))
    if q <= 0:
        return 0.0
    px = close * (1 - p.slippage_pct / 100.0)
    fee = q * px * p.commission_pct / 100.0
    usd = q * px - fee
    avg = pos["cost_krw"] / pos["qty"]
    gain = usd * fx - avg * q
    y = pd.Timestamp(d).year
    b["realized_krw"][str(y)] = b["realized_krw"].get(str(y), 0.0) + gain
    b["cash"] += usd
    pos["qty"] -= q
    pos["cost_krw"] -= avg * q
    trades.append(_trade(b, d, SIDE_SELL, t, q, px, fee, usd, gain / fx, reason, kind, signal_date, pos["first_buy"]))
    if pos["qty"] <= 0:
        del b["positions"][t]
    return usd


def _per_share_gain(pos, close, fx, p: ValueParams) -> float:
    net = close * (1 - p.slippage_pct / 100.0) * (1 - p.commission_pct / 100.0)
    return net * fx - pos["cost_krw"] / pos["qty"]


def _left(b, y, p: ValueParams) -> float:
    return p.deduction_krw - b["realized_krw"].get(str(y), 0.0)


def _unit(close, p: ValueParams) -> float:
    return close * (1 + p.slippage_pct / 100.0) * (1 + p.commission_pct / 100.0)


def initial_picks(scores: dict, closes: dict, p: ValueParams) -> list[str]:
    """시작일 고를 종목: 49점 이상 대상 중 점수 순, 업종 상한, 그날 종가 있음 — 최대 10."""
    order = [t for t in scores["order"] if (scores["total"].get(t) or 0) >= p.min_pick and t in closes]
    return vs.pick(order, p.slots, scores["sic2"], [], p.sector_cap)


def december(b, d, closes, fx, nov: dict, p: ValueParams, trades) -> dict:
    """12월 실행일: ① 갈아타기 ② 빈자리 ③ 공제 소진. 출력: {"swap": (판 종목, 산 종목, 판 수량) 또는 None, "fills": [..], "harvest": n}."""
    y = pd.Timestamp(d).year
    out = {"swap": None, "fills": [], "harvest": 0}
    held = list(b["positions"])
    held_score = {t: nov["total"].get(t) for t in held if nov["total"].get(t) is not None}
    cands = [t for t in nov["order"] if t not in held and t in closes]
    choice = vs.swap_choice(cands, {t: nov["total"].get(t) for t in cands}, held_score, nov["sic2"], p.min_swap, p.swap_gap,
                            p.sector_cap)
    if choice is not None:
        h, c = choice
        pos = b["positions"][h]
        ch = closes.get(h, pos["last_close"])
        q = math.floor(vs.swap_qty(_left(b, y, p), _per_share_gain(pos, ch, fx, p), pos["qty"]))
        if q > 0:
            usd = _sell(b, d, h, q, ch, fx, p, f"12월 갈아타기 → {c} (공제 한도 안)", K_SWAP_OUT, trades)
            _buy(b, d, c, usd // _unit(closes[c], p), closes[c], fx, p, f"12월 갈아타기 ← {h}", K_SWAP_IN, trades)
            out["swap"] = (h, c, q)
            b["swaps"].append({"date": pd.Timestamp(d), "sell": h, "buy": c, "qty": q})
    vac = p.slots - len(b["positions"])
    if vac > 0:
        order = [t for t in nov["order"] if (nov["total"].get(t) or 0) >= p.min_pick and t in closes]
        picks = vs.pick(order, vac, nov["sic2"], list(b["positions"]), p.sector_cap)
        if picks:
            each = b["cash"] / len(picks)
            for t in picks:
                _buy(b, d, t, each // _unit(closes[t], p), closes[t], fx, p, "12월 빈자리 채우기", K_FILL, trades)
            out["fills"] = picks
    out["harvest"] = harvest(b, d, closes, fx, p, trades)
    b["rebalances"] += 1
    return out


def harvest(b, d, closes, fx, p: ValueParams, trades) -> int:
    """공제 소진: 1주당 이익 + 인 보유를 평가 이익 큰 순으로, 남은 공제 ÷ 1주당 이익(내림)만큼 팔고 같은 날 같은 수량 다시 산다."""
    y = pd.Timestamp(d).year

    def unreal(t):
        pos = b["positions"][t]
        return pos["qty"] * closes.get(t, pos["last_close"]) * fx - pos["cost_krw"]

    n = 0
    for t in sorted(b["positions"], key=lambda k: -unreal(k)):
        left = _left(b, y, p)
        if left <= 1.0:
            break
        pos = b["positions"][t]
        c = closes.get(t, pos["last_close"])
        g = _per_share_gain(pos, c, fx, p)
        if g <= 0:
            continue
        q = min(math.floor(left / g), pos["qty"])
        if q <= 0:
            continue
        first = pos["first_buy"]
        _sell(b, d, t, q, c, fx, p, "12월 공제 소진(같은 날 다시 삼)", K_HARV_OUT, trades)
        _buy(b, d, t, min(q, b["cash"] // _unit(c, p)), c, fx, p, "12월 공제 소진 다시 사기", K_HARV_IN, trades, entry=first)
        if t in b["positions"]:
            b["positions"][t]["first_buy"] = first  # 같은 날 다시 산 것은 새 보유가 아니다(1년 조건 유지)
        n += 1
    return n


def _raise_cash(b, d, closes, fx, p, need_usd, trades) -> None:
    while b["cash"] < need_usd - 1e-9 and b["positions"]:
        t = max(b["positions"], key=lambda k: b["positions"][k]["qty"] * closes.get(k, b["positions"][k]["last_close"]))
        pos = b["positions"][t]
        c = closes.get(t, pos["last_close"])
        net = c * (1 - p.slippage_pct / 100.0) * (1 - p.commission_pct / 100.0)
        q = min(pos["qty"], math.ceil((need_usd - b["cash"]) / net - 1e-9))
        _sell(b, d, t, q, c, fx, p, "세금 납부 매도", K_TAX, trades)


def tax_for_year(b, y, p: ValueParams) -> float:
    """그해 세금(원화): (실현 순이익 − 공제) × 22%, 0 이상."""
    return max(b["realized_krw"].get(str(y), 0.0) - p.deduction_krw, 0.0) * p.tax_rate_pct / 100.0


# ── 하루 ─────────────────────────────────────────────────────────────────


def process_day(book: dict, d, *, closes: dict[str, float], fx: float, dividends: dict[str, float], splits: dict[str, float],
                p: ValueParams, start_scores: dict | None = None, quarter_scores: dict | None = None,
                nov_scores: dict | None = None, is_dec_exec: bool = False, is_may_first: bool = False,
                is_year_end: bool = False, has_next_day: bool = True) -> tuple[dict, dict | None, list[dict]]:
    """하루 처리. 출력: (새 장부, 일별 행(시작 전이면 None), 매매 목록).

    start_scores: 시작일에만(그 전 마지막 월말 점수표 board) / quarter_scores: 분기 말 거래일에만(그날 월말 점수표) /
    nov_scores: 12월 실행일에만(11월 말 점수표). 점수표는 board() 결과.
    """
    d = pd.Timestamp(d)
    b = copy.deepcopy(book)
    trades: list[dict] = []
    if b["status"] == WAITING:
        if d < pd.Timestamp(b["armed_date"]) or start_scores is None:
            return b, None, trades
        b = start_book(b, d, fx, p)
        picks = initial_picks(start_scores, closes, p)
        each = b["cash"] / p.slots
        for t in picks:
            _buy(b, d, t, each // _unit(closes[t], p), closes[t], fx, p, "시작: 점수 상위", K_INIT, trades)
        b["rebalances"] = 1
    else:
        keep = 1 - p.dividend_withholding_pct / 100.0
        for t, pos in list(b["positions"].items()):
            r = splits.get(t, 0.0)
            if r and r > 0 and r != 1.0:  # 분할: 수량 × 비율 (원화 취득가는 그대로)
                pos["qty"] = int(round(pos["qty"] * r))
                pos["last_close"] /= r
            if dividends.get(t, 0.0) > 0:
                usd = pos["qty"] * dividends[t] * keep
                b["cash"] += usd
                b["div_cum"] += usd
            if t in closes:
                pos["missing"] = 0
            else:
                pos["missing"] = pos.get("missing", 0) + 1
                if pos["missing"] > GONE_DAYS:
                    _sell(b, d, t, pos["qty"], pos["last_close"], fx, p, f"가격 끝 {GONE_DAYS}거래일 넘음 → 마지막 종가", K_GONE, trades)
        if is_may_first and b["pending_tax"] is not None:
            y, krw = b["pending_tax"]
            usd = krw / (fx * (1 - p.fx_spread_pct / 100.0))
            _raise_cash(b, d, closes, fx, p, usd, trades)
            b["cash"] -= usd
            b["tax_cum"] += usd
            b["tax_cum_krw"] += krw
            b["pending_tax"] = None
        for t in b.pop("sell_next", []):  # 분기 35점 미만 2번 → 다음 거래일
            pos = b["positions"].get(t)
            if pos is None or t not in closes:
                continue
            if (d - pd.Timestamp(pos["first_buy"])).days >= 365:
                _sell(b, d, t, pos["qty"], closes[t], fx, p, f"분기 점수 {p.min_keep * DISPLAY:.0f}점(100점) 미만 2번 연속",
                      K_QTR, trades)
        b["sell_next"] = []
        if is_dec_exec and nov_scores is not None:
            december(b, d, closes, fx, nov_scores, p, trades)
    if quarter_scores is not None and has_next_day:
        for t, pos in b["positions"].items():
            sc = quarter_scores["total"].get(t)
            if sc is None:
                pos["flag"] = ["unknown", str(pd.Timestamp(pos["first_buy"]).date())]
                continue
            cur = "below" if sc < p.min_keep else "ok"
            key = str(pd.Timestamp(pos["first_buy"]).date())
            if cur == "below" and pos.get("flag") == ["below", key]:
                b["sell_next"].append(t)
            pos["flag"] = [cur, key]
    if is_year_end:
        tax = tax_for_year(b, d.year, p)
        if tax > 0:
            b["pending_tax"] = (d.year, tax)
    for t, pos in b["positions"].items():
        if t in closes:
            pos["last_close"] = closes[t]
    s, c = p.slippage_pct / 100.0, p.commission_pct / 100.0
    value = b["cash"] + sum(pos["qty"] * pos["last_close"] for pos in b["positions"].values())
    b["peak"] = max(b["peak"], value)
    liq_stock = sum(pos["qty"] * pos["last_close"] * (1 - s) * (1 - c) for pos in b["positions"].values())
    unreal_krw = liq_stock * fx - sum(pos["cost_krw"] for pos in b["positions"].values())
    tax_now = max(b["realized_krw"].get(str(d.year), 0.0) + unreal_krw - p.deduction_krw, 0.0) * p.tax_rate_pct / 100.0
    pend = b["pending_tax"][1] if b["pending_tax"] is not None else 0.0
    liq = b["cash"] + liq_stock - (tax_now + pend) / fx
    b["last_date"] = d
    row = {"book": b["name"], "date": d, "holding": f"{len(b['positions'])}종목", "qty": len(b["positions"]), "cash": b["cash"],
           "value": value, "cum_return_pct": (value / b["initial_usd"] - 1.0) * 100.0,
           "drawdown_pct": (value / b["peak"] - 1.0) * 100.0,
           "realized_ytd": b["realized_krw"].get(str(d.year), 0.0) / fx, "tax_cum": b["tax_cum"],
           "value_after_tax_liq": liq, "dividends_cum": b["div_cum"], "sgov_value": float("nan")}
    return b, row, trades
