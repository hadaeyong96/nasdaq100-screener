"""가상 장부 10번 「동전주 거래량 수렴 전략」·그림자 「무작위 동전주 20」 — 순수 함수 (docs/prereg/penny_volume_squeeze.md).

네트워크·파일·DB·현재 시각에 접근하지 않는다. engine/penny_book.py가 DB에서 장부 상태를 읽어 아직 처리하지 않은
거래일마다 process_day를 부르고, 그날 종가·원/달러와 (선정 체결일이면) 새 명단을 넘긴다. 실제 주문과 무관하다.

규칙 요약 (사전 등록 `penny-squeeze-prereg` — 임의 변경 금지, 수정 = 새 장부):
- 선정: 선정일 전날(P) 값으로 판정(stats → squeeze_pick / random_pick), 선정일 종가 체결. 1·4·7·10월 첫 거래일마다 다시 고르기.
- 같은 금액: 새 종목 한 개 = 그날 총자산 ÷ 20(첫 선정은 시작 자금 ÷ 20). 겹치는 종목 유지, 남는 돈 BIL.
- +100%: 종가 ≥ 평균 매수 체결가 × 2인 날 → 다음 거래일 종가에 절반 매도(보유 한 번당 1회).
- 가격이 5거래일 연속 없으면 0원. 비용: 동전주 슬리피지 0.5%·수수료 0.07%, BIL 수수료만, 환전 0.1%(시작·세금).
- 세금 공용 규칙: 원화 실현 순이익 − 250만 원 → 22%, 다음 해 5월 첫 거래일 납부, 12월 실행일 공제 소진(팔고 같은 날 다시 삼).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

SIDE_BUY, SIDE_SELL = "매수", "매도"
K_INIT, K_IN, K_OUT, K_HALF, K_GONE, K_HARV_OUT, K_HARV_IN, K_TAX = (
    "PNY_INIT", "PNY_IN", "PNY_OUT", "PNY_HALF", "PNY_GONE", "PNY_HARVEST_OUT", "PNY_HARVEST_IN", "TAX")
WAITING, RUNNING = "waiting", "running"
SELECT_MONTHS = (1, 4, 7, 10)


@dataclass(frozen=True)
class PennyParams:
    """장부 설정 (퍼센트 값은 % 단위). 사전 등록값 그대로 — QQQ 비교(판정)만 slots=1·half_rule=False·cash_ticker=None으로 쓴다."""

    name: str
    slots: int = 20
    min_price: float = 1.0
    max_price: float = 5.0
    min_dollar_volume: float = 1_000_000.0
    min_history: int = 120
    short_days: int = 20
    long_days: int = 120
    max_ratio: float = 0.5
    max_width: float = 0.25
    half_gain: float = 1.0  # +100%
    half_rule: bool = True
    gone_days: int = 5
    start_krw: float = 100_000_000.0
    commission_pct: float = 0.07
    slippage_pct: float = 0.5
    fx_spread_pct: float = 0.1
    tax_rate_pct: float = 22.0
    deduction_krw: float = 2_500_000.0
    dividend_withholding_pct: float = 15.0
    cash_ticker: str | None = "BIL"
    seed: int = 20261021


# ── 고르기 (선정 전날 P 기준) ───────────────────────────────────────────────


def stats(frame: pd.DataFrame | None, p_date, p: PennyParams) -> dict | None:
    """한 종목의 P일 기준 대상 판정·지표. frame: index=날짜, 열 close·volume.

    출력: 대상이면 {close, dollar_volume, ratio(20/120일 평균 거래량, 120일 평균 0이면 None), width(20일 가격 폭)}, 아니면 None
    (P일 행 없음, 가격 기록 120행 미만, 종가 1~5달러 밖, 20일 평균 거래대금 100만 달러 미만)
    """
    if frame is None or frame.empty:
        return None
    d = pd.Timestamp(p_date)
    f = frame[frame.index <= d]
    f = f[f["close"].notna()]
    if f.empty or f.index[-1] != d or len(f) < p.min_history:
        return None
    close = float(f["close"].iat[-1])
    if not (p.min_price <= close <= p.max_price):
        return None
    vol = f["volume"].astype(float).fillna(0.0)
    s = f.iloc[-p.short_days:]
    sv = vol.iloc[-p.short_days:]
    dollar = float((s["close"] * sv).mean())
    if dollar < p.min_dollar_volume:
        return None
    lv = float(vol.iloc[-p.long_days:].mean())
    mean_c = float(s["close"].mean())
    return {"close": close, "dollar_volume": dollar, "ratio": (float(sv.mean()) / lv) if lv > 0 else None,
            "width": (float(s["close"].max()) - float(s["close"].min())) / mean_c if mean_c > 0 else None}


def eligible(frames: dict[str, pd.DataFrame], p_date, p: PennyParams) -> dict[str, dict]:
    """대상 전체 {티커: stats} (1~5달러·거래대금·120거래일 통과)."""
    out = {}
    for t in sorted(frames):
        s = stats(frames[t], p_date, p)
        if s is not None:
            out[t] = s
    return out


def squeeze_pick(elig: dict[str, dict], p: PennyParams) -> list[str]:
    """수렴 비율 ≤ 0.5·가격 폭 ≤ 25% 통과 종목을 수렴 비율 낮은 순(같으면 기호 순) 최대 20개."""
    ok = [(s["ratio"], t) for t, s in elig.items()
          if s["ratio"] is not None and s["width"] is not None and s["ratio"] <= p.max_ratio and s["width"] <= p.max_width]
    return [t for _, t in sorted(ok)][: p.slots]


def random_pick(elig: dict[str, dict], select_date, p: PennyParams) -> list[str]:
    """그림자: 같은 대상에서 무작위 20개 — numpy default_rng([시드, 선정일 YYYYMMDD]), 기호 순 정렬 뒤 중복 없이."""
    pool = sorted(elig)
    if not pool:
        return []
    rng = np.random.default_rng([int(p.seed), int(pd.Timestamp(select_date).strftime("%Y%m%d"))])
    idx = rng.choice(len(pool), size=min(p.slots, len(pool)), replace=False)
    return [pool[i] for i in idx]


def is_select_day(d, prev_trading_day) -> bool:
    """1·4·7·10월 첫 거래일인지 (prev_trading_day: d 직전 NYSE 거래일)."""
    d = pd.Timestamp(d)
    return d.month in SELECT_MONTHS and pd.Timestamp(prev_trading_day).month != d.month


# ── 장부 ─────────────────────────────────────────────────────────────────


def new_book(name: str) -> dict:
    """시작 전 장부. 첫 선정이 성공한 날 start_book으로 돈을 넣는다."""
    return {"name": name, "status": WAITING, "start_date": None, "start_fx": None, "initial_usd": None, "cash": 0.0,
            "positions": {}, "realized_krw": {}, "pending_tax": None, "tax_cum": 0.0, "tax_cum_krw": 0.0, "div_cum": 0.0,
            "peak": 0.0, "pending_half": [], "selections": 0, "last_select": None, "select_due": True, "last_date": None}


def start_book(book: dict, d, fx: float, p: PennyParams) -> dict:
    """시작일: 1억 원을 그날 원/달러(환전 0.1% 불리하게)로 달러로."""
    b = copy.deepcopy(book)
    usd = p.start_krw / fx * (1.0 - p.fx_spread_pct / 100.0)
    b.update(status=RUNNING, start_date=pd.Timestamp(d), start_fx=float(fx), initial_usd=usd, cash=usd, peak=usd)
    return b


def _slip(t: str, p: PennyParams) -> float:
    return 0.0 if t == p.cash_ticker else p.slippage_pct / 100.0


def _trade(b, d, side, t, q, px, fee, amount, pnl, reason, kind, signal_date=None, entry=None) -> dict:
    return {"book": b["name"], "signal_date": pd.Timestamp(signal_date) if signal_date is not None else pd.Timestamp(d),
            "fill_date": pd.Timestamp(d), "side": side, "ticker": t, "qty": int(q), "price": float(px), "commission": float(fee),
            "amount": float(amount), "realized_pnl": pnl, "reason": reason, "kind": kind, "in_briefing": None,
            "entry_date": pd.Timestamp(entry) if entry is not None else None}


def _unit(t: str, close: float, p: PennyParams) -> float:
    return close * (1 + _slip(t, p)) * (1 + p.commission_pct / 100.0)


def _buy(b, d, t, q, close, fx, p: PennyParams, reason, kind, trades, signal_date=None, keep_entry=False) -> None:
    q = int(q)
    if q <= 0:
        return
    px = close * (1 + _slip(t, p))
    fee = q * px * p.commission_pct / 100.0
    need = q * px + fee
    pos = b["positions"].get(t)
    if pos is None:
        pos = b["positions"][t] = {"qty": 0, "cost_krw": 0.0, "buy_usd": 0.0, "entry_px": px, "first_buy": pd.Timestamp(d),
                                   "last_close": close, "missing": 0, "half_done": False}
    if not keep_entry:  # 평균 매수 체결가(+100% 기준) — 공제 소진 다시 사기는 바꾸지 않는다
        pos["entry_px"] = (pos["entry_px"] * pos["qty"] + px * q) / (pos["qty"] + q)
    pos["qty"] += q
    pos["cost_krw"] += need * fx
    pos["last_close"] = close
    b["cash"] -= need
    if t != p.cash_ticker:
        trades.append(_trade(b, d, SIDE_BUY, t, q, px, fee, need, None, reason, kind, signal_date, pos["first_buy"]))


def _sell(b, d, t, q, close, fx, p: PennyParams, reason, kind, trades, signal_date=None) -> float:
    pos = b["positions"][t]
    q = int(min(q, pos["qty"]))
    if q <= 0:
        return 0.0
    px = close * (1 - _slip(t, p))
    fee = q * px * p.commission_pct / 100.0
    usd = q * px - fee
    avg = pos["cost_krw"] / pos["qty"]
    gain = usd * fx - avg * q
    y = str(pd.Timestamp(d).year)
    b["realized_krw"][y] = b["realized_krw"].get(y, 0.0) + gain
    b["cash"] += usd
    pos["qty"] -= q
    pos["cost_krw"] -= avg * q
    if t != p.cash_ticker:
        trades.append(_trade(b, d, SIDE_SELL, t, q, px, fee, usd, gain / fx, reason, kind, signal_date, pos["first_buy"]))
    if pos["qty"] <= 0:
        del b["positions"][t]
    return usd


def _stocks(b: dict, p: PennyParams) -> list[str]:
    return [t for t in b["positions"] if t != p.cash_ticker]


def _equity(b: dict) -> float:
    return b["cash"] + sum(pos["qty"] * pos["last_close"] for pos in b["positions"].values())


def _sweep(b, d, closes, fx, p: PennyParams, trades) -> None:
    """남는 현금 → BIL 정수 주(잔돈은 현금)."""
    t = p.cash_ticker
    if t is None or t not in closes:
        return
    _buy(b, d, t, math.floor(max(b["cash"], 0.0) / _unit(t, closes[t], p) + 1e-9), closes[t], fx, p, "남는 돈 BIL", "BIL", trades)


def _raise_cash(b, d, closes, fx, p: PennyParams, need_usd, trades) -> None:
    """세금 낼 현금: BIL 먼저, 그다음 평가액 큰 종목부터 판다."""
    def order():
        rest = sorted(_stocks(b, p), key=lambda k: -b["positions"][k]["qty"] * closes.get(k, b["positions"][k]["last_close"]))
        return ([p.cash_ticker] if p.cash_ticker in b["positions"] else []) + rest

    for t in order():
        if b["cash"] >= need_usd - 1e-9:
            break
        pos = b["positions"][t]
        c = closes.get(t, pos["last_close"])
        net = c * (1 - _slip(t, p)) * (1 - p.commission_pct / 100.0)
        q = min(pos["qty"], math.ceil((need_usd - b["cash"]) / net - 1e-9))
        _sell(b, d, t, q, c, fx, p, "세금 납부 매도", K_TAX, trades)


def _per_share_gain(t, pos, close, fx, p: PennyParams) -> float:
    net = close * (1 - _slip(t, p)) * (1 - p.commission_pct / 100.0)
    return net * fx - pos["cost_krw"] / pos["qty"]


def harvest(b, d, closes, fx, p: PennyParams, trades) -> int:
    """12월 공제 소진: 평가 이익 큰 동전주부터 남은 공제 ÷ 1주당 이익(내림)만큼 팔고 같은 날 같은 수량 다시 산다."""
    y = str(pd.Timestamp(d).year)

    def unreal(t):
        pos = b["positions"][t]
        return pos["qty"] * closes.get(t, pos["last_close"]) * fx - pos["cost_krw"]

    n = 0
    for t in sorted((t for t in _stocks(b, p) if t in closes), key=lambda k: -unreal(k)):
        left = p.deduction_krw - b["realized_krw"].get(y, 0.0)
        if left <= 1.0:
            break
        pos = b["positions"][t]
        g = _per_share_gain(t, pos, closes[t], fx, p)
        if g <= 0:
            continue
        q = min(math.floor(left / g), pos["qty"])
        if q <= 0:
            continue
        keep = {k: pos[k] for k in ("entry_px", "first_buy", "half_done")}
        _sell(b, d, t, q, closes[t], fx, p, "12월 공제 소진(같은 날 다시 삼)", K_HARV_OUT, trades)
        _buy(b, d, t, min(q, math.floor(max(b["cash"], 0.0) / _unit(t, closes[t], p) + 1e-9)), closes[t], fx, p,
             "12월 공제 소진 다시 사기", K_HARV_IN, trades, keep_entry=True)
        if t in b["positions"]:
            b["positions"][t].update(keep)  # 같은 날 다시 산 것은 새 보유가 아니다(매수가·+100% 1회 기록 유지)
        n += 1
    return n


def tax_for_year(b: dict, y: int, p: PennyParams) -> float:
    """그해 세금(원화): (실현 순이익 − 공제) × 22%, 0 이상."""
    return max(b["realized_krw"].get(str(y), 0.0) - p.deduction_krw, 0.0) * p.tax_rate_pct / 100.0


def _select(b, d, closes, fx, targets: list[str], p: PennyParams, trades, first: bool) -> None:
    """다시 고르기: 빠진 종목 전량 매도 → BIL 전량 매도 → 새 종목 매수(순서대로, 한 개 = 총자산 ÷ 20) → 남는 돈 BIL."""
    for t in _stocks(b, p):
        if t not in targets and t in closes:
            _sell(b, d, t, b["positions"][t]["qty"], closes[t], fx, p, "분기 다시 고르기: 새 명단에 없음", K_OUT, trades)
    per = _equity(b) / p.slots
    if p.cash_ticker in b["positions"] and p.cash_ticker in closes:
        _sell(b, d, p.cash_ticker, b["positions"][p.cash_ticker]["qty"], closes[p.cash_ticker], fx, p, "BIL 정리", "BIL", trades)
    new = [t for t in targets if t not in b["positions"] and t in closes]
    for i, t in enumerate(new):
        amount = min(per, max(b["cash"], 0.0) / (len(new) - i))
        _buy(b, d, t, math.floor(amount / _unit(t, closes[t], p) + 1e-9), closes[t], fx, p,
             "시작: 첫 선정" if first else "분기 다시 고르기: 새 종목", K_INIT if first else K_IN, trades)
    b["selections"] += 1
    b["last_select"] = pd.Timestamp(d)
    b["select_due"] = False


# ── 하루 ─────────────────────────────────────────────────────────────────


def process_day(book: dict, d, *, closes: dict[str, float], fx: float, dividends: dict[str, float], splits: dict[str, float],
                p: PennyParams, targets: list[str] | None = None, is_dec_exec: bool = False, is_may_first: bool = False,
                is_year_end: bool = False) -> tuple[dict, dict | None, list[dict]]:
    """하루 처리. 출력: (새 장부, 일별 행(시작 전이면 None), 매매 목록 — BIL 매매는 넣지 않음).

    closes: 그날 종가 {티커: 값}(보유·새 명단·BIL), targets: 그날 선정 체결일이면 새 명단(순서 = 사는 순서), 아니면 None.
    시작 전 장부는 targets가 있는 날 시작한다.
    """
    d = pd.Timestamp(d)
    b = copy.deepcopy(book)
    trades: list[dict] = []
    if b["status"] == WAITING:
        if targets is None:
            return b, None, trades
        b = start_book(b, d, fx, p)
        _select(b, d, closes, fx, targets, p, trades, first=True)
        _sweep(b, d, closes, fx, p, trades)
    else:
        keep = 1 - p.dividend_withholding_pct / 100.0
        for t, pos in list(b["positions"].items()):
            r = splits.get(t, 0.0)
            if r and r > 0 and r != 1.0:  # 분할: 수량 × 비율(내림), 소수 주는 그날 종가로 현금. 매수가는 비율만큼
                exact = pos["qty"] * r
                q = int(math.floor(exact + 1e-9))
                frac = exact - q
                pos["qty"] = q
                pos["entry_px"] /= r
                pos["last_close"] /= r
                if frac > 1e-9 and t in closes:
                    b["cash"] += frac * closes[t]
                if q <= 0:
                    del b["positions"][t]
                    continue
            if dividends.get(t, 0.0) > 0:
                usd = pos["qty"] * dividends[t] * keep
                b["cash"] += usd
                b["div_cum"] += usd
            if t in closes:
                pos["missing"] = 0
            elif t != p.cash_ticker:
                pos["missing"] = pos.get("missing", 0) + 1
                if pos["missing"] >= p.gone_days:  # 5거래일 연속 가격 없음 → 0원
                    _sell(b, d, t, pos["qty"], 0.0, fx, p, f"가격 {p.gone_days}거래일 연속 없음 → 0원", K_GONE, trades)
        if is_may_first and b["pending_tax"] is not None:
            _, krw = b["pending_tax"]
            usd = krw / (fx * (1 - p.fx_spread_pct / 100.0))
            _raise_cash(b, d, closes, fx, p, usd, trades)
            b["cash"] -= usd
            b["tax_cum"] += usd
            b["tax_cum_krw"] += krw
            b["pending_tax"] = None
        for t, sig in b.pop("pending_half", []):  # 어제 +100% 판정 → 오늘 종가에 절반
            pos = b["positions"].get(t)
            if pos is None or t not in closes or pos["half_done"]:
                continue
            n = pos["qty"] // 2
            if n > 0:
                _sell(b, d, t, n, closes[t], fx, p, f"매수가 대비 +{p.half_gain * 100:.0f}% → 절반 매도", K_HALF, trades, signal_date=sig)
                if t in b["positions"]:
                    b["positions"][t]["half_done"] = True
        b["pending_half"] = []
        if targets is not None:
            _select(b, d, closes, fx, targets, p, trades, first=False)
        if is_dec_exec:
            harvest(b, d, closes, fx, p, trades)
        if trades or targets is not None:
            _sweep(b, d, closes, fx, p, trades)
    for t, pos in b["positions"].items():
        if t in closes:
            pos["last_close"] = closes[t]
    if p.half_rule:  # 오늘 종가 ≥ 매수가 × 2 → 다음 거래일 절반 매도
        b["pending_half"] = [[t, d] for t in _stocks(b, p)
                             if not b["positions"][t]["half_done"] and t in closes
                             and closes[t] >= b["positions"][t]["entry_px"] * (1 + p.half_gain)]
    if is_year_end:
        tax = tax_for_year(b, d.year, p)
        if tax > 0:
            b["pending_tax"] = (d.year, tax)
    return b, _row(b, d, fx, p), trades


def _row(b: dict, d: pd.Timestamp, fx: float, p: PennyParams) -> dict:
    c = p.commission_pct / 100.0
    value = _equity(b)
    b["peak"] = max(b["peak"], value)
    liq_stock = sum(pos["qty"] * pos["last_close"] * (1 - _slip(t, p)) * (1 - c) for t, pos in b["positions"].items())
    unreal_krw = liq_stock * fx - sum(pos["cost_krw"] for pos in b["positions"].values())
    tax_now = max(b["realized_krw"].get(str(d.year), 0.0) + unreal_krw - p.deduction_krw, 0.0) * p.tax_rate_pct / 100.0
    pend = b["pending_tax"][1] if b["pending_tax"] is not None else 0.0
    liq = b["cash"] + liq_stock - (tax_now + pend) / fx
    b["last_date"] = d
    n = len(_stocks(b, p))
    bil = b["positions"].get(p.cash_ticker) if p.cash_ticker else None
    return {"book": b["name"], "date": d, "holding": f"{n}종목", "qty": n, "cash": b["cash"], "value": value,
            "cum_return_pct": (value / b["initial_usd"] - 1.0) * 100.0, "drawdown_pct": (value / b["peak"] - 1.0) * 100.0,
            "realized_ytd": b["realized_krw"].get(str(d.year), 0.0) / fx, "tax_cum": b["tax_cum"],
            "value_after_tax_liq": liq, "dividends_cum": b["div_cum"],
            "sgov_value": bil["qty"] * bil["last_close"] if bil else 0.0}


# ── 알림·판정 ──────────────────────────────────────────────────────────────


def alert_hits(frames: dict[str, pd.DataFrame], tickers: list[str], d, jump: float = 0.30, vol_mult: float = 5.0,
               days: int = 20) -> list[dict]:
    """기준일 d 알림: 하루 수익 ≥ +30% 또는 거래량 ≥ 직전 20거래일(d 제외) 평균 × 5. frames: 열 close·volume.

    출력: [{ticker, ret(소수), vol_x(배수 또는 None), close}] — 기호 순
    """
    d = pd.Timestamp(d)
    out = []
    for t in sorted(set(tickers)):
        f = frames.get(t)
        if f is None or d not in f.index or pd.isna(f.at[d, "close"]):
            continue
        f = f[(f.index <= d) & f["close"].notna()]
        if len(f) < 2:
            continue
        ret = float(f["close"].iat[-1] / f["close"].iat[-2] - 1.0)
        prev = f["volume"].astype(float).iloc[-1 - days:-1]
        avg = float(prev.mean()) if len(prev) else 0.0
        vx = float(f["volume"].iat[-1]) / avg if avg > 0 else None
        if ret >= jump or (vx is not None and vx >= vol_mult):
            out.append({"ticker": t, "ret": ret, "vol_x": vx, "close": float(f["close"].iat[-1])})
    return out


def judge_date(start, trading_days) -> pd.Timestamp | None:
    """판정일: 시작일 다음 해 같은 달·같은 날, 휴장이면 그 뒤 첫 거래일 (trading_days: 정렬된 거래일 목록)."""
    s = pd.Timestamp(start)
    try:
        target = s.replace(year=s.year + 1)
    except ValueError:  # 2월 29일
        target = s.replace(year=s.year + 1, day=28)
    after = [pd.Timestamp(x) for x in trading_days if pd.Timestamp(x) >= target]
    return after[0] if after else None


def after_tax_cum_pct(row: dict, initial_usd: float) -> float:
    """청산 가정 세후 누적(%) = 청산 가정 세후 총자산 ÷ 시작 자금 − 1."""
    return (float(row["value_after_tax_liq"]) / float(initial_usd) - 1.0) * 100.0


def judge(main_cum: float, shadow_cum: float, qqq_cum: float, main_mdd: float, mdd_limit: float = -50.0) -> dict:
    """판정 세 조건 (모두 충족해야 통과). 입력은 세후 누적(%)·최대 낙폭(%)."""
    checks = {"vs_qqq": main_cum > qqq_cum, "vs_random": main_cum > shadow_cum, "mdd": main_mdd > mdd_limit}
    return {**checks, "pass": all(checks.values()), "main_cum": main_cum, "shadow_cum": shadow_cum, "qqq_cum": qqq_cum,
            "main_mdd": main_mdd}


def qqq_compare_params(p: PennyParams, ticker: str = "QQQ") -> PennyParams:
    """판정용 QQQ 비교 장부 설정: 한 종목 일시 매수·보유(슬리피지 0·+100% 규칙 없음·BIL 없음), 비용·세금은 같다."""
    from dataclasses import replace

    return replace(p, name=f"{ticker} 비교(판정용)", slots=1, slippage_pct=0.0, half_rule=False, cash_ticker=None)
