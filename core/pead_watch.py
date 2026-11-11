"""실적급등 추천 장부(pead_watch) — 순수 함수 (사전 등록: docs/prereg/pead_watch.md, configs/pead_watch.yaml).

네트워크·파일·DB·현재 시각 접근 없음. 입력은 공시 목록·일봉·거래일 달력, 출력은 판정·매매 상태·숫자.
매수·매도 위치 규칙은 core/pead_timing(B1)과 같고, 비용은 core/sw1을 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core import pead_timing as pt
from core import sw1

PRE_MARKET_END = 9 * 60 + 30  # 09:30 (미국 동부)
LABELS = {"both": "둘 다", "ndx": "나스닥100", "sp100": "S&P100"}


@dataclass(frozen=True)
class Params:
    """신호·발표 규칙 (configs/pead_watch.yaml)."""

    ret_min: float = 0.05
    excess_min: float = 0.05
    vol_ratio_min: float = 2.0
    vol_window: int = 20
    min_rows: int = 22
    min_gap_days: int = 60
    ledgers: tuple[tuple[str, int], ...] = (("main", 90), ("compare", 60), ("reference", 20))


def params_from(pre: dict) -> Params:
    """configs/pead_watch.yaml 내용 → Params."""
    s, a, t = pre["signal"], pre["announcement"], pre["trade"]
    return Params(ret_min=float(s["stock_return_min"]), excess_min=float(s["excess_vs_qqq_min"]),
                  vol_ratio_min=float(s["volume_ratio_min"]), vol_window=int(s["volume_window_days"]),
                  min_rows=int(s["min_rows"]), min_gap_days=int(a["min_gap_days"]),
                  ledgers=tuple((k, int(v)) for k, v in t["ledgers"].items()))


# ── 발표·반응일 ─────────────────────────────────────────────────────────────


def is_8k_202(form: str, items: str) -> bool:
    """8-K(정정 8-K/A 제외)이고 Item 2.02가 있으면 True."""
    return form == "8-K" and "2.02" in [x.strip() for x in str(items or "").split(",")]


def counted_announcements(filings: list[dict], min_gap_days: int = 60) -> list[dict]:
    """한 회사의 접수 목록 → 실적 발표로 세는 8-K 2.02 (직전에 센 건으로부터 접수일 min_gap_days일 이상).

    입력: [{form, items, filing_date('YYYY-MM-DD'), accn}] (순서 무관). 출력: 센 건만, 접수일·accn 오름차순.
    """
    k = sorted((f for f in filings if is_8k_202(f["form"], f.get("items", ""))), key=lambda f: (f["filing_date"], f["accn"]))
    out, last = [], None
    for f in k:
        d = pd.Timestamp(f["filing_date"])
        if last is None or (d - last).days >= min_gap_days:
            out.append(f)
            last = d
    return out


def reaction_day(accepted_et: pd.Timestamp, days: pd.DatetimeIndex) -> pd.Timestamp | None:
    """접수 시각(동부, tz 없음) → 반응일. 09:30 전이면 그날(휴장이면 다음 거래일), 아니면 다음 거래일. 달력 밖이면 None."""
    t = pd.Timestamp(accepted_et)
    pre = t.hour * 60 + t.minute < PRE_MARKET_END
    pos = days.searchsorted(t.normalize(), side="left" if pre else "right")
    return days[pos] if pos < len(days) else None


def membership_label(ticker: str, ndx: set[str], sp100: set[str]) -> str | None:
    """명단 소속 → '나스닥100' | 'S&P100' | '둘 다' | None(둘 다 아님)."""
    a, b = ticker in ndx, ticker in sp100
    if a and b:
        return LABELS["both"]
    return LABELS["ndx"] if a else (LABELS["sp100"] if b else None)


def prev_month_end(d) -> pd.Timestamp:
    """날짜 → 그 전 달 말일."""
    return (pd.Timestamp(d).normalize() - pd.offsets.MonthEnd(1)).normalize()


def judge(px: pd.DataFrame, qqq_close: pd.Series, d0, p: Params = Params()) -> dict | None:
    """반응일 d0 신호 판정 (SW1 signal_on과 같은 계산). d0 뒤 행은 쓰지 않는다.

    입력: px(index=거래일, close·volume), qqq_close, d0. 출력: {ret, excess, vol_ratio, signal} 또는 자료 부족이면 None.
    """
    d0 = pd.Timestamp(d0)
    h = px.loc[:d0]
    b = qqq_close.loc[:d0]
    if len(h) < max(p.min_rows, p.vol_window + 2) or h.index[-1] != d0 or len(b) < 2 or b.index[-1] != d0:
        return None
    c0, c1 = float(h["close"].iloc[-1]), float(h["close"].iloc[-2])
    vol_avg = float(h["volume"].iloc[-p.vol_window - 1:-1].mean())
    if not (c1 > 0) or not (vol_avg > 0):
        return None
    ret = c0 / c1 - 1
    bret = float(b.iloc[-1]) / float(b.iloc[-2]) - 1
    ratio = float(h["volume"].iloc[-1]) / vol_avg
    return {"ret": ret, "excess": ret - bret, "vol_ratio": ratio,
            "signal": bool(ret >= p.ret_min and ret - bret >= p.excess_min and ratio >= p.vol_ratio_min)}


# ── 장부 진행 ───────────────────────────────────────────────────────────────


def _d(x) -> str:
    return pd.Timestamp(x).date().isoformat()


def _busy_until(t: dict) -> str | None:
    """같은 종목 새 신호를 막는 끝 날짜(그날 미포함). 안 된 매매는 None."""
    if t["status"] == "closed":
        return t["sell_day"]
    if t["status"] in ("pending", "open"):
        return t["plan_day"]
    return None


def step_day(trades: list[dict], d, new_signals: list[dict], prices: dict[str, pt.Aligned], bench: pt.Aligned,
             days: pd.DatetimeIndex, ledger: str, hold: int) -> tuple[list[dict], list[str]]:
    """한 장부를 거래일 d 하루 진행한다: ① 매수 체결(d 시가) ② 매도(예정일 d 종가) ③ d가 반응일인 새 신호.

    입력: trades(그 장부의 매매 dict 목록, 바꾸지 않음), new_signals([{ticker, d0, label, …}] — 반응일 = d인 신호),
          prices(종목 → 달력 배열, d 뒤 값은 보지 않음), bench(QQQ), days(미래까지 이어진 거래일 달력).
    출력: (새 매매 목록, 경고). 매매 dict: ledger, ticker, d0, label, status('pending'|'open'|'closed'|'skipped'),
          buy_day, plan_day, buy_px, q_buy, sell_day, sell_px, q_sell, skip.
    규칙: 매수일 체결가가 없으면 '가격 공백'으로 빠진다. 예정 매도일에 행이 없으면 매수일~예정일 마지막 종가(pt.exit_point).
          같은 종목: 앞 신호 반응일 ≤ 새 반응일 < 앞 매매 매도일(보유·대기 중이면 예정일)이면 '보유 중'으로 무시.
    """
    i = int(days.get_loc(pd.Timestamp(d)))
    ds = _d(d)
    out, warn = [], []
    for t in trades:
        t = dict(t)
        a = prices.get(t["ticker"])
        if t["status"] == "pending" and t["buy_day"] <= ds:
            ib = int(days.get_loc(pd.Timestamp(t["buy_day"])))
            if a is None:
                warn.append(f"{t['ticker']} 가격 없음 — 매수 체결을 다음 실행으로 미룸")
            else:
                px = a.open[ib]
                if np.isfinite(px) and px > 0 and np.isfinite(bench.open[ib]):
                    t.update(status="open", buy_px=float(px), q_buy=float(bench.open[ib]))
                else:
                    t.update(status="skipped", skip="가격 공백")
        if t["status"] == "open" and t["plan_day"] <= ds:
            ib = int(days.get_loc(pd.Timestamp(t["buy_day"])))
            j = int(days.get_loc(pd.Timestamp(t["plan_day"])))
            if a is None:
                warn.append(f"{t['ticker']} 가격 없음 — 매도를 다음 실행으로 미룸")
            else:
                e = pt.exit_point(a, ib, j)
                if e is None:
                    t.update(status="skipped", skip="가격 공백")
                else:
                    t.update(status="closed", sell_day=_d(days[e]), sell_px=float(a.close[e]), q_sell=float(bench.close[e]))
        out.append(t)
    for s in sorted(new_signals, key=lambda x: x["ticker"]):
        d0 = _d(s["d0"])
        base = {"ledger": ledger, "ticker": s["ticker"], "d0": d0, "label": s.get("label"), "buy_day": None, "plan_day": None,
                "buy_px": None, "q_buy": None, "sell_day": None, "sell_px": None, "q_sell": None, "skip": None}
        busy = [t for t in out if t["ticker"] == s["ticker"] and _busy_until(t) is not None
                and t["d0"] <= d0 < _busy_until(t)]
        if busy:
            out.append({**base, "status": "skipped", "skip": "보유 중"})
            continue
        if i + 1 + hold >= len(days):
            out.append({**base, "status": "skipped", "skip": "달력 끝"})
            continue
        out.append({**base, "status": "pending", "buy_day": _d(days[i + 1]), "plan_day": _d(days[i + 1 + hold])})
    return out, warn


def mark(t: dict, a: pt.Aligned | None, bench: pt.Aligned, days: pd.DatetimeIndex, as_of) -> dict:
    """보유 중 매매의 기준일 평가: 보유 거래일 수, 매수가 대비 %, QQQ 대비 %p(같은 시가 매수 QQQ와 차이, 비용 전)."""
    i = int(days.get_loc(pd.Timestamp(as_of)))
    ib = int(days.get_loc(pd.Timestamp(t["buy_day"])))
    out = {"held": i - ib, "ret": None, "vs_qqq": None, "last_close": None}
    if a is None or t.get("buy_px") is None:
        return out
    e = int(a.lastv[i])
    if e < ib:
        return out
    r = a.close[e] / t["buy_px"] - 1
    q = bench.close[i] / t["q_buy"] - 1 if t.get("q_buy") else np.nan
    out.update(ret=float(r), vs_qqq=float(r - q) if np.isfinite(q) else None, last_close=float(a.close[e]))
    return out


def closed_excess(t: dict, costs: sw1.Costs = sw1.Costs()) -> dict:
    """끝난 매매 → 달러 수익(비용 전)·QQQ 수익·원화 비용 뒤 초과(세금 전). fx_buy·fx_sell 없으면 원화 칸 None."""
    r = t["sell_px"] / t["buy_px"] - 1
    q = t["q_sell"] / t["q_buy"] - 1
    out = {"ret": r, "bench": q, "ex_pre": r - q, "ex_cost": None}
    if t.get("fx_buy") and t.get("fx_sell"):
        out["ex_cost"] = float(sw1.krw_after_cost(r, t["fx_buy"], t["fx_sell"], costs)
                               - sw1.krw_after_cost(q, t["fx_buy"], t["fx_sell"], costs))
    return out


def scorecard(trades: list[dict], costs: sw1.Costs = sw1.Costs()) -> dict:
    """끝난 매매들 → 건수·평균 초과(원화 비용 뒤, 세금 전 — 없으면 달러 비용 전)·승률 (% 단위)."""
    xs = [closed_excess(t, costs) for t in trades if t["status"] == "closed"]
    if not xs:
        return {"n": 0, "mean": None, "win": None, "basis": "cost"}
    use_cost = all(x["ex_cost"] is not None for x in xs)
    v = np.array([x["ex_cost"] if use_cost else x["ex_pre"] for x in xs], dtype=float)
    return {"n": len(v), "mean": float(v.mean() * 100), "win": float((v > 0).mean() * 100), "basis": "cost" if use_cost else "pre"}
