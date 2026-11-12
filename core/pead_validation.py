"""실적급등 추천 장부 참고 기록·보고 보강 — 순수 함수 (docs/prereg/pead_watch_reporting.md).

1) 포트폴리오(가상 1억, 신호마다 남은 현금 1/10, 남는 현금 = QQQ) 하루 진행 — 앞으로 기록(engine/pead_watch)과
   과거 1회 계산(scripts/pead_validation)이 같은 함수를 쓴다.
2) 다음 실적 발표 전·후 구간 초과.
3) 무작위 비교 요약(백분위·95% 구간)과 복원 추출 95% 구간.
판정·매매 규칙에 쓰지 않는다. 네트워크·파일·현재 시각 접근 없음.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core import pead_timing as pt


@dataclass(frozen=True)
class PfParams:
    """포트폴리오 설정. start_krw·min_buy_krw는 config.yaml pead_watch_portfolio, 수수료는 장부와 같은 0.07%."""

    start_krw: float = 100_000_000.0
    min_buy_krw: float = 1_000_000.0
    fraction: float = 0.1
    commission_pct: float = 0.07
    fx_spread_pct: float = 0.1


def pf_key(t: dict) -> str:
    """매매 → 포트폴리오 열쇠 '종목|반응일'."""
    return f"{t['ticker']}|{pd.Timestamp(t['d0']).date().isoformat()}"


def pf_init(fx_start: float, q_open: float, p: PfParams = PfParams()) -> dict:
    """시작 상태. 1억 원 → 시작일 환율(스프레드 뒤) 달러 → 전부 QQQ 시가. 최소 금액도 같은 환율로 달러 고정.

    출력: {start_usd, min_usd, q_units(현금으로 간주한 QQQ 단위), q100_units(QQQ 100% 비교), pos{열쇠: {ticker, shares, cost}}, done[열쇠]}
    """
    usd = p.start_krw * (1 - p.fx_spread_pct / 100) / fx_start
    return {"start_usd": usd, "min_usd": p.min_buy_krw / fx_start, "q_units": usd / q_open, "q100_units": usd / q_open,
            "pos": {}, "done": []}


def pf_step(state: dict, d, trades: list[dict], q_open: float, q_close: float, marks: dict[str, float],
            p: PfParams = PfParams()) -> tuple[dict, list[dict], dict]:
    """하루 진행: 시가 매수 → 종가 매도 → 종가 평가. state는 바꾸지 않는다.

    입력: trades(주 장부 매매 dict: ticker, d0, status, buy_day, plan_day, buy_px, q_buy, sell_px — 날짜는 'YYYY-MM-DD'),
          q_open·q_close(그날 QQQ), marks(종목 → 그날 또는 그 전 마지막 종가).
    매수 = status 'open'|'closed'이고 buy_day ≤ d이며 아직 처리 안 한 매매(늦게 체결된 매수도 그 매수가·그날 QQQ 시가 q_buy로).
    매도 = status 'closed'이고 plan_day ≤ d이며 보유 중인 매매(장부 매도가). 같은 날 매수는 종목 기호 순, 매번 남은 현금의 1/10.
    출력: (새 상태, 사건[{date, action('buy'|'sell'|'skip'), ticker, d0, shares, px, amount_usd, note}], 하루 줄{…}).
    """
    ds = pd.Timestamp(d).date().isoformat()
    st = {**state, "pos": {k: dict(v) for k, v in state["pos"].items()}, "done": list(state["done"])}
    done = set(st["done"])
    c = p.commission_pct / 100
    ev = []
    buys = sorted((t for t in trades if t["status"] in ("open", "closed") and t.get("buy_px") and t["buy_day"] <= ds
                   and pf_key(t) not in done), key=lambda t: (t["buy_day"], t["ticker"], str(t["d0"])))
    for t in buys:
        k = pf_key(t)
        qo = float(t.get("q_buy") or q_open)
        cash = st["q_units"] * qo
        budget = cash * p.fraction
        px = float(t["buy_px"])
        shares = math.floor(budget / (px * (1 + c))) if px > 0 else 0
        base = {"date": ds, "ticker": t["ticker"], "d0": str(t["d0"])[:10], "px": px}
        if budget < st["min_usd"] or shares <= 0:
            why = "최소 금액 미만" if budget < st["min_usd"] else "1주 값보다 예산이 작음"
            ev.append({**base, "action": "skip", "shares": 0, "amount_usd": budget, "note": why})
        else:
            cost = shares * px * (1 + c)
            st["q_units"] -= cost / qo
            st["pos"][k] = {"ticker": t["ticker"], "shares": shares, "cost": cost}
            ev.append({**base, "action": "buy", "shares": shares, "amount_usd": cost, "note": ""})
        done.add(k)
    for t in sorted(trades, key=lambda t: (str(t.get("plan_day")), t["ticker"])):
        k = pf_key(t)
        if t["status"] != "closed" or k not in st["pos"] or t["plan_day"] > ds:
            continue
        h = st["pos"].pop(k)
        got = h["shares"] * float(t["sell_px"]) * (1 - c)
        st["q_units"] += got / q_close
        ev.append({"date": ds, "action": "sell", "ticker": t["ticker"], "d0": str(t["d0"])[:10], "shares": h["shares"],
                   "px": float(t["sell_px"]), "amount_usd": got, "note": f"손익 {got - h['cost']:+.0f}달러"})
    st["done"] = sorted(done)
    stock = sum(h["shares"] * float(marks.get(h["ticker"], h["cost"] / h["shares"])) for h in st["pos"].values())
    cash = st["q_units"] * q_close
    value = cash + stock
    q100 = st["q100_units"] * q_close
    row = {"date": ds, "value_usd": value, "cash_usd": cash, "stock_usd": stock, "qqq100_usd": q100,
           "vs_qqq_pct": (value / q100 - 1) * 100, "n_pos": len(st["pos"])}
    return st, ev, row


def pf_summary(rows: pd.DataFrame, fx_end: float, p: PfParams = PfParams()) -> dict:
    """하루 줄 표(date, value_usd, qqq100_usd) → 최종 원화 평가액·QQQ 100% 대비·연 환산·최대 낙폭(%)."""
    v, q = rows["value_usd"].to_numpy(float), rows["qqq100_usd"].to_numpy(float)
    yrs = (pd.Timestamp(rows["date"].iloc[-1]) - pd.Timestamp(rows["date"].iloc[0])).days / 365.25
    v0 = v[0] if len(v) else np.nan

    def cagr(x):
        return ((x[-1] / x[0]) ** (1 / yrs) - 1) * 100 if yrs > 0 else np.nan

    def mdd(x):
        return float((x / np.maximum.accumulate(x) - 1).min() * 100)

    return {"days": int(len(v)), "end_krw": float(v[-1] * fx_end), "qqq100_krw": float(q[-1] * fx_end),
            "vs_qqq_pct": float((v[-1] / q[-1] - 1) * 100), "cagr": cagr(v), "qqq_cagr": cagr(q),
            "mdd": mdd(v), "qqq_mdd": mdd(q), "start_usd_value": float(v0),
            "avg_n_pos": float(rows["n_pos"].mean()) if "n_pos" in rows else np.nan,
            "avg_stock_share": float((rows["stock_usd"] / rows["value_usd"]).mean() * 100) if "stock_usd" in rows else np.nan}


def trade_dicts(T: pd.DataFrame, days: pd.DatetimeIndex, bench: pt.Aligned) -> list[dict]:
    """pt.make_trades(B1) 매매 표 → pf_step 입력 dict(모두 끝난 매매, 처리일 = 예정 매도 위치)."""
    out = []
    for r in T.itertuples(index=False):
        ib, j = int(r.ib), int(r.plan)
        out.append({"ticker": r.ticker, "d0": pd.Timestamp(r.d0).date().isoformat(), "status": "closed",
                    "buy_day": days[ib].date().isoformat(), "plan_day": days[j].date().isoformat(),
                    "buy_px": float(r.buy_px), "q_buy": float(bench.open[ib]), "sell_px": float(r.sell_px)})
    return out


def simulate(trades: list[dict], al: dict[str, pt.Aligned], bench: pt.Aligned, days: pd.DatetimeIndex, i_first: int,
             i_last: int, fx_start: float, p: PfParams = PfParams()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """과거 1회 계산: 달력 위치 i_first(시가 시작) ~ i_last 하루씩 pf_step. 출력: (하루 줄 표, 사건 표)."""
    st = pf_init(fx_start, float(bench.open[i_first]), p)
    rows, evs = [], []
    for i in range(i_first, i_last + 1):
        d = days[i]
        held = {h["ticker"] for h in st["pos"].values()} | {t["ticker"] for t in trades if t["buy_day"] == d.date().isoformat()}
        marks = {}
        for t in held:
            a = al.get(t)
            if a is not None and a.lastv[i] >= 0:
                marks[t] = float(a.close[a.lastv[i]])
        qc = float(bench.close[bench.lastv[i]])
        qo = float(bench.open[i]) if np.isfinite(bench.open[i]) else qc
        st, ev, row = pf_step(st, d, trades, qo, qc, marks, p)
        rows.append(row)
        evs += ev
    return pd.DataFrame(rows), pd.DataFrame(evs, columns=["date", "action", "ticker", "d0", "shares", "px", "amount_usd", "note"])


# ── 다음 실적 발표 ─────────────────────────────────────────────────────────


def next_announcement(ann_days: list, buy_day, last_day) -> pd.Timestamp | None:
    """그 종목 실적 발표 반응일 목록 → 매수일 < 반응일 ≤ last_day 중 첫째. 없으면 None."""
    b, e = pd.Timestamp(buy_day), pd.Timestamp(last_day)
    xs = sorted(pd.Timestamp(x) for x in ann_days if b < pd.Timestamp(x) <= e)
    return xs[0] if xs else None


def excess_to(a: pt.Aligned, bench: pt.Aligned, ib: int, i: int, buy_px: float, q_buy: float) -> float | None:
    """매수가(시가) → 위치 i 종가(없으면 그 전 마지막)까지 QQQ 대비 초과(달러, 비용 전)."""
    e = int(a.lastv[i])
    if e < ib or not (buy_px > 0 and q_buy > 0):
        return None
    return float(a.close[e] / buy_px - 1 - (bench.close[bench.lastv[i]] / q_buy - 1))


def split_at_next(a: pt.Aligned, bench: pt.Aligned, ib: int, ie: int, buy_px: float, sell_px: float, i_next: int) -> dict:
    """구간 A(매수일 시가 → 다음 발표 반응일 전날 종가)·B(그 종가 → 매도일 종가)·반응일 하루의 QQQ 대비 초과(달러, 비용 전).

    입력: ib < i_next ≤ ie (달력 위치). 종목 종가가 없는 날은 그 전 마지막 종가.
    """
    k = i_next - 1
    sk = float(a.close[a.lastv[k]])
    qk = float(bench.close[k])
    sn = float(a.close[a.lastv[i_next]])
    qo = float(bench.open[ib])
    exa = (sk / buy_px - 1) - (qk / qo - 1)
    exb = (sell_px / sk - 1) - (float(bench.close[ie]) / qk - 1)
    exd = (sn / sk - 1) - (float(bench.close[i_next]) / qk - 1)
    return {"ex_a": exa, "ex_b": exb, "ex_day": exd, "len_a": int(k - ib + 1), "len_b": int(ie - k)}


# ── 무작위 비교 보고 ─────────────────────────────────────────────────────────


def random_summary(value: float, draws: np.ndarray) -> dict:
    """신호 평균과 무작위 평균들 → 백분위(같은 값 절반)·무작위 중앙값·95% 구간(2.5~97.5 백분위수)."""
    d = np.asarray(draws, dtype=float)
    pct = float(((d < value).sum() + 0.5 * (d == value).sum()) / len(d) * 100.0)
    lo, med, hi = np.percentile(d, [2.5, 50, 97.5])
    return {"n": int(len(d)), "pct": pct, "median": float(med), "lo": float(lo), "hi": float(hi)}


def bootstrap_ci(x: np.ndarray, iters: int, seed: int) -> tuple[float, float]:
    """매매별 값 x를 iters번 복원 추출한 평균의 2.5·97.5 백분위수."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(iters, len(x)))].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(lo), float(hi)
