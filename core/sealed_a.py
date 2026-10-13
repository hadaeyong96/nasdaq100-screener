"""봉인 ⓐ 개봉 (docs/sealed_a_plan.md, configs/sealed_a.yaml) — B1 금리 환산·판정 순수 함수.

신호·합성 2배·장부는 core/l1b.py를 그대로 쓴다. 여기에는 봉인 ⓐ에서 새로 정한 것만 둔다.
미래 데이터 금지: t일 RF는 t−1 거래일까지 알려진 DTB3만 쓴다(양수 shift만).
"""

from __future__ import annotations

import pandas as pd

SEAL_LAST = pd.Timestamp("2026-09-30")


class SealError(RuntimeError):
    """봉인 마지막 날 뒤 행이 들어왔을 때."""


def check_seal(obj, name: str = "data", last: pd.Timestamp = SEAL_LAST):
    """last 뒤 행이 있으면 SealError(잘라내지 않음), 없으면 obj 그대로. 입력: Series/DataFrame"""
    if len(obj) and pd.DatetimeIndex(obj.index).max() > last:
        raise SealError(f"{name}: {last.date()} 뒤 행 {int((pd.DatetimeIndex(obj.index) > last).sum())}개 — 멈춤")
    return obj


def dtb3_daily_rf(dtb3_pct: pd.Series, days: pd.DatetimeIndex, day_count: int = 360) -> pd.Series:
    """DTB3 연율(%) → 거래일 t의 RF(소수) = 직전 거래일까지 알려진 DTB3/100 × (t − 직전 거래일) 달력 일수/360.

    첫 거래일은 0 (장부가 첫 행 수익을 쓰지 않음). 직전 거래일 전에 알려진 값이 없으면 NaN.
    입력: dtb3_pct(날짜 인덱스, 결측 제외), days(오름차순 거래일) / 출력: Series(인덱스=days)
    """
    days = pd.DatetimeIndex(days)
    s = dtb3_pct.dropna().sort_index()
    known = s.reindex(s.index.union(days)).ffill().reindex(days)  # 그날까지 알려진 값
    prev_rate = known.shift(1)
    gap = pd.Series(days, index=days).diff().dt.days
    out = prev_rate / 100.0 * gap / day_count
    out.iloc[0] = 0.0
    return out


def b1_verdict(full: dict, subs: dict, neighbors: dict) -> dict:
    """봉인 ⓐ B1 판정 ①~④ (반올림 전 비교).

    입력: full{"B0","B1"}·subs{구간: {"B0","B1"}}·neighbors{sma: posttax}의 값은 dict(posttax, mdd)
    출력: dict(pass, ok{1..4}, c1, c2, c3{구간: bool}, c4{sma: 차이})
    """
    c1 = full["B1"]["posttax"] - full["B0"]["posttax"]
    c2 = full["B1"]["mdd"] - full["B0"]["mdd"]
    c3 = {k: bool(v["B1"]["posttax"] > v["B0"]["posttax"]) for k, v in subs.items()}
    c4 = {k: v - full["B1"]["posttax"] for k, v in neighbors.items()}
    ok = {1: c1 >= 2.0, 2: c2 >= -2.0, 3: bool(c3) and all(c3.values()), 4: bool(c4) and all(abs(d) <= 1.0 for d in c4.values())}
    return {"pass": all(ok.values()), "ok": ok, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


def window_report(values: pd.Series, held: pd.Series, b0_values: pd.Series, start, end) -> dict:
    """구간(예: 2022 약세장) 보고: 보유 비율, 전환일, 구간 손실(구간 시작 전날 대비), 구간 안 최대 낙폭, B0 손실.

    입력: 세후 장부 평가액·보유 자산(같은 인덱스), B0 평가액, start·end / 출력: dict
    """
    ws, we = pd.Timestamp(start), pd.Timestamp(end)
    pos0 = max(values.index.searchsorted(ws) - 1, 0)
    win_h = held.loc[ws:we]
    span_v = values.iloc[pos0:].loc[:we]
    chg = held.iloc[pos0:].loc[:we]
    prev = chg.shift(1)
    moves = chg[(chg != prev) & prev.notna()]
    b_pos0 = max(b0_values.index.searchsorted(ws) - 1, 0)
    b_span = b0_values.iloc[b_pos0:].loc[:we]
    dd = span_v / span_v.cummax() - 1.0
    b_dd = b_span / b_span.cummax() - 1.0
    return {
        "share": {a: float((win_h == a).mean() * 100) for a in ("1x", "2x", "cash")},
        "moves": [(d, a) for d, a in moves.items()],
        "loss_pct": float((span_v.iloc[-1] / span_v.iloc[0] - 1) * 100),
        "mdd_pct": float(dd.min() * 100),
        "b0_loss_pct": float((b_span.iloc[-1] / b_span.iloc[0] - 1) * 100),
        "b0_mdd_pct": float(b_dd.min() * 100),
    }

