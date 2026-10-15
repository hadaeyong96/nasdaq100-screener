"""「QQQ 이기는 힘이 세지는 종목」 점수 — 순수 함수 (작업 6, 사전 등록 2026-10-15 — 임의 변경 금지).

브리핑 관찰 표(6-1)·가상 장부 「한 달 가속 따라하기」(6-2)·구역 ③ 판정(6-3)이 같은 함수를 쓴다.
네트워크·파일·DB·현재 시각에 접근하지 않는다. 미래 데이터 금지: 기준일 이하 값만, 음수 shift 없음.

정의:
- RS선 = 종목 종가 ÷ QQQ 종가 (두 날짜 모두 QQQ 거래일 기준)
- 최근 힘 = RS(t) / RS(t−20) − 1, 직전 힘 = RS(t−20) / RS(t−40) − 1, 점수 = 최근 힘 − 직전 힘
- 대상: 60거래일 이상 가격(QQQ 거래일 기준 t−59일 종가가 있음 — t·t−20·t−40 종가도 있어야 함), 기준일 종가 5달러 이상
- 선택: 점수 상위 10
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AccelParams:
    """매개변수는 사전 등록값 하나뿐 (바꾸지 않는다)."""

    step: int = 20  # 최근·직전 힘 구간(거래일) = 교체 주기
    min_history: int = 60  # 60거래일 이상 가격
    min_price: float = 5.0
    top_n: int = 10


def rank(closes: dict[str, pd.Series], qqq_close: pd.Series, d, universe: list[str], params: AccelParams = AccelParams()) -> dict:
    """기준일 d의 가속 점수 순위.

    입력: closes({티커: 종가 Series(날짜 인덱스)}), qqq_close(QQQ 종가 Series), d(기준일), universe(그날 명단), params
    출력: {"ok": bool, "dates": (t−40, t−20, t) 또는 None,
           "ranked": [{ticker, score, recent, prior, close}] 점수 큰 순(소수, 0.05 = 5%p),
           "targets": 상위 top_n, "excluded": {티커: 사유}}
    """
    d = pd.Timestamp(d)
    q = qqq_close[qqq_close.index <= d].dropna()
    k = int(params.step)
    need = max(2 * k, int(params.min_history) - 1)
    out = {"ok": False, "dates": None, "ranked": [], "targets": [], "excluded": {}}
    if len(q) <= need or q.index[-1] != d:
        return out
    t0, t1, t2 = q.index[-1 - 2 * k], q.index[-1 - k], q.index[-1]
    h0 = q.index[-int(params.min_history)]  # 60거래일째(오늘 포함) 날
    out.update(ok=True, dates=(t0, t1, t2))
    ranked = []
    for t in sorted(set(universe)):
        s = closes.get(t)
        vals = [None if s is None else s.get(x) for x in (h0, t0, t1, t2)]
        if any(v is None or pd.isna(v) or float(v) <= 0 for v in vals):
            out["excluded"][t] = f"{params.min_history}거래일 가격 없음"
            continue
        _, c0, c1, c2 = (float(v) for v in vals)
        if c2 < params.min_price:
            out["excluded"][t] = f"종가 {params.min_price:g}달러 미만"
            continue
        rs0, rs1, rs2 = c0 / float(q[t0]), c1 / float(q[t1]), c2 / float(q[t2])
        recent, prior = rs2 / rs1 - 1.0, rs1 / rs0 - 1.0
        ranked.append({"ticker": t, "score": recent - prior, "recent": recent, "prior": prior, "close": c2})
    ranked.sort(key=lambda r: (-r["score"], r["ticker"]))
    out["ranked"] = ranked
    out["targets"] = ranked[: int(params.top_n)]
    return out


def group_scores(ranked: list[dict], gmap: dict[str, str], top: int = 3, bottom: int = 3, group_min: int = 2) -> tuple[list[dict], list[dict]]:
    """업종 묶음별 점수 평균 상위·하위 (계산된 종목이 group_min 미만인 묶음은 뺀다, 하위는 약한 순, 겹치지 않게).

    입력: ranked(rank()의 ranked), gmap({티커: 묶음}) / 출력: (상위 목록, 하위 목록) — 각 {group, count, avg_score, tickers}
    """
    by: dict[str, list[dict]] = {}
    for r in ranked:
        g = gmap.get(r["ticker"])
        if g:
            by.setdefault(g, []).append(r)
    stats = [{"group": g, "count": len(rs), "avg_score": sum(x["score"] for x in rs) / len(rs), "tickers": [x["ticker"] for x in rs]}
             for g, rs in by.items() if len(rs) >= group_min]
    stats.sort(key=lambda x: (-x["avg_score"], x["group"]))
    rest = stats[top:]
    return stats[:top], (list(reversed(rest[-bottom:])) if rest else [])
