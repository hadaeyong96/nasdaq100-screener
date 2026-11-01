"""겨울엔 2배 — 신호·무작위 대조·참고 통계 (docs/prereg/winter_2x.md, 태그 winter2x-prereg).

순수 함수만 둔다(네트워크·파일·현재 시각 접근 없음). 음수 shift를 쓰지 않는다.
비율은 장부(core/shock_dip_perf.run_book)의 X(1배) 비중 wx로 나타낸다: 1배 = 1.0, 1.5배 = 0.5, 2배 = 0.0.
판단일 p의 목표는 p+1 거래일 종가에 체결한다(core/leverage_timing.events_from_monthly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import leverage_timing as lt

W_1X, W_15X, W_2X = lt.W_1X, lt.W_15X, lt.W_2X
SMA_DAYS = 200


def season(start_month: int, length: int) -> frozenset[int]:
    """start_month부터 length개월(12월 다음은 1월). 예: season(11, 6) = {11, 12, 1, 2, 3, 4}."""
    return frozenset(((start_month - 1 + i) % 12) + 1 for i in range(length))


WINTER = season(11, 6)
# 견고성(합격 기준 4): 겨울 구간을 한 달씩 옮긴 4개
ROBUST = {"10~3월": season(10, 6), "10~4월": season(10, 7), "11~3월": season(11, 5), "12~5월": season(12, 6)}


def decision_positions(days: pd.DatetimeIndex, w0: int) -> list[int]:
    """판단일 위치: 창 첫날 w0 + 그 뒤 매달 마지막 거래일(다음 거래일이 다른 달). 마지막 날은 다음 날이 없어 뺀다.

    입력: 거래일 인덱스, 창 시작 위치. 출력: 오름차순 위치 목록(중복 없음).
    """
    pos = lt.month_end_positions(days, w0, len(days))
    if w0 < len(days) - 1 and (not pos or pos[0] != w0):
        pos = [w0] + pos
    return pos


def held_month(days: pd.DatetimeIndex, p: int) -> tuple[int, int]:
    """판단일 p의 목표로 들고 있게 될 달 = 체결일(p+1)의 (연, 월)."""
    nxt = days[p + 1]
    return nxt.year, nxt.month


def h1_targets(days: pd.DatetimeIndex, pos: list[int], winter: frozenset[int] = WINTER) -> list[float]:
    """H1 순수: 체결일의 달이 겨울이면 2배, 아니면 1배. 출력: 판단일별 X 비중."""
    return [W_2X if held_month(days, p)[1] in winter else W_1X for p in pos]


def h2_targets(close: pd.Series, pos: list[int], winter: frozenset[int] = WINTER) -> list[float]:
    """H2: H1 + 판단일 종가가 200일 단순 이동평균 아래(종가 < 이평)면 겨울이라도 1배. 이평이 없으면(NaN) 1배.

    입력: 지수 종가(전체 — 창 전 값도 이평에 씀), 판단일 위치. 출력: 판단일별 X 비중.
    """
    c = close.to_numpy()
    sma = close.rolling(SMA_DAYS).mean().to_numpy()
    out = []
    for p, w in zip(pos, h1_targets(close.index, pos, winter)):
        if w == W_2X and not (np.isfinite(sma[p]) and c[p] >= sma[p]):
            w = W_1X
        out.append(w)
    return out


def events(pos: list[int], targets: list[float]) -> list[tuple[int, float]]:
    """판단일 목표 → 체결 이벤트 [(p+1, X 비중)] (처음 1배, 바뀔 때만)."""
    return lt.events_from_monthly(pos, targets)


# ── 무작위 대조 ─────────────────────────────────────────────────────────────


def random_targets(days: pd.DatetimeIndex, pos: list[int], rng: np.random.Generator) -> list[float]:
    """매년(체결일의 달력 연도) 시작 달 s를 1~12에서 무작위로 뽑아, 그해 s부터 연속 6개월(12월 다음은 같은 해 1월로 돈다)을 2배.

    해마다 2배 달이 정확히 6개 → 달 기준 평균 노출 1.5. 해마다 s = 11이면 H1과 같다. 출력: 판단일별 X 비중.
    """
    years = sorted({held_month(days, p)[0] for p in pos})
    starts = dict(zip(years, rng.integers(1, 13, size=len(years))))
    out = []
    for p in pos:
        y, m = held_month(days, p)
        out.append(W_2X if m in season(int(starts[y]), 6) else W_1X)
    return out


def leverage_of(w: np.ndarray | float) -> np.ndarray | float:
    """X 비중 → 배율(1배 비중 wx면 배율 = wx + 2(1 − wx) = 2 − wx)."""
    return 2.0 - w


# ── 참고 보고 (판정 아님) ────────────────────────────────────────────────────


def half_year_returns(price: pd.Series) -> pd.DataFrame:
    """겨울(전년 10월 말 → 4월 말)·여름(4월 말 → 10월 말) 6개월 수익률. 월말 = 그달 마지막 거래일 종가.

    입력: 가격(창 안만). 출력: DataFrame[연도, 겨울, 여름, 겨울−여름] — 양쪽 끝이 다 있는 반기만(없으면 NaN).
    """
    s = price.dropna()
    me = s.groupby([s.index.year, s.index.month]).tail(1)
    last_day = s.index[-1]
    me = me[~((me.index.year == last_day.year) & (me.index.month == last_day.month))]  # 끝 달은 미완성일 수 있어 뺀다
    at = {(d.year, d.month): float(v) for d, v in me.items()}
    rows = []
    for y in sorted({d.year for d in me.index}):
        w = at[(y, 4)] / at[(y - 1, 10)] - 1 if (y - 1, 10) in at and (y, 4) in at else np.nan
        su = at[(y, 10)] / at[(y, 4)] - 1 if (y, 4) in at and (y, 10) in at else np.nan
        if np.isnan(w) and np.isnan(su):
            continue
        rows.append({"연도": y, "겨울": w, "여름": su, "겨울−여름": w - su})
    return pd.DataFrame(rows)


def lever_spells(nav: np.ndarray, weights: np.ndarray, days: pd.DatetimeIndex) -> pd.DataFrame:
    """2배(X 비중 0) 구간별 손익. 구간 = 2배 체결일 종가 → 빠져나온 체결일 종가(끝까지면 마지막 날).

    입력: 일별 가치·위치별 X 비중(lt.exposure)·날짜(같은 길이). 출력: DataFrame[시작, 끝, 구간 수익률, 구간 안 최대 낙폭].
    """
    rows = []
    n = len(nav)
    p = 0
    while p < n:
        if weights[p] == W_2X:
            a = p
            while p < n and weights[p] == W_2X:
                p += 1
            b = min(p, n - 1)
            seg = nav[a:b + 1]
            peak = np.maximum.accumulate(seg)
            rows.append({"시작": days[a].date(), "끝": days[b].date(), "구간 수익률": float(seg[-1] / seg[0] - 1),
                         "구간 안 최대 낙폭": float((seg / peak - 1).min())})
        else:
            p += 1
    return pd.DataFrame(rows, columns=["시작", "끝", "구간 수익률", "구간 안 최대 낙폭"])
