"""레버리지 비율 조절 1 — 신호·무작위 타이밍 (docs/prereg/leverage_timing.md, 태그 leverage-timing-prereg).

순수 함수만 둔다(네트워크·파일·현재 시각 접근 없음). 음수 shift를 쓰지 않는다.
비율은 장부(core/shock_dip_perf.run_book)의 X(1배) 비중 wx로 나타낸다: 1배 = 1.0, 1.5배 = 0.5, 2배 = 0.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factory import regimes

W_1X, W_15X, W_2X = 1.0, 0.5, 0.0
VARIANTS = ["R1", "R2", "M1", "M2", "M3"]
LEVEL = {"R1": W_15X, "R2": W_2X, "M1": W_15X, "M2": W_15X, "M3": W_15X}


# ── 신호 → 체결 이벤트 ────────────────────────────────────────────────────────


def regime_targets(close: pd.Series, lev_w: float) -> pd.Series:
    """R1·R2: t일 종가까지의 국면(factory.regimes.classify)으로 t일 목표 비중. 상승장 → lev_w, 그 밖(하락·폭락·판단불가) → 1배."""
    reg = regimes.classify(close)
    return pd.Series(np.where(reg == regimes.UP, lev_w, W_1X), index=close.index)


def month_end_positions(days: pd.DatetimeIndex, start_pos: int, end_pos: int) -> list[int]:
    """[start_pos, end_pos) 안의 달마다 마지막 거래일 위치(다음 거래일이 다른 달인 날). 마지막 날(end_pos−1)은 다음 날이 없어 뺀다."""
    out = []
    for p in range(start_pos, min(end_pos, len(days)) - 1):
        if days[p].month != days[p + 1].month:
            out.append(p)
    return out


def events_from_daily_targets(targets: np.ndarray, w0: int) -> list[tuple[int, float]]:
    """일 단위 판단(R): t일 목표가 지금 비중과 다르면 t+1 종가에 맞춘다. 창 첫날 w0는 1배로 시작, w0일 판단부터."""
    ev, state = [], W_1X
    for t in range(w0, len(targets) - 1):
        if targets[t] != state:
            ev.append((t + 1, float(targets[t])))
            state = targets[t]
    return ev


def events_from_monthly(decision_pos: list[int], decisions: list[float]) -> list[tuple[int, float]]:
    """월 단위 판단(M): 판단일 p의 목표가 지금과 다르면 p+1 종가에 맞춘다. 처음은 1배."""
    ev, state = [], W_1X
    for p, w in zip(decision_pos, decisions):
        if w != state:
            ev.append((p + 1, float(w)))
            state = w
    return ev


# ── 공개 시점 기준 값 (발표 전 값 금지) ─────────────────────────────────────────


def last_before(s: pd.Series, day: pd.Timestamp, strict: bool = True) -> float:
    """day 전(strict) 또는 day까지의 가장 최근 값. 없으면 NaN."""
    s = s.dropna()
    if s.empty:
        return np.nan
    s = s[s.index < day] if strict else s[s.index <= day]
    return float(s.iloc[-1]) if len(s) else np.nan


def monthly_known(values: pd.Series, day: pd.Timestamp, lag_months: int) -> pd.Series:
    """월 m 값을 m+lag월 1일부터 안다고 볼 때, day에 알려진 월별 값들(월 시작일 인덱스)."""
    v = values.dropna()
    if v.empty:
        return v
    return v[(pd.DatetimeIndex(v.index) + pd.DateOffset(months=lag_months)) <= day]


def m1_decision(day: pd.Timestamp, t10y2y: pd.Series, gs10: pd.Series, gs2: pd.Series) -> tuple[float, str]:
    """M1: 판단일 전날까지의 T10Y2Y(없으면 GS10 − GS2 월별, 월 m 값은 m+1월 1일부터) ≥ 0 → 1.5배. 출력: (비중, 출처)."""
    v = last_before(t10y2y, day, strict=True)
    src = "T10Y2Y"
    if np.isnan(v):
        g10, g2 = monthly_known(gs10, day, 1), monthly_known(gs2, day, 1)
        both = pd.concat([g10, g2], axis=1, keys=["a", "b"]).dropna()
        if both.empty:
            return np.nan, "없음"
        v = float(both["a"].iloc[-1] - both["b"].iloc[-1])
        src = "GS10−GS2"
    return (W_15X if v >= 0 else W_1X), src


def m2_decision(value: float) -> float:
    """M2: NFCI < 0 → 1.5배 (값은 호출부가 판단일 기준으로 고른다 — 발표본 또는 현재 값). NaN이면 NaN."""
    if np.isnan(value):
        return np.nan
    return W_15X if value < 0 else W_1X


def nfci_current_asof(nfci: pd.Series, day: pd.Timestamp, lag_days: int = 5) -> float:
    """발표본이 없는 기간: 현재 값 중 (관측일 + lag_days) ≤ 판단일인 마지막 주 값. [분석 선택] 주간 공개 시차 5일."""
    s = nfci.dropna()
    s = s[(s.index + pd.Timedelta(days=lag_days)) <= day]
    return float(s.iloc[-1]) if len(s) else np.nan


def m3_decision(day: pd.Timestamp, m2sl: pd.Series) -> float:
    """M3: 판단일에 알려진 마지막 달 L(월 m 값은 m+2월 1일부터)의 전년비 증가율 > L−6개월의 전년비 증가율 → 1.5배."""
    k = monthly_known(m2sl, day, 2)
    if len(k) < 19:
        return np.nan
    yoy = k / k.shift(12) - 1.0
    last = yoy.index[-1]
    prev = last - pd.DateOffset(months=6)
    if prev not in yoy.index or np.isnan(yoy[prev]) or np.isnan(yoy[last]):
        return np.nan
    return W_15X if yoy[last] > yoy[prev] else W_1X


# ── 노출 구간·무작위 타이밍 ──────────────────────────────────────────────────


def exposure(events: list[tuple[int, float]], w0: int, n: int) -> np.ndarray:
    """위치별 보유 비중(체결일 종가부터 새 비중). 길이 n, w0 전은 1배."""
    w = np.full(n, W_1X)
    state, last = W_1X, w0
    for p, x in sorted(events):
        w[last:p] = state
        state, last = x, p
    w[last:] = state
    return w


def spells(levered: list[bool]) -> list[tuple[bool, int]]:
    """참/거짓 열 → [(레버리지 여부, 길이)] 연속 구간."""
    out: list[tuple[bool, int]] = []
    for x in levered:
        if out and out[-1][0] == x:
            out[-1] = (x, out[-1][1] + 1)
        else:
            out.append((x, 1))
    return out


def shuffle_spells(sp: list[tuple[bool, int]], rng: np.random.Generator) -> list[bool]:
    """레버리지 구간·1배 구간 길이 목록을 각각 섞어 번갈아 다시 놓는다(총 길이·레버리지 비율·길이 분포 유지).
    구간 수가 같으면 시작을 무작위로, 다르면 많은 쪽부터."""
    lev = [n for x, n in sp if x]
    one = [n for x, n in sp if not x]
    lev = list(rng.permutation(lev)) if lev else []
    one = list(rng.permutation(one)) if one else []
    if len(lev) > len(one):
        first = True
    elif len(one) > len(lev):
        first = False
    else:
        first = bool(rng.integers(0, 2))
    out, a, b = [], iter(lev), iter(one)
    cur = first
    while True:
        try:
            n = next(a) if cur else next(b)
        except StopIteration:
            break
        out += [cur] * int(n)
        cur = not cur
    rest = list(a) + list(b)
    for n in rest:  # 번갈아 놓고 남은 구간(구간 수 차이 2 이상은 생기지 않지만 방어)
        out += [cur] * int(n)
    return out


def random_daily_events(levered: list[bool], w0: int, lev_w: float, rng: np.random.Generator) -> list[tuple[int, float]]:
    """R 무작위: 창 [w0, …)의 일별 레버리지 여부를 섞은 열에서 바뀌는 날을 체결 이벤트로."""
    seq = shuffle_spells(spells(levered), rng)
    ev, state = [], False
    for i, x in enumerate(seq):
        if x != state:
            ev.append((w0 + i, lev_w if x else W_1X))
            state = x
    return ev


def random_monthly_events(decisions: list[bool], decision_pos: list[int], lev_w: float,
                          rng: np.random.Generator) -> list[tuple[int, float]]:
    """M 무작위: 달별 레버리지 여부를 같은 방식으로 섞어 각 판단일 다음 거래일에 체결."""
    seq = shuffle_spells(spells(decisions), rng)
    return events_from_monthly(decision_pos, [lev_w if x else W_1X for x in seq])
