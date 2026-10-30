"""QQQ 실수 빼기 전략(qqq-minus) — 비중·빼기 규칙 순수 함수 (docs/prereg/qqq_minus.md).

네트워크·파일 없음. 매달 첫 거래일에 그날 알려진 값(시가총액·첫 거래일 경과·주식 수 변화·200일선·매출 성장)만 받는다.
- cap_weights: 시가총액 비중(합 1). 뺀 종목을 제외하고 다시 나누면 = 뺀 비중을 나머지에 시총 비례로 재분배
- drop_reasons: 규칙 A(첫 거래일로부터 504거래일 미만)·B(주식 수 12개월 +5% 초과)·C(참고: 200일선 아래 + 직전 4분기 매출 감소)
- random_drop: 무작위 대조 — 같은 개수, 합계 비중이 비슷한(0.8~1.25배) 무작위 종목 묶음
"""

from __future__ import annotations

import numpy as np

AGE_DAYS = 504
DILUTION = 0.05
RAND_TRIES, RAND_LO, RAND_HI = 2000, 0.8, 1.25


def cap_weights(caps: dict[str, float], drop: set[str] | frozenset[str] = frozenset()) -> dict[str, float]:
    """시가총액 비중. drop 종목을 빼고 남은 종목 시총으로 다시 나눈다(현금 0). 남는 종목이 없으면 빈 dict."""
    keep = {t: v for t, v in caps.items() if t not in drop and v > 0}
    tot = sum(keep.values())
    return {t: v / tot for t, v in keep.items()} if tot > 0 else {}


def drop_reasons(age_ok: bool | None, share_g: float | None, below_sma: bool | None = None,
                 rev_g: float | None = None) -> set[str]:
    """빼기 이유 집합. A: age_ok가 False(첫 거래일로부터 504거래일 미만이 확실). B: 주식 수 변화 > +5%.
    C(참고판): 200일선 아래 그리고 직전 4분기 매출 전년 대비 < 0. 모르는 값(None)이면 그 규칙으로는 빼지 않는다."""
    out = set()
    if age_ok is False:
        out.add("A")
    if share_g is not None and share_g > DILUTION:
        out.add("B")
    if below_sma and rev_g is not None and rev_g < 0:
        out.add("C")
    return out


def random_drop(weights: dict[str, float], k: int, w_target: float, rng: np.random.Generator) -> set[str]:
    """무작위 대조: 대상(weights의 종목) 중 k개를 무작위로 뽑되, 합계 비중이 w_target의 0.8~1.25배인 첫 묶음.
    RAND_TRIES번 안에 없으면 합계 비중이 로그 기준으로 가장 가까운 묶음. k = 0이면 빈 집합."""
    if k <= 0 or not weights:
        return set()
    names = sorted(weights)
    w = np.array([weights[t] for t in names])
    k = min(k, len(names))
    best, best_gap = None, np.inf
    for _ in range(RAND_TRIES):
        idx = rng.choice(len(names), size=k, replace=False)
        s = float(w[idx].sum())
        if w_target <= 0 or RAND_LO * w_target <= s <= RAND_HI * w_target:
            return {names[i] for i in idx}
        gap = abs(np.log(s / w_target)) if s > 0 else np.inf
        if gap < best_gap:
            best, best_gap = idx, gap
    return {names[i] for i in best}
