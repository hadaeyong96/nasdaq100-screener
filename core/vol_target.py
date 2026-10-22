"""변동성 맞춤 레버리지 전략 — 신호·구성·무작위 대조 순수 함수 (docs/prereg/vol_target.md, 태그 vol-target-prereg).

자산 순서: 0 현금(국채), 1 1배, 2 2배. 음수 shift 없음 — 판단일 값은 그날까지의 가격만 쓴다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BETA = np.array([0.0, 1.0, 2.0])  # 자산별 노출
CASH, ONE, TWO = 0, 1, 2


def sigma(close: pd.Series, days: int = 20) -> pd.Series:
    """최근 days개 일간 수익률 표본 표준편차(ddof=1) × √252 (판단일 포함)."""
    return close.pct_change().rolling(days).std(ddof=1) * np.sqrt(252.0)


def weights_of(e: float) -> np.ndarray:
    """노출 e → [현금, 1배, 2배] 비중. e ≤ 1: 1배 e + 현금 1−e, 1 < e ≤ 2: 2배 e−1 + 1배 2−e."""
    e = float(min(max(e, 0.0), 2.0))
    if e <= 1.0:
        return np.array([1.0 - e, e, 0.0])
    return np.array([0.0, 2.0 - e, e - 1.0])


def exposure(w: np.ndarray) -> float:
    """비중 → 노출."""
    return float(np.dot(BETA, w))


def week_end_positions(dates: pd.DatetimeIndex, w0: int) -> list[int]:
    """w0 이상 위치 중 그 주(월~금)의 마지막 거래일(다음 날이 다른 주이거나 마지막 날)."""
    wk = dates.to_period("W-SUN")
    return [p for p in range(w0, len(dates)) if p == len(dates) - 1 or wk[p + 1] != wk[p]]


def target_exposures(close: pd.Series, positions: list[int], target_vol: float = 0.20, cap_below_sma: bool = False,
                     sma_days: int = 200) -> list[float]:
    """판단 위치마다 e = min(2, target_vol ÷ σ) (V2면 종가 < 200일선일 때 e ≤ 1). σ가 없거나 0이면 1."""
    sig = sigma(close).to_numpy()
    sma = close.rolling(sma_days).mean().to_numpy()
    c = close.to_numpy()
    out = []
    for p in positions:
        s = sig[p]
        e = 1.0 if (np.isnan(s) or s <= 0) else min(2.0, target_vol / s)
        if cap_below_sma and not np.isnan(sma[p]) and c[p] < sma[p]:
            e = min(e, 1.0)
        out.append(e)
    return out


def should_rebalance(threshold: float = 0.2):
    """현재 노출과 목표 노출 차이 ≥ threshold면 참."""
    def f(cur: np.ndarray, tgt: np.ndarray) -> bool:
        return abs(exposure(cur) - exposure(tgt)) >= threshold - 1e-12
    return f


def block_shuffle(values: list[float], rng: np.random.Generator, block: int = 4) -> list[float]:
    """앞에서부터 block개씩 묶은 블록(마지막은 남는 만큼)의 순서를 섞는다. 블록 안 순서는 그대로."""
    blocks = [values[i:i + block] for i in range(0, len(values), block)]
    order = rng.permutation(len(blocks))
    return [v for j in order for v in blocks[j]]
