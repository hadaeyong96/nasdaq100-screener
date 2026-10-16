"""시장 국면 달력 — 순수 함수 (전략공장 F1). 네트워크·파일·현재 시각에 접근하지 않는다.

국면 (그날 종가까지만 사용, 음수 shift 없음):
- 폭락장: 직전 252거래일(그날 포함) 최고 종가 대비 −20% 이하, 또는 최근 20거래일 수익률 −10% 이하 (가장 먼저 본다)
- 하락장: 폭락장이 아니고 종가 < 200일 이동평균
- 상승장: 폭락장이 아니고 종가 ≥ 200일 이동평균
- 판단불가: 200일 이동평균을 만들 데이터(200거래일)가 모자란 초기 구간

해석: 252거래일이 아직 안 찬 구간(200~251번째 날)의 최고가는 그때까지 있는 날로 본다. 20거래일 수익률이 아직 없는
날(처음 20일)은 그 조건만 빼고 본다(어차피 200일 전이라 판단불가).
"""

from __future__ import annotations

import pandas as pd

UP, DOWN, CRASH, UNKNOWN = "상승장", "하락장", "폭락장", "판단불가"
HIGH_DAYS, CRASH_FROM_HIGH = 252, -0.20
FAST_DAYS, CRASH_FAST = 20, -0.10
SMA_DAYS = 200
EPS = 1e-12  # 정확히 −20%·−10%(예: 90/100 − 1 = −0.09999999999999998)도 폭락장으로 보기 위한 부동소수 여유


def classify(qqq_close: pd.Series) -> pd.Series:
    """QQQ 종가 Series(날짜 오름차순) → 같은 인덱스의 국면 Series(상승장|하락장|폭락장|판단불가)."""
    q = qqq_close.astype(float)
    if not q.index.is_monotonic_increasing:
        raise ValueError("날짜가 오름차순이어야 합니다")
    high = q.rolling(HIGH_DAYS, min_periods=1).max()
    from_high = q / high - 1.0
    fast = q / q.shift(FAST_DAYS) - 1.0
    sma = q.rolling(SMA_DAYS, min_periods=SMA_DAYS).mean()
    crash = (from_high <= CRASH_FROM_HIGH + EPS) | (fast <= CRASH_FAST + EPS)
    out = pd.Series(UP, index=q.index, dtype=object)
    out[q < sma] = DOWN
    out[crash] = CRASH
    out[sma.isna()] = UNKNOWN
    return out


def segments(regime: pd.Series) -> pd.DataFrame:
    """국면 Series → 같은 국면이 이어진 구간 표(start, end, regime, days)."""
    key = (regime != regime.shift()).cumsum()
    g = regime.groupby(key)
    return pd.DataFrame({"start": g.apply(lambda s: s.index[0]), "end": g.apply(lambda s: s.index[-1]),
                         "regime": g.first(), "days": g.size()}).reset_index(drop=True)
