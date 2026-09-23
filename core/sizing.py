"""정리본 6장(포지션 크기: 2% 룰의 단계별 배분)의 수량 계산 순수 함수 모음.

수량 = 내림( 계좌 × 단계 위험 예산 ÷ 주당 위험 )
주당 위험 = (매수가 − 손절가) + 매수가 × 갭 여유
"""

from __future__ import annotations

import math

import pandas as pd

_BUDGET_KEY = {
    "A1": "a1_budget_pct",
    "A2": "a2_budget_pct",
    "A3": "a3_budget_pct",
    "B": "b_budget_pct",
}


def per_share_risk(entry_price: float, stop_price: float, cfg: dict) -> float:
    """주당 위험 = (매수가 − 손절가) + 매수가 × 갭 여유. 매수가의 최소 비율 미만이면 올린다."""
    risk_cfg = cfg["risk"]
    raw = (entry_price - stop_price) + entry_price * risk_cfg["gap_buffer_pct"] / 100
    min_risk = entry_price * risk_cfg["min_risk_per_share_pct"] / 100
    return max(raw, min_risk)

def budget_usd(stage: str, equity_usd: float, cfg: dict) -> float:
    """단계별 위험 예산(달러) = 계좌 × 단계 위험 예산 비율."""
    pct = cfg["risk"][_BUDGET_KEY[stage]]
    return equity_usd * pct / 100


def position_size(stage: str, entry_price: float, stop_price: float, equity_usd: float, cfg: dict) -> int:
    """단계별 추천 수량 = 내림(위험 예산 ÷ 주당 위험).

    입력값에 NaN이 있거나 주당 위험이 0 이하이면 0주.
    """
    if pd.isna(entry_price) or pd.isna(stop_price) or entry_price <= 0:
        return 0
    risk_per_share = per_share_risk(entry_price, stop_price, cfg)
    if risk_per_share <= 0:
        return 0
    budget = budget_usd(stage, equity_usd, cfg)
    return max(int(math.floor(budget / risk_per_share)), 0)


def cap_qty_by_position_limit(qty: int, entry_price: float, equity_usd: float, cfg: dict) -> int:
    """종목당 투입 금액 한도(계좌 대비 max_position_pct)를 넘지 않게 수량을 줄인다."""
    if qty <= 0 or entry_price <= 0:
        return 0
    max_value = equity_usd * cfg["risk"]["max_position_pct"] / 100
    max_qty_by_value = int(math.floor(max_value / entry_price))
    return max(min(qty, max_qty_by_value), 0)


def stop_price_a1_a2(df: pd.DataFrame, a1_date) -> float:
    """A1·A2 손절가: A1 발생일 기준 swing_low."""
    return float(df.loc[a1_date, "swing_low"])


def stop_price_a3(df: pd.DataFrame, a1_date, a3_date) -> float:
    """A3 손절가: max(A1 발생일 기준 swing_low, A3 확정일 구름 하단)."""
    a1_stop = stop_price_a1_a2(df, a1_date)
    cloud_bot = df.loc[a3_date, "cloud_bot"]
    if pd.isna(cloud_bot):
        return a1_stop
    return max(a1_stop, float(cloud_bot))


def stop_price_b(df: pd.DataFrame, entry_date) -> float:
    """B형 손절가: 진입일 기준 swing_low."""
    return float(df.loc[entry_date, "swing_low"])
