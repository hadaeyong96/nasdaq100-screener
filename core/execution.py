"""체결 모형 (P5-1 3번). 순수 함수 — 그날의 시가·저가만 보고 체결가·체결 여부를 정한다.

- 매수: 신호 다음 날 지정가 = 신호일 종가 × 1.01. 시가 ≤ 지정가면 시가에 체결(더
  유리한 가격). 시가 > 지정가지만 저가 ≤ 지정가면 지정가에 체결. 저가 > 지정가면
  미체결.
- 손절: 사용자는 증권사 손절 예약을 쓰므로 장중 기준이다. 저가 ≤ 손절가면 체결 —
  시가가 이미 손절가 아래로 갭 하락했으면 시가에(더 불리한 실제 체결가), 아니면
  손절가에 체결.
- E1·E2·E3·1차 만료 매도: 신호 다음 날 시가에 체결(항상 체결).
"""

from __future__ import annotations

import pandas as pd


def resolve_buy_fill(limit_price: float, open_: float, low: float) -> float | None:
    """매수 지정가 주문의 체결가를 정한다.

    입력: limit_price(지정가 = 신호일 종가 × 1.01), open_(다음 날 시가), low(다음 날 저가)
    출력: 체결가(달러) 또는 미체결이면 None
    """
    if pd.isna(open_) or pd.isna(low) or pd.isna(limit_price):
        return None
    if open_ <= limit_price:
        return float(open_)
    if low <= limit_price:
        return float(limit_price)
    return None


def resolve_stop_fill(stop_price: float, open_: float, low: float) -> float | None:
    """손절 예약 주문의 체결가를 정한다 (장중 기준, 갭 하락 반영).

    입력: stop_price(손절가), open_(그날 시가), low(그날 저가)
    출력: 체결가(달러) 또는 저가가 손절가 위면 미체결(None)
    """
    if pd.isna(open_) or pd.isna(low) or pd.isna(stop_price):
        return None
    if low > stop_price:
        return None
    return float(open_) if open_ <= stop_price else float(stop_price)


def exit_at_open(open_: float) -> float | None:
    """E1·E2·E3·1차 만료 매도: 다음 날 시가에 그대로 체결한다 (지정가·조건 없음).

    입력: open_(다음 날 시가)
    출력: 체결가(달러) 또는 시가를 모르면 None
    """
    return None if pd.isna(open_) else float(open_)
