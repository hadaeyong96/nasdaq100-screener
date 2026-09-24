"""해외주식 양도소득세·배당 원천징수 계산 (P5-1 3번). 순수 함수.

- 양도세: 연간(원화 기준) 실현손익 합계에서 250만 원 공제 후 22%. 손실이거나
  공제 이하면 0. 다음 해 5월에 납부한 것으로 처리한다(engine/backtest.py가 그
  시점에 현금에서 뺀다 — 이 모듈은 세액 계산만 한다).
- 배당: 15% 원천징수 후 재투자.
"""

from __future__ import annotations

DEFAULT_DEDUCTION_KRW = 2_500_000
DEFAULT_CAPITAL_GAINS_RATE = 0.22
DEFAULT_DIVIDEND_WITHHOLDING_RATE = 0.15


def capital_gains_tax(
    realized_gain_krw: float, deduction_krw: float = DEFAULT_DEDUCTION_KRW, rate: float = DEFAULT_CAPITAL_GAINS_RATE
) -> float:
    """연간 실현손익(원화, 매도 원화금액 − 매수 원화금액 − 비용의 합)에 대한 양도세.

    입력: realized_gain_krw(그해 실현손익 합계, 음수 가능), deduction_krw(기본 250만 원),
         rate(기본 22%)
    출력: 세액(원). 실현손익이 공제 이하(손실 포함)면 0.
    """
    taxable = max(realized_gain_krw - deduction_krw, 0.0)
    return taxable * rate


def dividend_after_withholding(gross_dividend_krw: float, rate: float = DEFAULT_DIVIDEND_WITHHOLDING_RATE) -> float:
    """배당 원천징수(기본 15%) 후 재투자할 실수령액.

    입력: gross_dividend_krw(세전 배당금, 원화), rate
    출력: 세후 배당금(원)
    """
    return gross_dividend_krw * (1 - rate)
