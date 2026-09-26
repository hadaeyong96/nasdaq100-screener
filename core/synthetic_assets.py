"""합성 자산 계산 (P6-1 1번). 순수 함수 — 네트워크·현재 시각 접근 없음.

- 국채 일율 변환: FRED DTB3(연율, %, 액면할인수익률 관행상 360일 기준)을 일별
  단리 수익률로 바꾼다. 채권시장 휴일(주식시장과 다름)은 직전 값을 그대로
  끌어와 채운다(공휴일 처리, P6-1 1-2번).
- 합성 QLD: 2006-06 이전 구간을 "QQQ 일간 수익 × 2 − 연 0.95%/252 − DTB3 일율"로
  만들고(docs/p6_1_instructions.md 1-3번 그대로, 임의로 다른 식을 쓰지 않는다),
  실제 QLD 첫날 종가에 이어 붙인다.
"""

from __future__ import annotations

import pandas as pd


def annual_yield_pct_to_daily_rate(annual_pct: pd.Series, day_count: int = 360) -> pd.Series:
    """연율(%, 예: FRED DTB3)을 일별 단리 수익률로 바꾼다.

    입력: annual_pct(인덱스=날짜, 값=연율 %), day_count(액면할인수익률 관행 360일)
    출력: 같은 인덱스의 일별 수익률(비율, 예: 5.0(%) -> 5.0/100/360)
    """
    return annual_pct / 100 / day_count


def align_rate_to_trading_days(rate_map: dict[str, float], trading_days: list) -> pd.Series:
    """채권시장 발표일과 주식시장 거래일이 달라 생기는 빈 날을 직전 값으로 채운다.

    입력: rate_map({"YYYY-MM-DD": 값, ...}), trading_days(pd.Timestamp 목록, 오름차순)
    출력: trading_days를 인덱스로 하는 pd.Series. trading_days 이전에 알려진 값이
         하나도 없으면 그 날짜는 NaN(호출부가 시작일을 늦추는 등으로 처리).
    """
    s = pd.Series(rate_map, dtype=float)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    idx = pd.DatetimeIndex(trading_days)
    return s.reindex(idx, method="ffill")


def synthesize_qld_daily_returns(
    qqq_close: pd.Series, daily_borrow_rate: pd.Series, annual_expense_pct: float = 0.95, leverage: float = 2.0
) -> pd.Series:
    """QLD 합성 일간 수익률 = QQQ 일간수익 × leverage − annual_expense_pct/100/252 − daily_borrow_rate.

    입력: qqq_close(날짜 인덱스, QQQ 종가), daily_borrow_rate(같은 인덱스로 정렬된
         일별 차입 비용 비율 — align_rate_to_trading_days 결과), annual_expense_pct(연 보수 %),
         leverage(배율, 기본 2.0)
    출력: pd.Series(날짜 인덱스, qqq_close.index[1:]와 같음 — 첫날은 전일 종가가 없어 계산 불가)
    """
    qqq_ret = qqq_close.pct_change()
    expense_daily = annual_expense_pct / 100 / 252
    out = qqq_ret * leverage - expense_daily - daily_borrow_rate
    return out.dropna()


def levels_from_daily_returns(daily_returns: pd.Series, start_level: float = 1.0) -> pd.Series:
    """일별 수익률을 누적해 가격 수준(지수) 시계열로 만든다.

    입력: daily_returns(날짜 인덱스), start_level(첫 수익률 적용 전 수준)
    출력: pd.Series(같은 인덱스, level[i] = start_level * prod_{k<=i}(1+r_k))
    """
    return start_level * (1 + daily_returns).cumprod()


def splice_synthetic_returns_before_real(synthetic_returns: pd.Series, real_df: pd.DataFrame) -> pd.DataFrame:
    """synthetic_returns(실제 자산 상장 전 구간의 합성 일간수익률)를 real_df(실제 자산,
    open/high/low/close) 앞에 이어 붙인다 (순수 함수, P6-1 1-3번).

    실제 자산 첫날 종가를 기준(수준=1)으로 그 이전 날짜를 거꾸로 나눠 가며 수준을
    구한다 — 접합 경계에서 가격이 어긋나지 않는다(engine.backtest.splice_pre_inception_series와
    같은 목적, 계산식만 "수익률 누적"으로 다르다). 합성 구간은 종가만 알 수 있으므로
    open=high=low=close로 둔다(문서 미기재 구간의 근사, 보고서에 표시할 것).

    입력: synthetic_returns(날짜 인덱스, real_df.index[0] 이전 구간만), real_df(open,high,low,close,
         index[0]이 접합 경계)
    출력: 합성 구간(수준을 맞춘) + real_df를 이어 붙인 DataFrame, 날짜 오름차순, 중복 없음
    """
    if synthetic_returns.empty:
        return real_df.copy()
    real_start = real_df.index[0]
    pre_returns = synthetic_returns.loc[synthetic_returns.index < real_start]
    if pre_returns.empty:
        return real_df.copy()

    real_start_close = float(real_df.loc[real_start, "close"])
    # 접합 경계 바로 다음날의 수준을 1로 두고 거꾸로(최근->과거) 나눠 가며 수준을 구한다.
    reversed_returns = pre_returns.iloc[::-1]
    level_reversed = (1.0 / (1 + reversed_returns)).cumprod()
    level = level_reversed.iloc[::-1] * real_start_close
    pre_df = pd.DataFrame({"open": level, "high": level, "low": level, "close": level})
    out = pd.concat([pre_df, real_df])
    assert out.index.is_monotonic_increasing and not out.index.duplicated().any()
    return out
