"""가상 장부용 ETF 일봉·분배금 받기 (docs/design/virtual_books.md).

data/prices.fetch_one은 history_days(300거래일)만 받는데, 가상 장부는 시작일이 고정이라
시간이 갈수록 더 긴 구간(시작일 − 200일선 준비 기간 ~ 오늘)이 필요하다 — 그래서 여기서
start를 직접 지정해 받는다. 가격 기준은 같다: auto_adjust=False의 Close(분할 반영, 배당 미반영).
분배금은 같은 호출의 Dividends 열(배당락일 기준 주당 금액)을 쓴다.

확정 봉만: 실행 시각(미국 동부) 기준 가장 최근에 마감된 거래일보다 뒤의 행은 버린다.
네트워크 실패는 조용히 넘기지 않는다 — 실패 종목을 모아 VirtualBooksDataError로 올린다.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from data.market_calendar import latest_closed_trading_day, trading_days_between
from data.prices import US_EASTERN

WARMUP_CALENDAR_DAYS = 420  # 200거래일 이동평균 + 여유 (달력일)


class VirtualBooksDataError(RuntimeError):
    """필요한 종목 가격을 받지 못했을 때 (실패 종목·사유 포함)."""


def _yf_history(ticker: str, start: date) -> pd.DataFrame:
    import yfinance as yf

    return yf.Ticker(ticker).history(start=start.isoformat(), auto_adjust=False, actions=True)


def fetch_closes_and_dividends(
    tickers: list[str], start: date, now_et: datetime | None = None, history_fn=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """종목별 확정 종가·분배금을 받아 신호 종목(첫 번째) 거래일에 맞춘다.

    입력: tickers(첫 번째가 신호 종목 — 인덱스 기준), start(받기 시작일), now_et(테스트용 시각),
         history_fn(ticker, start) -> yfinance history 형식 DataFrame(테스트 주입)
    출력: (closes, dividends) — 둘 다 index=신호 종목 거래일(tz-naive), columns=tickers.
         맨 끝 종가가 빈 행은 잘라내고 그 경고를 closes.attrs["warnings"]에 담는다
    예외: 한 종목이라도 실패하면 VirtualBooksDataError(실패 목록)
    """
    history_fn = history_fn or _yf_history
    last_ok = latest_closed_trading_day(now_et or datetime.now(US_EASTERN))
    closes, divs, failed = {}, {}, []
    for t in tickers:
        try:
            raw = history_fn(t, start)
            if raw is None or raw.empty or "Close" not in raw.columns:
                raise ValueError("가격 데이터 없음")
            idx = pd.DatetimeIndex(raw.index)
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            raw = raw.set_axis(idx.normalize())
            raw = raw[raw.index.date <= last_ok]
            raw = raw[~raw.index.duplicated(keep="last")]
            if raw.empty:
                raise ValueError("확정된 가격 데이터 없음")
            closes[t] = raw["Close"].astype(float)
            divs[t] = (raw["Dividends"] if "Dividends" in raw.columns else pd.Series(0.0, index=raw.index)).astype(float)
        except Exception as exc:  # 조용히 넘기지 않는다 — 모아서 올린다
            failed.append(f"{t}: {type(exc).__name__}: {exc}")
    if failed:
        raise VirtualBooksDataError("가상 장부 가격 받기 실패 — " + "; ".join(failed))
    base_index = closes[tickers[0]].index
    close_df = pd.DataFrame({t: closes[t].reindex(base_index) for t in tickers})
    div_df = pd.DataFrame({t: divs[t].reindex(base_index).fillna(0.0) for t in tickers})
    close_df, div_df, warnings = trim_trailing_empty_closes(close_df, div_df)
    close_df.attrs["warnings"] = warnings
    return close_df, div_df


def trim_trailing_empty_closes(
    close_df: pd.DataFrame, div_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """맨 끝에서부터 종가가 하나라도 빈 행을 잘라낸다 (중간 날짜의 빈 칸은 그대로 둔다).

    마감 직후 yfinance가 그날 행을 종가 없이 주는 경우가 있다(2026-10-15 예약 실행, 10-14 행).
    장부는 매 실행 시작일부터 다시 계산하므로 하루 늦게 반영해도 다음 실행에서 따라잡는다.
    중간 날짜가 비면 여기서 채우지 않는다 — core가 멈춘다(조용히 채우지 않음).

    입력: close_df, div_df(같은 인덱스)
    출력: (잘라낸 close_df, 같은 날을 잘라낸 div_df, 경고 문구 목록)
    예외: 다 잘려 남는 행이 없으면 VirtualBooksDataError
    """
    complete = close_df.notna().all(axis=1)
    keep = len(close_df)
    while keep and not complete.iloc[keep - 1]:
        keep -= 1
    if keep == len(close_df):
        return close_df, div_df, []
    dropped = close_df.index[keep:]
    if keep == 0:
        raise VirtualBooksDataError("가상 장부 가격 받기 실패 — 종가가 있는 날이 없습니다")
    empty = {d: [t for t in close_df.columns if pd.isna(close_df.at[d, t])] for d in dropped}
    detail = ", ".join(f"{d.date()}({'·'.join(ts)})" for d, ts in empty.items())
    warning = (f"맨 끝 종가가 빈 날 {len(dropped)}일을 잘라 {close_df.index[keep - 1].date()} 기준으로 계산합니다: "
               f"{detail} — 다음 실행에서 따라잡습니다")
    return close_df.iloc[:keep].copy(), div_df.iloc[:keep].copy(), [warning]


def year_end_trading_days(first_year: int, last_year: int) -> set:
    """각 해의 마지막 NYSE 거래일 Timestamp 집합 (세금 정산일)."""
    out = set()
    for y in range(first_year, last_year + 1):
        days = trading_days_between(date(y, 12, 1), date(y, 12, 31))
        if len(days):
            out.add(pd.Timestamp(days[-1]))
    return out


def fetch_actions(tickers: list[str], start: date, history_fn=None) -> dict[str, pd.DataFrame]:
    """보유 종목의 분배금·분할 (추천 따라하기 장부 — 하루씩 이어 가므로 그날 일어난 일만 반영한다).

    입력: tickers, start(받기 시작일), history_fn(ticker, start) -> yfinance history 형식(테스트 주입)
    출력: {티커: DataFrame(index=날짜(tz-naive), dividend=주당 분배금, split=분할 비율(없으면 0))}
    예외: 한 종목이라도 실패하면 VirtualBooksDataError(실패 목록) — 조용히 넘기지 않는다
    """
    history_fn = history_fn or _yf_history
    out, failed = {}, []
    for t in tickers:
        try:
            raw = history_fn(t, start)
            if raw is None or raw.empty:
                raise ValueError("데이터 없음")
            idx = pd.DatetimeIndex(raw.index)
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            raw = raw.set_axis(idx.normalize())
            div = raw["Dividends"] if "Dividends" in raw.columns else pd.Series(0.0, index=raw.index)
            spl = raw["Stock Splits"] if "Stock Splits" in raw.columns else pd.Series(0.0, index=raw.index)
            out[t] = pd.DataFrame({"dividend": div.astype(float).fillna(0.0), "split": spl.astype(float).fillna(0.0)})
        except Exception as exc:
            failed.append(f"{t}: {type(exc).__name__}: {exc}")
    if failed:
        raise VirtualBooksDataError("가상 장부 분배금·분할 받기 실패 — " + "; ".join(failed))
    return out
