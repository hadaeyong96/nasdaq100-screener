"""L1 레버리지 국면 전환 — 국면 신호와 목표 비중 (docs/l1_plan.md, configs/l1_preregistration.yaml).

순수 함수만 둔다(네트워크·파일·현재 시각 접근 없음). 미래 데이터 금지:
- 음수 shift를 쓰지 않는다. 그날 값은 그날까지 확정된 데이터만으로 만든다.
- 여기서 내는 비중은 "그날 종가에 정한 목표"다. 체결은 다음 거래일 시가에
  engine/portfolio.py simulate_target_weights가 한다.
- 월별 거시 지표(UNRATE)는 M월 값을 M+2월 첫 거래일부터 쓴다(발표 지연 보수 규칙).
"""

from __future__ import annotations

import pandas as pd

LAST_ALLOWED_DATE = pd.Timestamp("2021-12-31")  # 봉인: 이 날짜 뒤 데이터는 읽지도 계산하지도 않는다


class SealError(RuntimeError):
    """봉인 구간(2022-01-01 이후) 데이터가 들어왔을 때."""


def enforce_seal(obj, last_date=LAST_ALLOWED_DATE, name: str = "data"):
    """DataFrame/Series(날짜 인덱스)나 {"YYYY-MM-DD": 값} dict에 last_date 뒤 행이 있으면 SealError.

    입력: obj, last_date, name(에러 메시지용) / 출력: obj 그대로(검사만)
    """
    last = pd.Timestamp(last_date)
    if isinstance(obj, dict):
        keys = [pd.Timestamp(k) for k in obj]
        bad = [k for k in keys if k > last]
    elif len(obj) == 0:
        bad = []
    else:
        idx = pd.DatetimeIndex(obj.index)
        bad = list(idx[idx > last])
    if bad:
        raise SealError(f"봉인 위반: {name}에 {last.date()} 이후 행 {len(bad)}개 (첫 행 {min(bad).date()})")
    return obj


def truncate_to_seal(obj, last_date=LAST_ALLOWED_DATE):
    """last_date까지만 남긴다(dict·DataFrame·Series). 잘라낸 뒤 enforce_seal로 다시 확인하는 것은 호출부 몫."""
    last = pd.Timestamp(last_date)
    if isinstance(obj, dict):
        return {k: v for k, v in obj.items() if pd.Timestamp(k) <= last}
    if len(obj) == 0:
        return obj
    return obj.loc[pd.DatetimeIndex(obj.index) <= last]


def above_sma(close: pd.Series, sma_days: int) -> pd.Series:
    """종가 > sma_days일 단순이동평균이면 True, 같거나 아래면 False, 평균을 못 구한 날은 NaN.

    입력: close(날짜 인덱스, 오름차순), sma_days / 출력: 같은 인덱스의 object Series(True/False/NaN)
    """
    sma = close.rolling(sma_days).mean()
    out = (close > sma).astype(object)
    out[sma.isna()] = float("nan")
    return out


def monthly_release_dates(months: pd.DatetimeIndex, trading_days: pd.DatetimeIndex, lag_months: int = 2) -> pd.Series:
    """월 값(M월)을 처음 쓸 수 있는 날 = M+lag_months월의 첫 거래일.

    입력: months(각 월의 1일), trading_days(오름차순 거래일), lag_months(기본 2)
    출력: Series(인덱스=month, 값=사용 시작 거래일). M+lag_months월 1일 이후 첫 거래일이며,
         그날이 trading_days 시작보다 이르면(이미 발표된 과거 월) 첫 거래일부터 쓸 수 있다.
         trading_days 끝보다 늦게 발표되면 NaT.
    """
    td = pd.DatetimeIndex(trading_days)
    out = {}
    for m in months:
        first_day = (pd.Timestamp(m) + pd.DateOffset(months=lag_months)).replace(day=1)
        pos = td.searchsorted(first_day)
        out[pd.Timestamp(m)] = td[pos] if pos < len(td) else pd.NaT
    return pd.Series(out)


def monthly_above_avg(monthly: pd.Series, trading_days: pd.DatetimeIndex, n_months: int, lag_months: int = 2) -> pd.Series:
    """거래일마다 "그날 쓸 수 있는 최근 월 값 > 최근 n_months개 월 값의 평균"인지.

    입력: monthly(인덱스=월 1일, 값=월별 지표, 예: UNRATE), trading_days, n_months(10/12/14), lag_months
    출력: Series(인덱스=trading_days, True/False). 쓸 수 있는 값이 n_months개 미만이면 False.
    """
    monthly = monthly.dropna().sort_index()
    release = monthly_release_dates(monthly.index, trading_days, lag_months)
    td = pd.DatetimeIndex(trading_days)
    out = pd.Series(False, index=td, dtype=bool)
    # 각 월 값이 풀리는 날 이후로만 쓸 수 있다 — 풀린 순서대로 상태를 갱신한다.
    events = sorted((d, m) for m, d in release.items() if not pd.isna(d))
    available: list = []
    state = False
    j = 0
    for day in td:
        while j < len(events) and events[j][0] <= day:
            available.append(events[j][1])
            j += 1
            vals = monthly.loc[sorted(available)].iloc[-n_months:]
            state = bool(len(vals) >= n_months and vals.iloc[-1] > vals.mean())
        out[day] = state
    return out


def lagged_daily(series: pd.Series, trading_days: pd.DatetimeIndex, lag_days: int = 1) -> pd.Series:
    """일별 지표(T10Y2Y·BAA10Y 등)를 거래일에 맞추고 lag_days거래일 늦춰 쓴다(직전 값 채움, 양수 shift만)."""
    s = series.sort_index().reindex(pd.DatetimeIndex(trading_days), method="ffill")
    return s.shift(lag_days)


def target_weights(candidate: str, above: pd.Series, unrate_up: pd.Series | None = None) -> pd.DataFrame:
    """후보별 날짜마다 목표 비중(core·qld·reserve, 합 1)을 정한다 (그날 종가 기준 목표).

    입력: candidate(L0~L4), above(above_sma 결과 — NaN이면 "아래 아님"으로 보지 않고 공격 유지),
         unrate_up(monthly_above_avg 결과, L3·L4에 필요)
    출력: DataFrame(인덱스=above.index, 열=core, qld, reserve, regime("공격"|"방어"))
    """
    idx = above.index
    attack = {
        "L0": (1.0, 0.0, 0.0), "L1": (0.0, 1.0, 0.0), "L2": (0.0, 1.0, 0.0),
        "L3": (0.0, 1.0, 0.0), "L4": (0.5, 0.5, 0.0),
    }[candidate]
    defense = (0.0, 0.0, 1.0)
    below = above.map(lambda v: v is False or v == False)  # noqa: E712 — NaN(평균 미산출)은 아래가 아니다
    if candidate in ("L0", "L1"):
        is_def = pd.Series(False, index=idx)
    elif candidate == "L2":
        is_def = below.astype(bool)
    elif candidate in ("L3", "L4"):
        if unrate_up is None:
            raise ValueError(f"{candidate}에는 실업률 조건이 필요합니다")
        is_def = below.astype(bool) & unrate_up.reindex(idx).fillna(False).astype(bool)
    else:
        raise ValueError(f"알 수 없는 후보: {candidate}")
    rows = [defense if d else attack for d in is_def]
    out = pd.DataFrame(rows, index=idx, columns=["core", "qld", "reserve"])
    out["regime"] = ["방어" if d else "공격" for d in is_def]
    return out
