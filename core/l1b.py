"""L1b 새 기간 국면 검증 — 신호·합성 2배·장부 시뮬레이터 (docs/l1b_plan.md, configs/l1b_preregistration.yaml).

순수 함수만 둔다(네트워크·파일·현재 시각 접근 없음). 미래 데이터 금지:
- 음수 shift를 쓰지 않는다. t일 신호는 t일까지의 종가만으로 만든다.
- 일별 후보: t일 종가 신호의 목표 자산은 t+1 거래일 종가에 바꾼다(held_after_close = target.shift(1)).
- 월별 후보: 월말 신호로 그 월말 종가에 바로 바꾼다(계획 문구 "다음 달").
봉인: 1998-12-31 뒤 행이 들어오면 core.regime.SealError로 멈춘다(L1과 같은 검사 함수, 날짜만 다름).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import regime as rg

SEAL_DATE = pd.Timestamp("1998-12-31")
SealError = rg.SealError
ASSET_1X, ASSET_2X, ASSET_CASH, ASSET_HIDIV = "1x", "2x", "cash", "hidiv"


def seal(obj, name: str = "data"):
    """1998-12-31 뒤 행이 있으면 SealError, 없으면 obj 그대로. 입력: Series/DataFrame/dict"""
    return rg.enforce_seal(obj, SEAL_DATE, name)


def cut_and_seal(obj, name: str = "data"):
    """1998-12-31까지 자른 뒤 다시 검사한다. 입력·출력: Series/DataFrame"""
    return seal(rg.truncate_to_seal(obj, SEAL_DATE), name)


# ── 시리즈 만들기 ────────────────────────────────────────────────────────────


def level_from_returns(returns: pd.Series, start_level: float = 1.0) -> pd.Series:
    """일별 수익률(소수)을 첫 행부터 복리 누적한 지수. 첫 행 수익도 반영한다.

    입력: returns(날짜 인덱스) / 출력: 같은 인덱스의 지수 Series
    """
    return start_level * (1.0 + returns).cumprod()


def align_rf(rf: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """프렌치 일별 RF(소수)를 다른 거래일(days)에 맞춘다: (직전 day, 이번 day] 구간 RF의 복리.

    첫 day는 그날 이하 RF 중 그날 것만(없으면 0). 이자 누락·중복이 없다.
    입력: rf(프렌치 날짜 인덱스), days(오름차순) / 출력: Series(인덱스=days)
    """
    days = pd.DatetimeIndex(days)
    log_rf = np.log1p(rf.sort_index())
    cum = log_rf.cumsum()
    at = cum.reindex(cum.index.union(days)).ffill().fillna(0.0).reindex(days)
    first = float(log_rf.get(days[0], 0.0)) if len(days) else 0.0
    diff = at.diff()
    diff.iloc[0] = first
    return np.expm1(diff)


def synthetic_leveraged_returns(r: pd.Series, rf: pd.Series, k: float = 2.0, expense_annual_pct: float = 0.95,
                                divisor: int = 252) -> pd.Series:
    """합성 k배 일간 수익 = k·r − (k−1)·RF − 보수/divisor (행마다, 매일 재조정). −100% 아래는 −100%로 자른다.

    k=2이면 L1b의 2·r − RF − 보수와 같다(L1d에서 k=3으로 일반화).
    입력: r(지수 일간 수익), rf(같은 인덱스 행당 금리), k(배수) / 출력: Series
    """
    out = k * r - (k - 1.0) * rf.reindex(r.index) - expense_annual_pct / 100.0 / divisor
    return out.clip(lower=-1.0)


def synthetic_2x_returns(r: pd.Series, rf: pd.Series, expense_annual_pct: float = 0.95, divisor: int = 252) -> pd.Series:
    """합성 2배 일간 수익 = 2·r − RF − 보수/divisor (synthetic_leveraged_returns k=2). 입력·출력은 위와 같다."""
    return synthetic_leveraged_returns(r, rf, 2.0, expense_annual_pct, divisor)


def compound_by_month(daily: pd.Series) -> pd.Series:
    """일별 수익을 달마다 복리 합산. 인덱스 = 그달 마지막 행 날짜.

    입력: daily(날짜 인덱스) / 출력: Series(인덱스=월말 실제 마지막 거래일)
    """
    g = daily.groupby(daily.index.to_period("M"))
    vals = g.apply(lambda s: float(np.prod(1.0 + s.values) - 1.0))
    last_days = g.apply(lambda s: s.index[-1])
    return pd.Series(vals.values, index=pd.DatetimeIndex(last_days.values))


def month_end_levels(level: pd.Series) -> pd.Series:
    """일별 지수의 월말(그달 마지막 행) 값. 인덱스 = 그 날짜."""
    g = level.groupby(level.index.to_period("M"))
    return pd.Series(g.last().values, index=pd.DatetimeIndex(g.apply(lambda s: s.index[-1]).values))


# ── 일별 신호 ────────────────────────────────────────────────────────────────


def above_sma(level: pd.Series, sma_days: int) -> pd.Series:
    """종가 > sma_days행 단순이동평균이면 1.0, 같거나 아래면 0.0, 평균 미산출이면 NaN.

    입력: level(오름차순) / 출력: float Series
    """
    sma = level.rolling(sma_days).mean()
    out = (level > sma).astype(float)
    out[sma.isna()] = np.nan
    return out


def target_b0(level: pd.Series) -> pd.Series:
    """B0: 늘 1배."""
    return pd.Series(ASSET_1X, index=level.index, dtype=object)


def target_b1(level: pd.Series, sma_days: int = 200) -> pd.Series:
    """B1: 종가 > SMA면 2배, 아니면 국채. SMA 미산출일은 None.

    입력: level, sma_days / 출력: object Series("2x"|"cash"|None)
    """
    above = above_sma(level, sma_days)
    out = pd.Series(None, index=level.index, dtype=object)
    out[above == 1.0] = ASSET_2X
    out[above == 0.0] = ASSET_CASH
    return out


def crash_flag(level: pd.Series, crash_pct: float = -30.0) -> pd.Series:
    """B2 폭락 표시: 최고가(첫 행부터 그날까지) 대비 낙폭 <= crash_pct%면 켬, 켜진 뒤 종가 >= 직전 최고가면 끔.

    입력: level, crash_pct(음수 %) / 출력: bool Series
    """
    vals = level.to_numpy(dtype=float)
    out = np.zeros(len(vals), dtype=bool)
    peak = -np.inf
    flag = False
    for i, v in enumerate(vals):
        prev_peak = peak
        peak = max(peak, v)
        if flag and v >= prev_peak:
            flag = False
        if not flag and (v / peak - 1.0) * 100.0 <= crash_pct:
            flag = True
        out[i] = flag
    return pd.Series(out, index=level.index)


def target_b2(level: pd.Series, sma_days: int = 200, crash_pct: float = -30.0) -> pd.Series:
    """B2: 표시 켜짐 & 종가 > SMA → 2배, 그 외 1배. 표시 켜짐인데 SMA 미산출이면 None.

    입력: level, sma_days, crash_pct / 출력: object Series("2x"|"1x"|None)
    """
    above = above_sma(level, sma_days)
    flag = crash_flag(level, crash_pct)
    out = pd.Series(ASSET_1X, index=level.index, dtype=object)
    out[flag & (above == 1.0)] = ASSET_2X
    out[flag & above.isna()] = None
    return out


def confirmed_side(level: pd.Series, sma_days: int = 200, confirm_days: int = 3) -> pd.Series:
    """B3 상태: 첫 SMA 산출일은 그날의 위/아래, 이후 confirm_days행 연속 반대쪽일 때만 바뀐다.

    입력: level, sma_days, confirm_days / 출력: float Series(1.0 위, 0.0 아래, NaN 미산출)
    """
    above = above_sma(level, sma_days).to_numpy()
    out = np.full(len(above), np.nan)
    state = np.nan
    run_side, run_len = np.nan, 0
    for i, a in enumerate(above):
        if np.isnan(a):
            run_side, run_len = np.nan, 0
            out[i] = state
            continue
        if a == run_side:
            run_len += 1
        else:
            run_side, run_len = a, 1
        if np.isnan(state):
            state = a
        elif a != state and run_len >= confirm_days:
            state = a
        out[i] = state
    return pd.Series(out, index=level.index)


def target_b3(level: pd.Series, sma_days: int = 200, confirm_days: int = 3) -> pd.Series:
    """B3: 확인된 상태가 위면 2배, 아래면 국채, 미산출이면 None."""
    side = confirmed_side(level, sma_days, confirm_days)
    out = pd.Series(None, index=level.index, dtype=object)
    out[side == 1.0] = ASSET_2X
    out[side == 0.0] = ASSET_CASH
    return out


def daily_target(candidate: str, level: pd.Series, sma_days: int = 200, crash_pct: float = -30.0,
                 confirm_days: int = 3) -> pd.Series:
    """후보 이름으로 일별 목표 자산(그날 종가 신호)을 만든다. 입력: candidate B0~B3 / 출력: object Series"""
    if candidate == "B0":
        return target_b0(level)
    if candidate == "B1":
        return target_b1(level, sma_days)
    if candidate == "B2":
        return target_b2(level, sma_days, crash_pct)
    if candidate == "B3":
        return target_b3(level, sma_days, confirm_days)
    raise ValueError(f"알 수 없는 후보: {candidate}")


def held_after_close_daily(target: pd.Series) -> pd.Series:
    """t일 장 마감 후 보유 = t−1일 신호(다음 거래일 종가 체결). 양수 shift만 쓴다."""
    return target.shift(1)


# ── 월별 신호 ────────────────────────────────────────────────────────────────


def target_monthly(month_end_level: pd.Series, avg_months: int = 10, defense: str = ASSET_CASH) -> pd.Series:
    """월말 지수 > 그 달 포함 최근 avg_months개 월말 평균이면 2배, 아니면 defense. 평균 미산출은 None.

    반환값은 그 월말 종가에 바꿔 다음 달 동안 드는 자산(= 그 월말 장 마감 후 보유).
    입력: month_end_level, avg_months, defense("cash"|"hidiv"|"1x") / 출력: object Series
    """
    avg = month_end_level.rolling(avg_months).mean()
    out = pd.Series(None, index=month_end_level.index, dtype=object)
    out[month_end_level > avg] = ASSET_2X
    out[(month_end_level <= avg) & avg.notna()] = defense
    return out


# ── 장부 시뮬레이터 ──────────────────────────────────────────────────────────


def simulate(held: pd.Series, returns: pd.DataFrame, *, cost_pct: float = 0.1, tax_rate: float = 0.22,
             apply_tax: bool = True, initial: float = 100_000.0) -> dict:
    """한 자산 100% 보유 장부 시뮬레이션(일별·월별 공통, 순수 함수).

    입력:
      held: 행마다 "그 행 종가 거래 뒤 보유 자산"(인덱스 = 기간 안 행만, 오름차순, None 금지)
      returns: 같은 인덱스, 열 = 자산 이름("1x","2x","cash","hidiv"), 행 수익(소수). "cash" 열은 이자로 과세.
    규칙(구현 해석 18~25): 첫 행 종가에 held[0] 매수(비용 0.1%). 이후 행마다 보유 자산 수익 반영 → held가 바뀌면
    전량 매도·매수 → 그해 마지막 행이면 세금 납부(순이익 22%, 보유 자산 매도, 그 매도 손익은 다음 해 장부).
    마지막 행: 전량 청산·그해 세금 → after_tax_final. apply_tax=False면 세금 없이 마지막 평가액 = final.
    출력: dict(values, held, switches, taxes{연도:세금}, after_tax_final, pretax_final, years, cagr_pct)
    """
    if held.isna().any():
        raise ValueError("보유 자산이 정해지지 않은 행이 있습니다(워밍업 부족)")
    idx = held.index
    c = cost_pct / 100.0
    years_idx = np.asarray(idx.year)
    is_year_end = np.r_[years_idx[1:] != years_idx[:-1], True]  # 그해 마지막 행(다음 행 연도가 다름)

    asset = held.iloc[0]
    proceeds = initial
    value = proceeds / (1.0 + c)
    basis = proceeds
    income: dict[int, float] = {}
    taxes: dict[int, float] = {}
    values = np.zeros(len(idx))
    switches = 0
    ret = returns.reindex(idx)

    def book(year, amount):
        income[year] = income.get(year, 0.0) + amount

    for i, d in enumerate(idx):
        y = d.year
        if i > 0 and value > 0:
            r = float(ret.at[d, asset])
            if asset == ASSET_CASH:
                interest = value * r
                book(y, interest)
                basis += interest
            value = max(value * (1.0 + r), 0.0)
            want = held.iloc[i]
            if want != asset:
                sale = value * (1.0 - c)
                book(y, sale - basis)
                value, basis, asset = sale / (1.0 + c), sale, want
                switches += 1
        last = i == len(idx) - 1
        if apply_tax and is_year_end[i] and not last:
            tax = tax_rate * max(income.get(y, 0.0), 0.0)
            taxes[y] = tax
            if tax > 0 and value > 0:
                x = min(tax / (1.0 - c), value)
                cost_out = basis * x / value
                book(y + 1, x * (1.0 - c) - cost_out)  # 납부용 매도 손익은 다음 해 장부
                value -= x
                basis -= cost_out
        values[i] = value

    pretax_final = values[-1]
    y_last = idx[-1].year
    sale = values[-1] * (1.0 - c)
    if apply_tax:
        book(y_last, sale - basis)
        final_tax = tax_rate * max(income.get(y_last, 0.0), 0.0)
        taxes[y_last] = taxes.get(y_last, 0.0) + final_tax
        after_tax_final = sale - final_tax
    else:
        after_tax_final = sale
    years = (idx[-1] - idx[0]).days / 365.25
    final_for_cagr = after_tax_final if apply_tax else pretax_final
    cagr = ((final_for_cagr / initial) ** (1.0 / years) - 1.0) * 100.0 if years > 0 and final_for_cagr > 0 else -100.0
    return {
        "values": pd.Series(values, index=idx), "held": held, "switches": switches, "taxes": taxes,
        "after_tax_final": after_tax_final, "pretax_final": pretax_final, "years": years, "cagr_pct": cagr,
    }


# ── 지표 ────────────────────────────────────────────────────────────────────


def max_drawdown(values: pd.Series) -> dict:
    """최대 낙폭(%)과 고점·저점 날짜. 입력: 평가액 Series / 출력: dict(mdd_pct, peak, trough)"""
    dd = values / values.cummax() - 1.0
    trough = dd.idxmin()
    peak = values.loc[:trough].idxmax()
    return {"mdd_pct": float(dd.min() * 100.0), "peak": peak, "trough": trough}


def longest_recovery(values: pd.Series) -> dict:
    """직전 고점을 다시 넘기까지 걸린 행 수의 최댓값. 끝까지 못 넘은 구간이 더 길면 open_ended=True.

    입력: 평가액 Series / 출력: dict(rows, start, end, open_ended)
    """
    v = values.to_numpy()
    peak_i, best = 0, (0, None, None, False)
    for i in range(1, len(v)):
        if v[i] >= v[peak_i]:
            if i - peak_i > best[0]:
                best = (i - peak_i, values.index[peak_i], values.index[i], False)
            peak_i = i
    if len(v) - 1 - peak_i > best[0]:
        best = (len(v) - 1 - peak_i, values.index[peak_i], values.index[-1], True)
    return {"rows": best[0], "start": best[1], "end": best[2], "open_ended": best[3]}


def yearly_returns(values: pd.Series, initial: float = 100_000.0) -> pd.Series:
    """연도별 수익(%): 연말 평가액 / 전년 말(첫해는 initial) − 1. 입력: 평가액 Series / 출력: Series(인덱스=연도)"""
    ye = values.groupby(values.index.year).last()
    prev = ye.shift(1)
    prev.iloc[0] = initial
    return (ye / prev - 1.0) * 100.0
