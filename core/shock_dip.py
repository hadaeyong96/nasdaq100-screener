"""폭락 연구 관문 2 — 실시간 신호 재구성 (docs/prereg/shock_dip.md v2).

순수 함수만 둔다(네트워크·파일·현재 시각 접근 없음). 음수 shift를 쓰지 않는다.
여기서는 진입 신호일·보유 상태만 만든다 — 손익(수익률)은 계산하지 않는다(관문 3에서, 승인 후).

용어: pos = 거래일 위치(정수). 신호일 t의 판단은 t일 종가까지, 체결은 t+1 거래일 종가.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

VARIANTS = ["A", "B1", "B2", "B3", "B4", "D"]


# ── 데이터 보정 ─────────────────────────────────────────────────────────────


def correct_with_reference(close: pd.Series, ref: pd.Series, max_diff: float = 0.02) -> tuple[pd.Series, list]:
    """일간 수익률이 기준 지수와 max_diff 넘게 다른 날은 기준 지수 일간 수익률로 바꿔 종가를 다시 쌓는다 (v2-5).

    입력: close(QQQ 종가), ref(^NDX 종가). 기준 값이 없는 날은 원래 수익률을 쓴다.
    출력: (보정 종가 Series — 첫날은 원래 종가, 대체한 날짜 list)
    """
    r = close.pct_change()
    rr = ref.reindex(close.index).pct_change()
    bad = (r - rr).abs() > max_diff
    bad &= rr.notna()
    r2 = r.where(~bad, rr)
    r2.iloc[0] = 0.0
    out = close.iloc[0] * (1.0 + r2).cumprod()
    return out.rename(close.name), list(close.index[bad.to_numpy()])


# ── 공개일 기준 as-of (발표 전 값 금지) ─────────────────────────────────────


def asof_daily(values: pd.Series, available: pd.Series, days: pd.DatetimeIndex, strict: bool) -> pd.Series:
    """각 거래일에 공개된 가장 최근 관측값.

    입력: values(관측 시점 인덱스), available(같은 인덱스의 공개 시각 Timestamp), days(거래일),
    strict=True면 공개일 < 거래일(해외 시장: 미국 발표가 그날 장 마감 뒤), False면 ≤.
    출력: days 인덱스 Series(아직 공개된 값이 없으면 NaN). 공개일이 같으면 늦은 관측이 이긴다.
    """
    df = pd.DataFrame({"v": values.to_numpy(dtype=float), "avail": pd.to_datetime(available.reindex(values.index)).to_numpy(),
                       "obs": values.index})
    df = df.dropna(subset=["v", "avail"]).sort_values(["avail", "obs"])
    left = pd.DataFrame({"day": days})
    m = pd.merge_asof(left, df, left_on="day", right_on="avail", allow_exact_matches=not strict)
    return pd.Series(m["v"].to_numpy(), index=days)


def asof_percentile(values: pd.Series, available: pd.Series, days: pd.DatetimeIndex, strict: bool,
                    since: pd.Timestamp | None = None) -> pd.Series:
    """각 거래일에 공개된 값들(since 이후 관측) 중 최신 값 이하 비율(0~1).

    공개 순서대로 k번째 값이 나올 때의 백분위 = (0..k번째 값 ≤ k번째 값) 비율 — 이때 알려진 값만 쓴다.
    출력: days 인덱스 Series(없으면 NaN).
    """
    v = values.dropna()
    if since is not None:
        v = v[v.index >= since]
    av = pd.to_datetime(available.reindex(v.index))
    order = pd.DataFrame({"v": v.to_numpy(), "avail": av.to_numpy(), "obs": v.index}).dropna().sort_values(["avail", "obs"])
    arr = order["v"].to_numpy()
    pct = np.array([(arr[: k + 1] <= arr[k]).mean() for k in range(len(arr))])
    s = pd.Series(pct, index=pd.Index(order["obs"]))
    return asof_daily(s, pd.Series(order["avail"].to_numpy(), index=pd.Index(order["obs"])), days, strict)


def month_start_available(months: pd.Index, lag_months: int) -> pd.Series:
    """월 m(월 시작일 인덱스) 값을 m+lag_months월 1일부터 쓴다. 출력: 관측 인덱스 → 공개 시작 Timestamp.

    "m+1월 첫 거래일부터" = 공개 시작일 1일 이상인 첫 거래일 — asof_daily(strict=False)와 함께 쓰면 같다.
    """
    idx = pd.DatetimeIndex(months)
    return pd.Series(idx + pd.DateOffset(months=lag_months), index=idx)


def release_after_month_end(months: pd.Index, release_dates: pd.DatetimeIndex) -> pd.Series:
    """월 m 값 공개일 근사 = 월 m 말일 뒤 첫 발표일(예: UNRATE 빈티지). 없으면 NaT."""
    rel = pd.DatetimeIndex(sorted(release_dates))
    out = {}
    for m in pd.DatetimeIndex(months):
        later = rel[rel > m + pd.offsets.MonthEnd(0)]
        out[m] = later[0] if len(later) else pd.NaT
    return pd.Series(out)


# ── 가격 지표 ─────────────────────────────────────────────────────────────


def price_features(close: pd.Series) -> pd.DataFrame:
    """종가에서 신호용 지표. 모두 그날까지의 값만 쓴다.

    출력 열: close, ret20(20거래일 수익률), sma200, above200_20ago(20거래일 전 종가 > 그날 200일선),
    sma20, cross20_up(종가가 20일선 위로 올라온 날: 오늘 > 20일선, 어제 ≤ 20일선), high252(오늘 포함 252거래일 최고 종가),
    above200(오늘 종가 > 200일선)
    """
    c = close.astype(float)
    sma200 = c.rolling(200).mean()
    sma20 = c.rolling(20).mean()
    above20 = c > sma20
    prev_above20 = above20.shift(1)
    cross = above20 & sma20.shift(1).notna() & sma20.notna() & (prev_above20 == False)  # noqa: E712
    return pd.DataFrame({
        "close": c,
        "ret20": c / c.shift(20) - 1.0,
        "sma200": sma200,
        "above200_20ago": (c.shift(20) > sma200.shift(20)) & sma200.shift(20).notna(),
        "sma20": sma20,
        "cross20_up": cross,
        "high252": c.rolling(252).max(),
        "above200": (c > sma200) & sma200.notna(),
    })


def base_condition(f: pd.DataFrame, sahm: pd.Series) -> pd.Series:
    """A의 진입 조건 ①②③(t일): 20거래일 수익률 ≤ −10%, 20거래일 전 종가 > 그날 200일선, 샴 < 0.5. NaN이면 거짓."""
    return (f["ret20"] <= -0.10) & f["above200_20ago"] & (sahm < 0.5)


# ── 상태 기계 (진입 신호일·청산 신호일만, 손익 없음) ───────────────────────


@dataclass
class Trade:
    signal_pos: int  # 진입 신호일 t
    entry_pos: int  # 체결일 t+1
    exit_signal_pos: int | None  # 청산 판단일(B3은 None)
    exit_pos: int | None  # 청산 체결일(데이터 끝까지 보유면 None)
    reason: str  # high / sahm / time / below200 / fixed20 / open


def _run_exit_a(f: pd.DataFrame, sahm: np.ndarray, signal_pos: int, entry_pos: int, hold_days: int = 60) -> tuple:
    """A식 청산: 진입일 종가부터 매 종가에 판단 — 종가 ≥ 진입 직전 252거래일 최고가(신호일까지),
    샴 ≥ 0.5, 진입 후 hold_days거래일 경과. 하나라도 성립한 날 u → u+1 체결."""
    c = f["close"].to_numpy()
    target = c[max(0, signal_pos - 251): signal_pos + 1].max()
    n = len(c)
    for u in range(entry_pos, n):
        if c[u] >= target:
            reason = "high"
        elif not np.isnan(sahm[u]) and sahm[u] >= 0.5:
            reason = "sahm"
        elif u - entry_pos >= hold_days:
            reason = "time"
        else:
            continue
        return (u, u + 1, reason) if u + 1 < n else (u, None, reason)
    return (None, None, "open")


def simulate(variant: str, f: pd.DataFrame, cond: pd.Series, sahm: pd.Series, allowed: pd.Series) -> list[Trade]:
    """한 변형의 거래 목록. 보유 중(신호일 ~ 청산 체결일)에는 새 신호를 무시한다.

    입력: variant(A·B1·B2·B3·B4·D), f(price_features), cond(그 변형의 그날 진입 조건 bool — B4는 A 조건),
    sahm(일별 as-of 샴), allowed(판정 구간 안이고 필요한 데이터가 있는 날 bool).
    출력: Trade list. 신호일은 allowed인 날만.
    """
    n = len(f)
    cnd = cond.fillna(False).to_numpy(dtype=bool)
    ok = allowed.fillna(False).to_numpy(dtype=bool)
    s = sahm.to_numpy(dtype=float)
    trades: list[Trade] = []
    t = 0
    while t < n - 1:
        if variant == "B4":
            if not (ok[t] and cnd[t]):
                t += 1
                continue
            setup = t
            cross = f["cross20_up"].to_numpy()
            sig = next((p for p in range(setup + 1, min(setup + 61, n)) if cross[p] and ok[p]), None)
            if sig is None:  # 60거래일 안에 확인 없음 → 취소, 그 뒤 다시 조건을 찾는다
                t = setup + 61
                continue
            t_sig = sig
        else:
            if not (ok[t] and cnd[t]):
                t += 1
                continue
            t_sig = t
        if t_sig + 1 >= n:
            break
        entry = t_sig + 1
        if variant == "B3":
            ex = entry + 20
            tr = Trade(t_sig, entry, None, ex if ex < n else None, "fixed20" if ex < n else "open")
        elif variant == "D":
            c = f["close"].to_numpy()
            above = f["above200"].to_numpy()
            target = f["high252"].to_numpy()[t_sig]
            tr = Trade(t_sig, entry, None, None, "open")
            for u in range(entry, n):
                reason = "high" if c[u] >= target else ("below200" if not above[u] else None)
                if reason:
                    tr = Trade(t_sig, entry, u, u + 1 if u + 1 < n else None, reason)
                    break
        else:
            u, ex, reason = _run_exit_a(f, s, t_sig, entry)
            tr = Trade(t_sig, entry, u, ex, reason)
        trades.append(tr)
        if tr.exit_pos is None:
            break
        t = tr.exit_pos + 1  # 청산 체결일 다음 날부터 새 신호
    return trades


def variant_conditions(f: pd.DataFrame, sahm: pd.Series, baa_pct: pd.Series, cape_pct: pd.Series,
                       fear: pd.Series) -> dict[str, pd.Series]:
    """변형별 그날 진입 조건(bool). B4는 A 조건(셋업)을 쓰고 확인은 simulate가 한다. 데이터 없음(NaN)은 거짓."""
    a = base_condition(f, sahm)
    return {
        "A": a,
        "B1": a & (baa_pct < 0.9),
        "B2": a & (cape_pct < 0.9),
        "B3": fear >= 40,
        "B4": a,
        "D": f["above200"] & (f["close"] <= 0.9 * f["high252"]) & f["high252"].notna(),
    }
