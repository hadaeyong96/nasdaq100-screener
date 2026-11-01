"""마이크론 사이클 해부 1단계 — 순수 계산 (docs/results/mu_cycle/v1).

네트워크·파일·현재 시각 접근 없음. 매매 전략 수익률은 계산하지 않는다(분석만).
지그재그 꼭대기·바닥은 뒤에 확정되는 사후 표시다 — 신호로 쓰지 않고 사이클을 나누는 데만 쓴다.
지표 값은 공개 시차를 반영해 "그 월말에 알 수 있었던 값"으로 바꾼 뒤에 쓴다(known_monthly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = {"monthly": pd.DateOffset(months=2), "weekly": pd.Timedelta(days=7), "daily": pd.Timedelta(days=1)}


# ── 가격·지그재그 ────────────────────────────────────────────────────────────


def month_end_close(close: pd.Series) -> pd.DataFrame:
    """일별 종가 → 달마다 마지막 거래일 종가. 출력: DataFrame[date(실제 거래일), close], 인덱스 = 그달 말일."""
    s = close.dropna()
    g = s.groupby(s.index.to_period("M"))
    out = pd.DataFrame({"date": g.apply(lambda x: x.index[-1]), "close": g.last()})
    out.index = out.index.to_timestamp("M")
    return out


def zigzag(prices: pd.Series, down: float = 0.40, up: float = 0.60) -> pd.DataFrame:
    """지그재그 꼭대기·바닥. 직전 꼭대기에서 −down 이상 떨어지면 그 꼭대기 확정, 바닥에서 +up 이상 오르면 그 바닥 확정.

    입력: 가격 Series(월말 종가). 출력: DataFrame[date, kind('peak'|'trough'), price, confirmed(확정된 날)] — 확정된 점만.
    처음에는 방향 없이 최고·최저를 같이 따라가다 먼저 문턱을 넘는 쪽으로 시작한다.
    """
    p = prices.dropna()
    v, idx = p.to_numpy(), p.index
    rows = []
    mode = None
    hi_i = lo_i = 0
    for i in range(1, len(v)):
        if mode is None:
            if v[i] > v[hi_i]:
                hi_i = i
            if v[i] < v[lo_i]:
                lo_i = i
            if v[i] <= v[hi_i] * (1 - down) and hi_i < i:
                rows.append((idx[hi_i], "peak", v[hi_i], idx[i]))
                mode, lo_i = "down", i
            elif v[i] >= v[lo_i] * (1 + up) and lo_i < i:
                rows.append((idx[lo_i], "trough", v[lo_i], idx[i]))
                mode, hi_i = "up", i
        elif mode == "down":
            if v[i] < v[lo_i]:
                lo_i = i
            elif v[i] >= v[lo_i] * (1 + up):
                rows.append((idx[lo_i], "trough", v[lo_i], idx[i]))
                mode, hi_i = "up", i
        else:
            if v[i] > v[hi_i]:
                hi_i = i
            elif v[i] <= v[hi_i] * (1 - down):
                rows.append((idx[hi_i], "peak", v[hi_i], idx[i]))
                mode, lo_i = "down", i
    return pd.DataFrame(rows, columns=["date", "kind", "price", "confirmed"])


def months_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    """a → b 개월 수(월 단위 차)."""
    return (b.year - a.year) * 12 + (b.month - a.month)


def cycle_table(points: pd.DataFrame, price: pd.Series, bench: pd.Series) -> pd.DataFrame:
    """꼭대기마다 한 줄: 꼭대기·바닥·하락폭·하락 기간·회복 기간(바닥 → 꼭대기 값 회복 월말)·다음 꼭대기·상승폭·벤치 대비.

    입력: zigzag 결과, 월말 가격, 같은 인덱스의 벤치마크 월말 가격. 출력: DataFrame(값이 없으면 NaN·None).
    """
    pts = points.reset_index(drop=True)
    rows = []
    for i, r in pts.iterrows():
        if r["kind"] != "peak" or i + 1 >= len(pts):
            continue
        t = pts.loc[i + 1]
        nxt = pts.loc[i + 2] if i + 2 < len(pts) else None
        after = price[price.index > t["date"]]
        rec = after[after >= r["price"]]
        row = {"꼭대기": r["date"].date(), "바닥": t["date"].date(), "하락폭": t["price"] / r["price"] - 1,
               "하락 기간(개월)": months_between(r["date"], t["date"]),
               "회복 기간(개월)": months_between(t["date"], rec.index[0]) if len(rec) else np.nan,
               "다음 꼭대기": nxt["date"].date() if nxt is not None else None,
               "상승폭": nxt["price"] / t["price"] - 1 if nxt is not None else np.nan,
               "사이클 길이(개월)": months_between(r["date"], pts.loc[i + 2]["date"]) if nxt is not None else np.nan}
        bd = bench[t["date"]] / bench[r["date"]] - 1
        row["하락 구간 벤치 대비(%p)"] = (row["하락폭"] - bd) * 100
        if nxt is not None:
            row["상승 구간 벤치 대비(%p)"] = (row["상승폭"] - (bench[nxt["date"]] / bench[t["date"]] - 1)) * 100
        else:
            row["상승 구간 벤치 대비(%p)"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ── 지표: 공개 시차·변환 ────────────────────────────────────────────────────


def known_monthly(obs: pd.Series, freq: str, month_ends: pd.DatetimeIndex) -> pd.Series:
    """관측값을 "그 월말에 알 수 있었던 가장 최근 값"으로 바꾼다.

    공개 시차: monthly = 관측 월 + 2개월째 1일, weekly = 관측일 + 7일, daily = 관측일 + 1일.
    입력: 관측일 인덱스 Series, freq, 월말 인덱스. 출력: 월말 인덱스 Series(아직 아무 값도 모르면 NaN).
    """
    s = obs.dropna().sort_index()
    avail = pd.Series(s.to_numpy(), index=s.index + LAGS[freq])
    avail = avail[~avail.index.duplicated(keep="last")]
    return avail.reindex(avail.index.union(month_ends)).ffill().reindex(month_ends)


def transforms(m: pd.Series, is_rate: bool) -> dict[str, pd.Series]:
    """두 형태: 전년비(12개월)·6개월 변화. 금리·금리차(값이 % 단위)는 차이(%p), 나머지는 증가율(%)."""
    if is_rate:
        return {"전년비": m - m.shift(12), "6개월": m - m.shift(6)}
    return {"전년비": (m / m.shift(12) - 1) * 100, "6개월": (m / m.shift(6) - 1) * 100}


# ── 선행 개월·신호 ──────────────────────────────────────────────────────────


def lead_months(ind: pd.Series, dates: list[pd.Timestamp], kind: str, window: int = 12) -> list[float]:
    """주가 바닥(kind='trough')·꼭대기('peak') 월마다 지표의 최저(최고)가 몇 개월 먼저 왔는지(+ = 지표가 먼저).

    지표 극값은 그 달 ±window개월 안에서 찾는다. 창 안 값이 하나라도 없으면 NaN(자료 부족).
    입력: 월말 인덱스 지표, 월말 날짜 목록. 출력: 날짜별 선행 개월.
    """
    out = []
    pos = {d: i for i, d in enumerate(ind.index)}
    for d in dates:
        i = pos.get(d)
        if i is None or i - window < 0 or i + window >= len(ind):
            out.append(np.nan)
            continue
        seg = ind.iloc[i - window:i + window + 1]
        if seg.isna().any():
            out.append(np.nan)
            continue
        j = int(np.argmin(seg.to_numpy()) if kind == "trough" else np.argmax(seg.to_numpy()))
        out.append(float(window - j))
    return out


def rebound_condition(ind: pd.Series, rebound: float = 2.0, lookback: int = 12) -> pd.Series:
    """지표가 최근 lookback개월(그달 포함) 최저보다 rebound 이상 높으면 True. 창에 빈 값이 있으면 NaN."""
    lo = ind.rolling(lookback, min_periods=lookback).min()
    cond = (ind - lo) >= rebound
    return cond.where(lo.notna() & ind.notna())


def signal_dates(ind: pd.Series, rebound: float = 2.0, lookback: int = 12) -> list[pd.Timestamp]:
    """바닥 전환 신호: 조건(rebound_condition)이 거짓 → 참으로 바뀐 달. 출력: 월말 날짜 목록."""
    c = rebound_condition(ind, rebound, lookback)
    prev = c.shift(1)
    fire = (c == True) & (prev == False)  # noqa: E712 — NaN은 어느 쪽도 아님
    return list(c.index[fire.to_numpy(dtype=bool)])


def signal_stats(ind: pd.Series, price: pd.Series, troughs: list[pd.Timestamp], start: pd.Timestamp,
                 horizon: int = 12, drop: float = 0.20, rebound: float = 2.0) -> dict:
    """적중·놓침·헛신호.

    - 평가 신호: start 이후, 뒤에 horizon개월이 다 있는 신호.
    - 적중: 신호 달 t ~ t+horizon 안에 주가 바닥이 있음. 적중률 = 적중 / 평가 신호.
    - 헛신호: t+1 ~ t+horizon 월말 종가 최저가 t 종가보다 drop 이상 낮음.
    - 놓침: 바닥 b 앞 horizon개월(b−horizon ~ b)에 신호가 없음. 그 창에 조건 값이 하나라도 없으면 평가 제외.
    입력: 월말 인덱스 지표·가격, 바닥 월말 날짜. 출력: dict.
    """
    idx = ind.index
    n = len(idx)
    pos = {d: i for i, d in enumerate(idx)}
    cond = rebound_condition(ind, rebound)
    sig = [d for d in signal_dates(ind, rebound) if d >= start and pos[d] + horizon < n]
    tset = {pos[t] for t in troughs if t in pos}
    pv = price.reindex(idx).to_numpy()
    hits = sum(any(pos[d] <= b <= pos[d] + horizon for b in tset) for d in sig)
    false = sum(np.nanmin(pv[pos[d] + 1:pos[d] + horizon + 1]) <= pv[pos[d]] * (1 - drop) for d in sig)
    all_sig = {pos[d] for d in signal_dates(ind, rebound)}
    judged, missed = 0, 0
    for t in troughs:
        b = pos.get(t)
        if b is None or b < horizon or t < start or cond.iloc[b - horizon:b + 1].isna().any():
            continue
        judged += 1
        if not any(b - horizon <= s <= b for s in all_sig):
            missed += 1
    return {"신호 수": len(sig), "적중": hits, "적중률": hits / len(sig) if sig else np.nan, "헛신호": int(false),
            "헛신호 비율": false / len(sig) if sig else np.nan, "평가 바닥": judged, "놓침": missed,
            "놓침 비율": missed / judged if judged else np.nan}


def forward_return(price: pd.Series, months: int = 12) -> pd.Series:
    """월말 t의 months개월 뒤 수익률(분석 표시용 — 끝 months개월은 NaN). 음수 shift 없이 위치로 계산한다."""
    v = price.to_numpy()
    out = np.full(len(v), np.nan)
    out[:len(v) - months] = v[months:] / v[:len(v) - months] - 1
    return pd.Series(out, index=price.index)


def spearman(a: pd.Series, b: pd.Series) -> float:
    """순위 상관(둘 다 값이 있는 달만)."""
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 10:
        return np.nan
    return float(df.iloc[:, 0].rank().corr(df.iloc[:, 1].rank()))


# ── 재무(PER)·단계별 특징 ────────────────────────────────────────────────────


def quarterly_eps(facts: pd.DataFrame, known_days: int = 60) -> pd.DataFrame:
    """EDGAR 희석 EPS 사실 → 분기 EPS(3개월). 4분기는 연간 − 같은 회계연도 세 분기. 같은 기간은 처음 공시된 값.

    알게 된 날 = min(처음 공시일, 기간 끝 + known_days일). MU의 2010년 초기 XBRL 공시가 companyfacts에 빠져 있어
    그 분기들이 1년 뒤 공시의 비교 값으로만 나오기 때문이다(10-K 기한 60일 — 10-Q 40일보다 늦게 잡음).
    입력: DataFrame[start, end, filed, val]. 출력: DataFrame[end, filed(알게 된 날), eps] (end 오름차순).
    """
    if facts.empty:
        return pd.DataFrame(columns=["end", "filed", "eps"])
    f = facts.dropna(subset=["start", "end"]).sort_values("filed").drop_duplicates(["start", "end"], keep="first").copy()
    f["filed"] = f["filed"].where(f["filed"] <= f["end"] + pd.Timedelta(days=known_days), f["end"] + pd.Timedelta(days=known_days))
    f["days"] = (f["end"] - f["start"]).dt.days
    q = f[(f["days"] >= 80) & (f["days"] <= 100)][["start", "end", "filed", "val"]].rename(columns={"val": "eps"})
    rows = [q]
    for _, y in f[(f["days"] >= 350) & (f["days"] <= 380)].iterrows():
        inside = q[(q["start"] >= y["start"] - pd.Timedelta(days=5)) & (q["end"] <= y["end"] + pd.Timedelta(days=5))]
        if len(inside) == 3 and not (q["end"].sub(y["end"]).abs() <= pd.Timedelta(days=5)).any():
            rows.append(pd.DataFrame([{"start": inside["end"].max() + pd.Timedelta(days=1), "end": y["end"],
                                       "filed": max(y["filed"], inside["filed"].max()), "eps": y["val"] - inside["eps"].sum()}]))
    out = pd.concat(rows, ignore_index=True).sort_values("end").drop_duplicates("end", keep="first")
    return out[["end", "filed", "eps"]].reset_index(drop=True)


def ttm_eps(q: pd.DataFrame, day: pd.Timestamp) -> float:
    """day까지 공시된 분기 중 마지막 4개(끝 날짜가 이어진 1년 안)의 합. 모자라면 NaN."""
    k = q[q["filed"] <= day].sort_values("end")
    if len(k) < 4:
        return np.nan
    last = k.tail(4)
    if (last["end"].iloc[-1] - last["end"].iloc[0]).days > 300:
        return np.nan
    return float(last["eps"].sum())


def stage_windows(points: pd.DataFrame, months: int = 6) -> list[tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """단계 창: (단계 이름, 기준 점, 창 시작(제외), 창 끝(포함)). 바닥 전·후, 꼭대기 전·후 각각 months개월."""
    out = []
    for _, r in points.iterrows():
        d = r["date"]
        name = "바닥" if r["kind"] == "trough" else "꼭대기"
        out.append((f"{name} 전 {months}개월", d, d - pd.DateOffset(months=months), d))
        out.append((f"{name} 후 {months}개월", d, d, d + pd.DateOffset(months=months)))
    return out


def window_features(daily: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp, months: int = 6) -> dict:
    """(lo, hi] 창의 연율 변동성(일간 종가 수익률)·거래량 변화(창 평균 ÷ 앞 months개월 평균 − 1).

    입력: 일봉 DataFrame[Close, Volume]. 출력: dict(값이 없으면 NaN).
    """
    w = daily[(daily.index > lo) & (daily.index <= hi)]
    pre = daily[(daily.index > lo - pd.DateOffset(months=months)) & (daily.index <= lo)]
    if len(w) < 20:
        return {"변동성": np.nan, "거래량 변화": np.nan}
    r = daily["Close"].pct_change().reindex(w.index)
    vol = float(r.std() * np.sqrt(252))
    vchg = float(w["Volume"].mean() / pre["Volume"].mean() - 1) if len(pre) >= 20 and pre["Volume"].mean() > 0 else np.nan
    return {"변동성": vol, "거래량 변화": vchg}
