"""실패 구간 분석기 2~4단계 — 부진·호조 구간, 월별 원인 후보 표, 전략별 실패 지도, 공통 원인(무작위 배치 대조), 예상 4개.

입력: data/cache/failure/curves/*.csv(1단계), data/cache/failure/ind_raw(3단계 지표), docs/research/event_calendar_1977_2021.csv
출력: data/cache/failure/analysis.json, exports/failure/실패지도.xlsx·공통원인.xlsx·사건달력.xlsx
모든 값은 2021-12-31까지. 월별 지표는 그 달 말까지 공개된 값만(월간 경제지표는 한 달 늦춤, NBER 침체는 사후 표시).
실행: python scripts/failure/analyze.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
C = ROOT / "data" / "cache" / "failure"
CURVES, IND = C / "curves", C / "ind_raw"
EXP = ROOT / "exports" / "failure"
EXP.mkdir(parents=True, exist_ok=True)
LAST = pd.Timestamp("2021-12-31")
SEED = 20261022
N_RANDOM = 1000
THRESH = 0.05

# (파일, 보여 줄 이름, 묶음, 시장 모양을 잴 지수)
SERIES = [
    ("RSI_MACD_일목__QQQ기준", "RSI·MACD·일목 신호", "종목 고르기", "NDX"),
    ("MACD126_P5__QQQ기준", "MACD 1:2:6 신호(P5)", "종목 고르기", "NDX"),
    ("강한업종눌림B0__QQQ기준", "강한 업종 MACD 눌림", "종목 고르기", "NDX"),
    ("6개월강세RS26__QQQ기준", "6개월 강세 종목", "종목 고르기", "NDX"),
    ("가치평가__QQQ기준", "가치평가", "종목 고르기", "NDX"),
    ("상승장RSI눌림__QQQ기준", "상승장 RSI 눌림", "종목 고르기", "NDX"),
    ("실적급등SW1__QQQ기준", "실적급등 다음날 매수", "종목 고르기", "NDX"),
    ("200일선2배전환__S&P500", "200일선 2배 전환 · S&P500", "지수 타이밍", "GSPC"),
    ("200일선2배전환__나스닥 종합", "200일선 2배 전환 · 나스닥", "지수 타이밍", "IXIC"),
    ("200일선2배전환__QQQ(보정)", "200일선 2배 전환 · QQQ", "지수 타이밍", "NDX"),
    ("변동성맞춤V1__S&P500", "변동성 맞춤 V1 · S&P500", "지수 타이밍", "GSPC"),
    ("변동성맞춤V1__나스닥 종합", "변동성 맞춤 V1 · 나스닥", "지수 타이밍", "IXIC"),
    ("변동성맞춤V1__QQQ(보정)", "변동성 맞춤 V1 · QQQ", "지수 타이밍", "NDX"),
    ("갈아타기__QQQ(보정)", "주식·채권·금 갈아타기 · QQQ", "자산 배분", "NDX"),
    ("2배균형__S&P500", "주식·채권 2배 균형 · S&P500", "자산 배분", "GSPC"),
    ("2배균형__QQQ(보정)", "주식·채권 2배 균형 · QQQ", "자산 배분", "NDX"),
]


# ── 2단계: 부진·호조 구간 ─────────────────────────────────────────────────


def slumps(ex: pd.Series, thr: float = THRESH) -> list[dict]:
    """초과 곡선의 직전 고점 대비 −thr 이상 하락 구간: 시작(고점)·저점·끝(회복 또는 마지막 날)·깊이·길이."""
    out, peak_i, in_dd, trough_i = [], 0, False, 0
    v = ex.to_numpy()
    for i in range(1, len(v)):
        if v[i] >= v[peak_i]:
            if in_dd:
                out.append(_ep(ex, peak_i, trough_i, i, True))
                in_dd = False
            peak_i = i
            continue
        if not in_dd and v[i] / v[peak_i] - 1 <= -thr:
            in_dd, trough_i = True, i
        if in_dd and v[i] < v[trough_i]:
            trough_i = i
    if in_dd:
        out.append(_ep(ex, peak_i, trough_i, len(v) - 1, False))
    return out


def _ep(ex, a, t, b, recovered):
    return {"start": ex.index[a], "trough": ex.index[t], "end": ex.index[b], "depth": float(ex.iloc[t] / ex.iloc[a] - 1),
            "days": int(b - a), "recovered": recovered}


def rallies(ex: pd.Series, thr: float = THRESH) -> list[dict]:
    """지그재그(thr): 직전 저점 대비 +thr 이상 오른 구간(저점 → 다음 thr 하락 전 고점)."""
    v = ex.to_numpy()
    out, lo, hi, mode = [], 0, 0, "up"
    for i in range(1, len(v)):
        if mode == "up":
            if v[i] > v[hi]:
                hi = i
            if v[i] / v[hi] - 1 <= -thr:
                if v[hi] / v[lo] - 1 >= thr:
                    out.append({"start": ex.index[lo], "peak": ex.index[hi], "gain": float(v[hi] / v[lo] - 1), "days": int(hi - lo)})
                mode, lo = "down", i
        else:
            if v[i] < v[lo]:
                lo = i
            if v[i] / v[lo] - 1 >= thr:
                mode, hi = "up", i
    if mode == "up" and v[hi] / v[lo] - 1 >= thr:
        out.append({"start": ex.index[lo], "peak": ex.index[hi], "gain": float(v[hi] / v[lo] - 1), "days": int(hi - lo)})
    return out


# ── 3단계: 월별 지표 ──────────────────────────────────────────────────────


def _fred(sid):
    s = pd.read_csv(IND / f"fred_{sid}.csv", index_col=0, parse_dates=True).iloc[:, 0]
    return s[s.index <= LAST].dropna()


def _yf(tk, col="Adj Close"):
    s = pd.read_csv(IND / f"yf_{tk}.csv", index_col=0, parse_dates=True)[col]
    return s[s.index <= LAST].dropna()


def month_end(s: pd.Series) -> pd.Series:
    return s.groupby(s.index.to_period("M")).last()


def shape(px: pd.Series) -> pd.DataFrame:
    """시장 모양(그 달 말): 200일선 아래, 20일 변동성(연율), 고점 대비 낙폭, V자 반등(최근 3개월 저점 대비 +20% 이상)."""
    d = pd.DataFrame({"px": px})
    d["below200"] = (px < px.rolling(200).mean()).astype(float).where(px.rolling(200).mean().notna())
    d["vol20"] = px.pct_change().rolling(20).std() * np.sqrt(252)
    d["dd"] = px / px.cummax() - 1
    d["vrebound"] = (px / px.rolling(63).min() - 1 >= 0.20).astype(float)
    return d.groupby(d.index.to_period("M")).last().drop(columns="px")


def ret6(a: pd.Series, b: pd.Series | None = None) -> pd.Series:
    m = month_end(a)
    r = m / m.shift(6) - 1
    if b is not None:
        mb = month_end(b)
        r = r - (mb / mb.shift(6) - 1)
    return r


def indicators() -> tuple[pd.DataFrame, dict]:
    """월(Period) × 지표. 출력: (표, 시장 모양 표 {지수: DataFrame})."""
    shapes = {"GSPC": shape(_yf("GSPC", "Close")), "IXIC": shape(_yf("IXIC", "Close")), "NDX": shape(_yf("NDX", "Close"))}
    dgs10 = month_end(_fred("DGS10"))
    ind = pd.DataFrame({
        "VIX": month_end(_fred("VIXCLS")),
        "동일비중−SPY 6개월(RSP)": ret6(_yf("RSP"), _yf("SPY")),
        "동일비중−QQQ 6개월(QQQE)": ret6(_yf("QQQE"), _yf("QQQ")),
        "소형−SPY 6개월(IWM)": ret6(_yf("IWM"), _yf("SPY")),
        "가치−성장 6개월(IWD−IWF)": ret6(_yf("IWD"), _yf("IWF")),
        "10년 금리": dgs10,
        "10년 금리 6개월 변화": dgs10 - dgs10.shift(6),
        "10년−2년": month_end(_fred("T10Y2Y")),
    })
    ff = _fred("FEDFUNDS")
    ff.index = ff.index.to_period("M") + 1  # 월평균은 다음 달 초 공개 → 한 달 늦춤
    ind["기준금리 6개월 변화"] = ff - ff.shift(6)
    baa = _fred("BAA")
    baa.index = baa.index.to_period("M") + 1
    ind["신용 스프레드(BAA−10년)"] = baa.reindex(ind.index) - dgs10.reindex(ind.index)
    usd_old, usd_new = month_end(_fred("DTWEXM")), month_end(_fred("DTWEXBGS"))
    chg = (usd_old / usd_old.shift(6) - 1).combine_first(usd_new / usd_new.shift(6) - 1)
    ind["달러 6개월 변화"] = chg
    un = _fred("UNRATE")
    un.index = un.index.to_period("M") + 1
    ind["실업률 6개월 변화"] = un - un.shift(6)
    cpi = _fred("CPIAUCSL")
    cpi.index = cpi.index.to_period("M") + 1
    ind["CPI 전년비"] = cpi / cpi.shift(12) - 1
    rec = _fred("USREC")
    rec.index = rec.index.to_period("M")
    ind["침체(NBER, 사후)"] = rec
    ev = pd.read_csv(ROOT / "docs" / "research" / "event_calendar_1977_2021.csv", parse_dates=["date"])
    ind["사건 수"] = ev.groupby(ev["date"].dt.to_period("M")).size().reindex(ind.index).fillna(0)
    ind = ind[(ind.index >= pd.Period("1977-01", "M")) & (ind.index <= pd.Period("2021-12", "M"))]
    return ind, {k: v[(v.index >= pd.Period("1977-01", "M")) & (v.index <= pd.Period("2021-12", "M"))] for k, v in shapes.items()}


def pct_rank(df: pd.DataFrame) -> pd.DataFrame:
    """지표마다 1977~2021(있는 달) 안 백분위(0~100)."""
    return df.rank(pct=True) * 100


CONDS = {  # 이름 → (열, 함수(값, 백분위) → bool 또는 NaN)
    "200일선 아래": ("below200", lambda v, p: v == 1),
    "변동성 높음(상위 20%)": ("vol20", lambda v, p: p >= 80),
    "낙폭 −10% 이하": ("dd", lambda v, p: v <= -0.10),
    "V자 반등(3개월 저점 대비 +20%)": ("vrebound", lambda v, p: v == 1),
    "VIX 높음(상위 20%)": ("VIX", lambda v, p: p >= 80),
    "대형주 쏠림 — 동일비중 RSP 부진(하위 20%)": ("동일비중−SPY 6개월(RSP)", lambda v, p: p <= 20),
    "대형주 쏠림 — QQQE 부진(하위 20%)": ("동일비중−QQQ 6개월(QQQE)", lambda v, p: p <= 20),
    "소형주 부진(IWM 하위 20%)": ("소형−SPY 6개월(IWM)", lambda v, p: p <= 20),
    "성장주 강세(가치−성장 하위 20%)": ("가치−성장 6개월(IWD−IWF)", lambda v, p: p <= 20),
    "가치주 강세(가치−성장 상위 20%)": ("가치−성장 6개월(IWD−IWF)", lambda v, p: p >= 80),
    "금리 높음(상위 20%)": ("10년 금리", lambda v, p: p >= 80),
    "금리 급등(6개월 변화 상위 20%)": ("10년 금리 6개월 변화", lambda v, p: p >= 80),
    "금리 급락(6개월 변화 하위 20%)": ("10년 금리 6개월 변화", lambda v, p: p <= 20),
    "장단기 역전(10년−2년 < 0)": ("10년−2년", lambda v, p: v < 0),
    "기준금리 인상(6개월 +0.25%p 이상)": ("기준금리 6개월 변화", lambda v, p: v >= 0.25),
    "기준금리 인하(6개월 −0.25%p 이하)": ("기준금리 6개월 변화", lambda v, p: v <= -0.25),
    "신용 스프레드 넓음(상위 20%)": ("신용 스프레드(BAA−10년)", lambda v, p: p >= 80),
    "달러 강세(6개월 상위 20%)": ("달러 6개월 변화", lambda v, p: p >= 80),
    "달러 약세(6개월 하위 20%)": ("달러 6개월 변화", lambda v, p: p <= 20),
    "실업률 상승(6개월 +0.5%p 이상)": ("실업률 6개월 변화", lambda v, p: v >= 0.5),
    "물가 높음(CPI 전년비 상위 20%)": ("CPI 전년비", lambda v, p: p >= 80),
    "침체(NBER, 사후 표시)": ("침체(NBER, 사후)", lambda v, p: v == 1),
    "큰 사건이 있는 달(사건 달력)": ("사건 수", lambda v, p: v >= 1),
}


def month_table(ind: pd.DataFrame, shp: pd.DataFrame, months: pd.PeriodIndex) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """시리즈 한 개의 (값, 백분위, 조건 0/1/NaN) 월별 표."""
    full = ind.join(shp, how="outer")
    pr = pct_rank(full)
    vals, pcs = full.reindex(months), pr.reindex(months)
    cond = pd.DataFrame(index=months)
    for name, (col, fn) in CONDS.items():
        v, p = vals[col], pcs[col]
        cond[name] = np.where(v.isna(), np.nan, fn(v, p).astype(float))
    return vals, pcs, cond


def slump_months(eps: list[dict], months: pd.PeriodIndex) -> np.ndarray:
    m = np.zeros(len(months), dtype=bool)
    for e in eps:
        m |= (months >= e["start"].to_period("M")) & (months <= e["end"].to_period("M"))
    return m


def episode_spans(eps, months):
    pos = {p: i for i, p in enumerate(months)}
    return [(pos[e["start"].to_period("M")], pos[e["end"].to_period("M")]) for e in eps]


def random_mask(spans, n, rng, mask=None) -> np.ndarray:
    """같은 길이(달 수) 구간을 겹치지 않게 무작위로 다시 놓는다(20번 시도). 부진 달이 많아 못 놓으면
    원래 부진 달 표시를 무작위 칸만큼 원형으로 민다(구간 길이·간격 그대로)."""
    lens = sorted((b - a + 1 for a, b in spans), reverse=True)
    for _ in range(20):
        m = np.zeros(n, dtype=bool)
        ok = True
        for L in lens:
            starts = np.arange(0, n - L + 1)
            if len(starts) == 0:
                ok = False
                break
            occ = np.concatenate([[0], np.cumsum(m)])
            free = starts[(occ[starts + L] - occ[starts]) == 0]
            if len(free) == 0:
                ok = False
                break
            st = int(rng.choice(free))
            m[st:st + L] = True
        if ok:
            return m
    return np.roll(mask, int(rng.integers(1, n))) if mask is not None else np.zeros(n, dtype=bool)


def main() -> None:
    ind, shapes = indicators()
    series = []
    for fname, label, group, mkt in SERIES:
        p = CURVES / f"{fname}.csv"
        if not p.exists():
            print("없음(건너뜀):", fname)
            continue
        df = pd.read_csv(p, index_col=0, parse_dates=True)[["strategy", "benchmark"]].dropna()
        df = df[df.index <= LAST]
        ex = (df["strategy"] / df["strategy"].iloc[0]) / (df["benchmark"] / df["benchmark"].iloc[0])
        r63 = ex / ex.shift(63) - 1
        low10 = r63 <= r63.quantile(0.10)
        months = pd.period_range(ex.index[0].to_period("M"), ex.index[-1].to_period("M"), freq="M")
        vals, pcs, cond = month_table(ind, shapes[mkt], months)
        eps = slumps(ex)
        series.append({"file": fname, "label": label, "group": group, "mkt": mkt, "ex": ex, "low10": low10, "r63": r63,
                       "months": months, "vals": vals, "pcs": pcs, "cond": cond, "slumps": eps, "rallies": rallies(ex),
                       "smask": slump_months(eps, months), "spans": episode_spans(eps, months)})
    # 전략별 실패 지도 (깊이 상위 5)
    maps = []
    for s in series:
        for e in sorted(s["slumps"], key=lambda e: e["depth"])[:5]:
            mm = (s["months"] >= e["start"].to_period("M")) & (s["months"] <= e["end"].to_period("M"))
            avg = s["pcs"][mm].mean()
            dev = (avg - 50).abs().dropna().sort_values(ascending=False)
            top3 = [f"{k} {avg[k]:.0f}백분위" for k in dev.index[:3]]
            maps.append({"전략": s["label"], "묶음": s["group"], "시작": str(e["start"].date()), "저점": str(e["trough"].date()),
                         "끝": str(e["end"].date()) + ("" if e["recovered"] else "(미회복)"), "깊이": e["depth"], "길이(거래일)": e["days"],
                         "그때 달랐던 지표 3개": " · ".join(top3), **{f"백분위:{k}": avg[k] for k in avg.index}})
    # 공통 원인: 묶은 배율 + 무작위 배치
    rng = np.random.default_rng(SEED)
    names = list(CONDS)

    def pooled(masks):
        out = {}
        for c in names:
            num_in = den_in = num_all = den_all = 0.0
            for s, m in zip(series, masks):
                x = s["cond"][c].to_numpy()
                ok = ~np.isnan(x)
                num_all += np.nansum(x[ok]); den_all += ok.sum()
                num_in += np.nansum(x[ok & m]); den_in += (ok & m).sum()
            rin = num_in / den_in if den_in else np.nan
            rall = num_all / den_all if den_all else np.nan
            out[c] = (rin / rall if rall and rall > 0 else np.nan, rin, rall, den_in)
        return out

    actual = pooled([s["smask"] for s in series])
    rand = {c: [] for c in names}
    for _ in range(N_RANDOM):
        masks = []
        for s in series:
            masks.append(random_mask(s["spans"], len(s["months"]), rng, s["smask"]))
        for c, v in pooled(masks).items():
            rand[c].append(v[0])
    common = []
    for c in names:
        ratio, rin, rall, n_in = actual[c]
        r = np.array([x for x in rand[c] if not np.isnan(x)])
        pct = float(100 * ((r < ratio).sum() + 0.5 * (r == ratio).sum()) / len(r)) if len(r) and not np.isnan(ratio) else np.nan
        # 부진 구간 수: 그 구간 달의 절반 이상에서 조건이 맞은 구간
        n_eps = 0
        for s in series:
            for a, b in s["spans"]:
                x = s["cond"][c].to_numpy()[a:b + 1]
                x = x[~np.isnan(x)]
                if len(x) and x.mean() >= 0.5:
                    n_eps += 1
        cand = (not np.isnan(pct)) and pct >= 95
        common.append({"조건": c, "배율": ratio, "부진 구간 안 발생률": rin, "전체 발생률": rall, "부진 달 수(자료 있음)": int(n_in),
                       "무작위 백분위": pct, "무작위 95백분위": float(np.percentile(r, 95)) if len(r) else np.nan,
                       "패턴 후보(95백분위 이상)": cand, "해당 부진 구간 수": n_eps, "패턴(후보 + 3구간 이상)": cand and n_eps >= 3})
    common = sorted(common, key=lambda x: -(x["배율"] if not np.isnan(x["배율"]) else -9))
    # 묶음별 배율(예상 확인용)
    by_group = {}
    for g in ("종목 고르기", "지수 타이밍", "자산 배분"):
        sub = [(s, s["smask"]) for s in series if s["group"] == g]
        res = {}
        for c in names:
            ni = nall = di = dall = 0.0
            for s, m in sub:
                x = s["cond"][c].to_numpy(); ok = ~np.isnan(x)
                ni += np.nansum(x[ok & m]); di += (ok & m).sum(); nall += np.nansum(x[ok]); dall += ok.sum()
            res[c] = (ni / di) / (nall / dall) if di and dall and nall else np.nan
        by_group[g] = res
    # 예상 확인
    def year_share(group, years):
        a = b = 0
        for s in series:
            if s["group"] != group:
                continue
            ys = np.array([p.year for p in s["months"]])
            hit = np.isin(ys, years)
            a += (hit & s["smask"]).sum() / max(s["smask"].sum(), 1)
            b += hit.mean()
        n = sum(1 for s in series if s["group"] == group)
        return a / n, b / n

    pred = {}
    ys, yb = year_share("종목 고르기", [2017, 2020, 2021])
    pred["①"] = {"부진 달 중 2017·2020·2021 비율": ys, "전체 달 중 비율": yb,
                 "QQQE 부진 배율(종목 고르기)": by_group["종목 고르기"]["대형주 쏠림 — QQQE 부진(하위 20%)"],
                 "RSP 부진 배율(종목 고르기)": by_group["종목 고르기"]["대형주 쏠림 — 동일비중 RSP 부진(하위 20%)"]}
    ys2, yb2 = year_share("지수 타이밍", [2009, 2020, 2015, 2016, 2018])
    pred["②"] = {"부진 달 중 2009·2020·2015~16·2018 비율": ys2, "전체 달 중 비율": yb2,
                 "V자 반등 배율(지수 타이밍)": by_group["지수 타이밍"]["V자 반등(3개월 저점 대비 +20%)"]}
    bal = [s for s in series if s["file"].startswith("2배균형")]
    a = b = 0
    for s in bal:
        ys_ = np.array([p.year for p in s["months"]])
        hit = np.isin(ys_, [1977, 1978, 1979, 1980, 1981, 1994])
        a += (hit & s["smask"]).sum() / max(s["smask"].sum(), 1)
        b += hit.mean()
    mm = [s for s in bal]
    ni = di = nall = dall = 0.0
    for s in mm:
        x = s["cond"]["금리 급등(6개월 변화 상위 20%)"].to_numpy(); ok = ~np.isnan(x)
        ni += np.nansum(x[ok & s["smask"]]); di += (ok & s["smask"]).sum(); nall += np.nansum(x[ok]); dall += ok.sum()
    pred["③"] = {"부진 달 중 1977~81·1994 비율": a / len(bal), "전체 달 중 비율": b / len(bal),
                 "금리 급등 배율(2배 균형)": (ni / di) / (nall / dall) if di and nall else np.nan}
    ev = next(x for x in common if x["조건"] == "큰 사건이 있는 달(사건 달력)")
    best_ind = [x for x in common if x["조건"] != "큰 사건이 있는 달(사건 달력)" and x["패턴 후보(95백분위 이상)"]]
    pred["④"] = {"사건 달 배율": ev["배율"], "사건 무작위 백분위": ev["무작위 백분위"],
                 "패턴 후보 지표 수": len(best_ind), "가장 강한 지표": (best_ind[0]["조건"], best_ind[0]["배율"]) if best_ind else None}
    # 저장
    out = {
        "series": [{"label": s["label"], "group": s["group"], "file": s["file"], "start": str(s["ex"].index[0].date()),
                    "end": str(s["ex"].index[-1].date()), "excess_total": float(s["ex"].iloc[-1] / s["ex"].iloc[0] - 1),
                    "n_slumps": len(s["slumps"]), "n_rallies": len(s["rallies"]),
                    "slump_month_share": float(s["smask"].mean())} for s in series],
        "maps": maps, "common": common, "by_group": by_group, "pred": pred,
    }
    (C / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    write_excel(series, maps, common, by_group)
    print("done", len(series), "시리즈")


def write_excel(series, maps, common, by_group) -> None:
    with pd.ExcelWriter(EXP / "실패지도.xlsx") as w:
        pd.DataFrame(maps).to_excel(w, sheet_name="부진 상위5(전체)", index=False)
        for s in series:
            rows = []
            for e in s["slumps"]:
                mm = (s["months"] >= e["start"].to_period("M")) & (s["months"] <= e["end"].to_period("M"))
                avg = s["pcs"][mm].mean()
                rows.append({"구분": "부진", "시작": e["start"].date(), "저점/고점": e["trough"].date(), "끝": e["end"].date(),
                             "깊이/상승": e["depth"], "길이(거래일)": e["days"], "회복": e["recovered"], **{f"백분위:{k}": avg[k] for k in avg.index}})
            for r in s["rallies"]:
                mm = (s["months"] >= r["start"].to_period("M")) & (s["months"] <= r["peak"].to_period("M"))
                avg = s["pcs"][mm].mean()
                rows.append({"구분": "호조", "시작": r["start"].date(), "저점/고점": r["peak"].date(), "끝": r["peak"].date(),
                             "깊이/상승": r["gain"], "길이(거래일)": r["days"], "회복": None, **{f"백분위:{k}": avg[k] for k in avg.index}})
            sheet = s["label"][:28].replace("·", "_").replace(":", "_").replace("/", "_")
            pd.DataFrame(rows).to_excel(w, sheet_name=sheet, index=False)
        lows = []
        for s in series:
            d = pd.DataFrame({"전략": s["label"], "63거래일 초과": s["r63"], "하위 10%": s["low10"]}).dropna()
            lows.append(d[d["하위 10%"]].reset_index().rename(columns={"index": "date", "date": "date"}))
        pd.concat(lows).to_excel(w, sheet_name="63일 초과 하위10%", index=False)
    with pd.ExcelWriter(EXP / "공통원인.xlsx") as w:
        pd.DataFrame(common).to_excel(w, sheet_name="공통 원인(전체)", index=False)
        pd.DataFrame(by_group).reset_index().rename(columns={"index": "조건"}).to_excel(w, sheet_name="묶음별 배율", index=False)
        ind, shapes = indicators()
        ind.index = ind.index.astype(str)
        ind.to_excel(w, sheet_name="월별 지표(값)")
        pct_rank(ind).to_excel(w, sheet_name="월별 지표(백분위)")
    ev = pd.read_csv(ROOT / "docs" / "research" / "event_calendar_1977_2021.csv")
    with pd.ExcelWriter(EXP / "사건달력.xlsx") as w:
        ev.to_excel(w, sheet_name="사건 달력", index=False)


if __name__ == "__main__":
    main()
