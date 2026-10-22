"""실패 구간 분석기 4단계 — 부진 정의 3가지로 공통 원인·실패 지도·예상 확인 (analyze.py 도구 그대로 씀).

정의:
- 계획: 초과 곡선 직전 고점 대비 −5% ~ 회복(또는 끝) — 계획 문서 그대로.
- 보조A(계획 외, 결과를 본 뒤 추가): 하락 다리 = 지그재그 5%의 고점 → 저점 구간(떨어지는 동안만).
- 보조B(계획 외, 계획의 "63거래일 하위 10% 보조 표시"를 달 단위로): 그 달 안에 63거래일 초과 수익이 그 전략의 하위 10%인 날이 있는 달.
추가 이유: 10년 내내 기준보다 뒤처진 전략은 계획 정의로는 부진 구간 하나가 전체 기간(부진 달 90~100%)이 되어
조건별 배율이 모두 1에 가까워지고 무작위 대조가 뜻이 없어진다.
출력: data/cache/failure/analysis2.json, exports/failure/공통원인.xlsx·실패지도.xlsx에 탭 추가
실행: python scripts/failure/analyze2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402


def falls(ex: pd.Series, thr: float = A.THRESH) -> list[dict]:
    """지그재그(thr): 직전 고점 대비 −thr 이상 떨어진 다리(고점 → 다음 thr 반등 전 저점)."""
    v = ex.to_numpy()
    out, hi, lo, mode = [], 0, 0, "down"
    for i in range(1, len(v)):
        if mode == "down":
            if v[i] < v[lo]:
                lo = i
            if v[i] / v[lo] - 1 >= thr:
                if v[lo] / v[hi] - 1 <= -thr:
                    out.append({"start": ex.index[hi], "trough": ex.index[lo], "end": ex.index[lo], "depth": float(v[lo] / v[hi] - 1),
                                "days": int(lo - hi), "recovered": None})
                mode, hi = "up", i
        else:
            if v[i] > v[hi]:
                hi = i
            if v[i] / v[hi] - 1 <= -thr:
                mode, lo = "down", i
    if mode == "down" and v[lo] / v[hi] - 1 <= -thr:
        out.append({"start": ex.index[hi], "trough": ex.index[lo], "end": ex.index[lo], "depth": float(v[lo] / v[hi] - 1),
                    "days": int(lo - hi), "recovered": None})
    return out


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    out, a = [], None
    for i, x in enumerate(mask):
        if x and a is None:
            a = i
        if not x and a is not None:
            out.append((a, i - 1)); a = None
    if a is not None:
        out.append((a, len(mask) - 1))
    return out


def common(series, key, rng) -> list[dict]:
    names = list(A.CONDS)

    def pooled(masks):
        out = {}
        for c in names:
            ni = di = na = da = 0.0
            for s, m in zip(series, masks):
                x = s["cond"][c].to_numpy(); ok = ~np.isnan(x)
                na += np.nansum(x[ok]); da += ok.sum(); ni += np.nansum(x[ok & m]); di += (ok & m).sum()
            rin, rall = (ni / di if di else np.nan), (na / da if da else np.nan)
            out[c] = (rin / rall if rall and rall > 0 else np.nan, rin, rall, di)
        return out

    act = pooled([s[key]["mask"] for s in series])
    rand = {c: [] for c in names}
    for _ in range(A.N_RANDOM):
        ms = [A.random_mask(s[key]["spans"], len(s["months"]), rng, s[key]["mask"]) for s in series]
        for c, v in pooled(ms).items():
            rand[c].append(v[0])
    rows = []
    for c in names:
        ratio, rin, rall, n_in = act[c]
        r = np.array([x for x in rand[c] if not np.isnan(x)])
        pct = float(100 * ((r < ratio).sum() + 0.5 * (r == ratio).sum()) / len(r)) if len(r) and not np.isnan(ratio) else np.nan
        n_eps = 0
        for s in series:
            x_all = s["cond"][c].to_numpy()
            for a, b in s[key]["spans"]:
                x = x_all[a:b + 1]; x = x[~np.isnan(x)]
                if len(x) and x.mean() >= 0.5:
                    n_eps += 1
        cand = (not np.isnan(pct)) and pct >= 95 and ratio > 1
        rows.append({"조건": c, "배율": ratio, "부진 안 발생률": rin, "전체 발생률": rall, "부진 달 수(자료 있음)": int(n_in),
                     "무작위 백분위": pct, "무작위 95백분위 배율": float(np.percentile(r, 95)) if len(r) else np.nan,
                     "패턴 후보": cand, "해당 부진 구간 수": n_eps, "패턴": cand and n_eps >= 3})
    return sorted(rows, key=lambda x: -(x["배율"] if not np.isnan(x["배율"]) else -9))


def group_ratio(series, key, group, cond):
    ni = di = na = da = 0.0
    for s in series:
        if s["group"] != group:
            continue
        x = s["cond"][cond].to_numpy(); ok = ~np.isnan(x); m = s[key]["mask"]
        ni += np.nansum(x[ok & m]); di += (ok & m).sum(); na += np.nansum(x[ok]); da += ok.sum()
    return (ni / di) / (na / da) if di and da and na else np.nan


def year_share(series, key, groups, years, files=None):
    a = b = n = 0
    for s in series:
        if s["group"] not in groups or (files and not any(s["file"].startswith(f) for f in files)):
            continue
        ys = np.array([p.year for p in s["months"]])
        hit = np.isin(ys, years)
        m = s[key]["mask"]
        if m.sum() == 0:
            continue
        a += (hit & m).sum() / m.sum(); b += hit.mean(); n += 1
    return (a / n, b / n) if n else (np.nan, np.nan)


def main() -> None:
    ind, shapes = A.indicators()
    series = []
    for fname, label, group, mkt in A.SERIES:
        p = A.CURVES / f"{fname}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, index_col=0, parse_dates=True)[["strategy", "benchmark"]].dropna()
        df = df[df.index <= A.LAST]
        ex = (df["strategy"] / df["strategy"].iloc[0]) / (df["benchmark"] / df["benchmark"].iloc[0])
        months = pd.period_range(ex.index[0].to_period("M"), ex.index[-1].to_period("M"), freq="M")
        vals, pcs, cond = A.month_table(ind, shapes[mkt], months)
        s = {"file": fname, "label": label, "group": group, "ex": ex, "months": months, "vals": vals, "pcs": pcs, "cond": cond}
        plan = A.slumps(ex)
        s["계획"] = {"eps": plan, "mask": A.slump_months(plan, months), "spans": A.episode_spans(plan, months)}
        fl = falls(ex)
        s["보조A"] = {"eps": fl, "mask": A.slump_months(fl, months), "spans": A.episode_spans(fl, months)}
        r63 = ex / ex.shift(63) - 1
        low = (r63 <= r63.quantile(0.10)).groupby(ex.index.to_period("M")).any().reindex(months).fillna(False).to_numpy()
        s["보조B"] = {"eps": None, "mask": low, "spans": runs_of(low)}
        series.append(s)
    rng = np.random.default_rng(A.SEED)
    out = {"defs": {}, "maps": [], "pred": {}}
    for key in ("계획", "보조A", "보조B"):
        out["defs"][key] = {"common": common(series, key, rng),
                            "slump_month_share": {s["label"]: float(s[key]["mask"].mean()) for s in series},
                            "by_group": {g: {c: group_ratio(series, key, g, c) for c in A.CONDS} for g in ("종목 고르기", "지수 타이밍", "자산 배분")}}
    # 실패 지도: 보조A(하락 다리) 깊이 상위 5 — 백분위는 그 전략 자기 기간 안(평소 대비; 1977~2021 전체 기준이면 2010년대 저금리 같은 시대 차이가 섞임)
    ren = {"below200": "200일선 아래", "vol20": "20일 변동성", "dd": "고점 대비 낙폭", "vrebound": "V자 반등"}
    for s in series:
        own = s["vals"].rank(pct=True) * 100
        own = own.rename(columns=ren)
        for e in sorted(s["보조A"]["eps"], key=lambda e: e["depth"])[:5]:
            mm = (s["months"] >= e["start"].to_period("M")) & (s["months"] <= e["end"].to_period("M"))
            avg = own[mm].mean().drop(labels=["200일선 아래", "V자 반등", "침체(NBER, 사후)", "사건 수"], errors="ignore")
            dev = (avg - 50).abs().dropna().sort_values(ascending=False)
            out["maps"].append({"전략": s["label"], "묶음": s["group"], "시작(고점)": str(e["start"].date()), "저점": str(e["trough"].date()),
                                "깊이": e["depth"], "길이(거래일)": e["days"],
                                "그때 달랐던 지표 3개": " · ".join(f"{k} {avg[k]:.0f}백분위" for k in dev.index[:3]),
                                **{f"백분위:{k}": avg[k] for k in avg.index}})
    for key in ("계획", "보조A", "보조B"):
        p1 = year_share(series, key, ["종목 고르기"], [2017, 2020, 2021])
        p2 = year_share(series, key, ["지수 타이밍"], [2009, 2015, 2016, 2018, 2020])
        p3 = year_share(series, key, ["자산 배분"], [1977, 1978, 1979, 1980, 1981, 1994], files=["2배균형"])
        cm = out["defs"][key]["common"]
        ev = next(x for x in cm if x["조건"] == "큰 사건이 있는 달(사건 달력)")
        best = [x for x in cm if x["패턴 후보"] and x["조건"] != "큰 사건이 있는 달(사건 달력)"]
        out["pred"][key] = {
            "①": {"부진 달 중 2017·2020·2021": p1[0], "전체 달 중": p1[1],
                  "QQQE 부진 배율": group_ratio(series, key, "종목 고르기", "대형주 쏠림 — QQQE 부진(하위 20%)"),
                  "RSP 부진 배율": group_ratio(series, key, "종목 고르기", "대형주 쏠림 — 동일비중 RSP 부진(하위 20%)")},
            "②": {"부진 달 중 2009·2015~16·2018·2020": p2[0], "전체 달 중": p2[1],
                  "V자 반등 배율": group_ratio(series, key, "지수 타이밍", "V자 반등(3개월 저점 대비 +20%)")},
            "③": {"부진 달 중 1977~81·1994": p3[0], "전체 달 중": p3[1],
                  "금리 급등 배율(자산 배분)": group_ratio([s for s in series if s["file"].startswith("2배균형")], key, "자산 배분",
                                                    "금리 급등(6개월 변화 상위 20%)")},
            "④": {"사건 달 배율": ev["배율"], "사건 무작위 백분위": ev["무작위 백분위"], "패턴 후보 지표 수": len(best),
                  "지표 상위": [(x["조건"], x["배율"], x["무작위 백분위"]) for x in best[:3]]}}
    (A.C / "analysis2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    with pd.ExcelWriter(A.EXP / "공통원인.xlsx", mode="a", if_sheet_exists="replace") as w:
        for key in ("계획", "보조A", "보조B"):
            pd.DataFrame(out["defs"][key]["common"]).to_excel(w, sheet_name=f"공통 원인({key})", index=False)
    with pd.ExcelWriter(A.EXP / "실패지도.xlsx", mode="a", if_sheet_exists="replace") as w:
        pd.DataFrame(out["maps"]).to_excel(w, sheet_name="하락 다리 상위5(보조A)", index=False)
    print("done")


if __name__ == "__main__":
    main()
