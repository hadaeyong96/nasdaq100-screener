"""마이크론 사이클 해부 1단계 — 분석 보고서 (매매 전략 수익률 계산 없음).

발견 구간 1990-01 ~ 2012-12만 쓴다(data/mu_cycle.guard가 2012-12-31 뒤 값을 막는다). 지표는 공개 시차를 반영한 월말 값.
데이터는 미리 캐시에 받아 둔다: python -c "from data import mu_cycle as m; ..." (data/mu_cycle.fetch_*).

실행: python scripts/mu_cycle_report.py
출력: docs/results/mu_cycle/v{N}/ (report.md, 표 CSV, mu_cycle.png) — 이미 있는 버전은 덮어쓰지 않는다.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import mu_cycle as mc  # noqa: E402
from data import mu_cycle as dm  # noqa: E402

OUT_BASE = ROOT / "docs" / "results" / "mu_cycle"
START = pd.Timestamp("1990-01-31")
PEERS = ["AMAT", "LRCX", "KLAC", "TXN", "MCHP"]
# (FRED id, 이름, 공개 주기, 금리 여부). 달러는 DTWEXBGS가 2006년부터라 1995년부터 있는 TWEXB를 전 구간에 쓴다(잇지 않음).
INDICATORS = [("IPG3344S", "반도체 산업생산", "monthly", False), ("A34SNO", "컴퓨터·전자 신규주문", "monthly", False),
              ("INDPRO", "산업생산", "monthly", False), ("T10Y2Y", "장단기 금리차", "daily", True),
              ("TWEXB", "달러(광의)", "weekly", False), ("ICSA", "실업수당 신규청구", "weekly", False),
              ("PCOPPUSDM", "구리", "monthly", False), ("FEDFUNDS", "연방기금금리", "monthly", True)]
COLORS = {"line": "#2a78d6", "peak": "#eb6834", "trough": "#1baf7a", "ink": "#0b0b0b", "ink2": "#52514e",
          "grid": "#e4e3df", "surface": "#fcfcfb"}


def month_index() -> pd.DatetimeIndex:
    return pd.date_range(dm.DATA_START, dm.CUTOFF, freq="ME")


def company(ticker: str, me: pd.DatetimeIndex) -> dict:
    """종목 하나: 일봉, 월말 종가(1990~), 지그재그 점, 바닥·꼭대기 월말."""
    daily = dm.load_yf(ticker)
    m = mc.month_end_close(daily["Close"]).reindex(me)["close"]
    start = max(START, m.first_valid_index())
    pts = mc.zigzag(m[m.index >= START])
    pts = pts[pts["date"] != start].reset_index(drop=True)  # 창 첫 달은 그 전을 볼 수 없어 진짜 꼭대기·바닥인지 알 수 없다
    return {"daily": daily, "m": m, "pts": pts, "start": start,
            "troughs": list(pts.loc[pts["kind"] == "trough", "date"]), "peaks": list(pts.loc[pts["kind"] == "peak", "date"])}


def bench_index(me: pd.DatetimeIndex) -> pd.Series:
    """벤치마크 월말 지수: 1999-03까지 ^IXIC, 그 뒤 QQQ 월말 수익률로 잇는다(가격 기준)."""
    ix = mc.month_end_close(dm.load_yf("^IXIC")["Close"]).reindex(me)["close"]
    qq = mc.month_end_close(dm.load_yf("QQQ")["Close"]).reindex(me)["close"]
    cut = pd.Timestamp("1999-03-31")
    out = ix.copy()
    later = me[me > cut]
    out[later] = ix[cut] * qq[later] / qq[cut]
    return out


def indicator_forms(me: pd.DatetimeIndex) -> dict[tuple[str, str], pd.Series]:
    out = {}
    for sid, name, freq, is_rate in INDICATORS:
        known = mc.known_monthly(dm.load_fred(sid), freq, me)
        for form, s in mc.transforms(known, is_rate).items():
            out[(name, form)] = s
    return out


def analyze(c: dict, forms: dict) -> pd.DataFrame:
    """지표·형태마다 선행 개월·적중·놓침·헛신호·순위 상관."""
    rows = []
    fwd = mc.forward_return(c["m"], 12)
    for (name, form), s in forms.items():
        lt = mc.lead_months(s, c["troughs"], "trough")
        lp = mc.lead_months(s, c["peaks"], "peak")
        st = mc.signal_stats(s, c["m"], c["troughs"], c["start"])
        both = [x for x in lt + lp if not np.isnan(x)]
        lt_ok = [x for x in lt if not np.isnan(x)]
        rows.append({"지표": name, "형태": form, "바닥 선행 개월": lt, "꼭대기 선행 개월": lp,
                     "바닥 선행 중앙값": np.median(lt_ok) if lt_ok else np.nan,
                     "꼭대기 선행 중앙값": np.median([x for x in lp if not np.isnan(x)]) if any(not np.isnan(x) for x in lp) else np.nan,
                     "바닥 선행 IQR": float(np.subtract(*np.percentile(lt_ok, [75, 25]))) if len(lt_ok) >= 2 else np.nan,
                     "선행 양수 비율": float(np.mean([x > 0 for x in both])) if both else np.nan, "선행 표본": len(both),
                     "바닥 표본": len(lt_ok), **st,
                     "12개월 뒤 수익률 순위 상관": mc.spearman(s[s.index >= c["start"]], fwd[fwd.index >= c["start"]])})
    return pd.DataFrame(rows)


def rank(a: pd.DataFrame) -> pd.DataFrame:
    """다음 단계 후보 순위(사전 고정 규칙): 평가 신호 ≥ 3·바닥 선행 표본 ≥ 3인 것 중
    점수 = 선행 양수 비율 + 적중률 − 놓침 비율 − 헛신호 비율, 같으면 바닥 선행 IQR이 작은 쪽."""
    e = a[(a["신호 수"] >= 3) & (a["바닥 표본"] >= 3)].copy()
    e["점수"] = e["선행 양수 비율"] + e["적중률"] - e["놓침 비율"].fillna(1.0) - e["헛신호 비율"]
    return e.sort_values(["점수", "바닥 선행 IQR"], ascending=[False, True])


def stage_table(c: dict, q: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """단계별 평균 PER·변동성·거래량 변화, 그리고 꼭대기·바닥 월말 PER."""
    daily = c["daily"]
    rows = []
    for stage, at, lo, hi in mc.stage_windows(c["pts"]):
        f = mc.window_features(daily, lo, hi)
        w = daily[(daily.index > lo) & (daily.index <= hi)]
        pe, loss = np.nan, 0
        if not q.empty and len(w):
            vals = []
            for d, px in w["Close"].resample("ME").last().dropna().items():
                e = mc.ttm_eps(q, d)
                if np.isnan(e):
                    continue
                if e <= 0:
                    loss += 1
                else:
                    vals.append(px / e)
            pe = float(np.mean(vals)) if vals else np.nan
        rows.append({"단계": stage, "기준 점": at.date(), **f, "평균 PER": pe, "적자 달": loss})
    st = pd.DataFrame(rows)
    pts = []
    for _, r in c["pts"].iterrows():
        e = mc.ttm_eps(q, r["date"]) if not q.empty else np.nan
        pts.append({"점": "꼭대기" if r["kind"] == "peak" else "바닥", "날짜": r["date"].date(), "종가": r["price"],
                    "TTM 희석 EPS": e, "PER": r["price"] / e if (not np.isnan(e) and e > 0) else np.nan,
                    "비고": "자료 없음" if np.isnan(e) else ("적자" if e <= 0 else "")})
    return st, pd.DataFrame(pts)


def fmt_pct(x, d=1):
    return "—" if pd.isna(x) else f"{x * 100:.{d}f}%"


def fmt_num(x, d=1):
    return "—" if pd.isna(x) else f"{x:.{d}f}"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("—" if (not isinstance(v, (list, str)) and pd.isna(v)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def leads_str(xs):
    return ", ".join("—" if np.isnan(x) else f"{x:+.0f}" for x in xs)


def plot(c: dict, forms: dict, top2: list[tuple[str, str]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": False,
                         "axes.edgecolor": COLORS["grid"], "axes.labelcolor": COLORS["ink2"],
                         "xtick.color": COLORS["ink2"], "ytick.color": COLORS["ink2"], "text.color": COLORS["ink"]})
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]},
                             facecolor=COLORS["surface"])
    m = c["m"][c["m"].index >= START]
    ax = axes[0]
    ax.plot(m.index, m.to_numpy(), color=COLORS["line"], lw=2)
    ax.set_yscale("log")
    pk, tr = c["pts"][c["pts"]["kind"] == "peak"], c["pts"][c["pts"]["kind"] == "trough"]
    ax.scatter(pk["date"], pk["price"], marker="^", s=64, color=COLORS["peak"], edgecolor=COLORS["surface"], lw=2, zorder=3, label="꼭대기")
    ax.scatter(tr["date"], tr["price"], marker="v", s=64, color=COLORS["trough"], edgecolor=COLORS["surface"], lw=2, zorder=3, label="바닥")
    for _, r in c["pts"].iterrows():
        ax.annotate(r["date"].strftime("%y.%m"), (r["date"], r["price"]), textcoords="offset points",
                    xytext=(0, 9 if r["kind"] == "peak" else -15), ha="center", fontsize=8, color=COLORS["ink2"])
    ax.set_title("MU 월말 종가 (로그, 분할 반영 Close) · 지그재그 −40% / +60%", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    for ax2, key in zip(axes[1:], top2):
        s = forms[key]
        s = s[s.index >= START]
        ax2.plot(s.index, s.to_numpy(), color=COLORS["line"], lw=2)
        ax2.axhline(0, color=COLORS["ink2"], lw=0.8)
        unit = "%p" if any(n == key[0] and r for _, n, _, r in INDICATORS) else "%"
        ax2.set_title(f"{key[0]} · {key[1]} ({unit}, 공개 시차 반영 월말 값)", loc="left", fontsize=10)
    for a in axes:
        a.set_facecolor(COLORS["surface"])
        a.grid(axis="y", color=COLORS["grid"], lw=0.6)
        a.spines[["top", "right"]].set_visible(False)
        for d in pk["date"]:
            a.axvline(d, color=COLORS["peak"], lw=1, ls="--", alpha=0.7, zorder=0)
        for d in tr["date"]:
            a.axvline(d, color=COLORS["trough"], lw=1, alpha=0.7, zorder=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=COLORS["surface"])
    plt.close(fig)


def main() -> None:
    out = OUT_BASE / "v1"
    n = 1
    while out.exists():
        n += 1
        out = OUT_BASE / f"v{n}"
    me = month_index()
    forms = indicator_forms(me)
    mu = company("MU", me)
    bench = bench_index(me)
    cyc = mc.cycle_table(mu["pts"], mu["m"], bench)
    a_mu = analyze(mu, forms)
    ranked = rank(a_mu)
    peers = {t: company(t, me) for t in PEERS}
    a_peer = {t: analyze(c, forms) for t, c in peers.items()}
    q = mc.quarterly_eps(dm.load_mu_eps())
    stages, pe_pts = stage_table(mu, q)
    top3 = list(ranked.head(3)[["지표", "형태"]].itertuples(index=False, name=None))
    out.mkdir(parents=True)
    plot(mu, forms, top3[:2], out / "mu_cycle.png")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    write(out, commit, mu, cyc, a_mu, ranked, a_peer, peers, stages, pe_pts, q, top3)
    print("완료", out.relative_to(ROOT).as_posix())


def write(out, commit, mu, cyc, a_mu, ranked, a_peer, peers, stages, pe_pts, q, top3) -> None:
    base = mc.base_hit_rate(mu["m"].index, mu["troughs"], mu["start"])
    cyc.to_csv(out / "mu_cycles.csv", index=False, encoding="utf-8-sig")
    a_mu.to_csv(out / "mu_indicators.csv", index=False, encoding="utf-8-sig")
    # 사이클 표
    c = cyc.copy()
    for k in ("하락폭", "상승폭"):
        c[k] = c[k].map(fmt_pct)
    for k in ("하락 구간 벤치 대비(%p)", "상승 구간 벤치 대비(%p)"):
        c[k] = c[k].map(lambda v: "—" if pd.isna(v) else f"{v:+.0f}")
    for k in ("회복 기간(개월)", "사이클 길이(개월)"):
        c[k] = c[k].map(lambda v: "미회복(2012 말까지)" if pd.isna(v) and k.startswith("회복") else fmt_num(v, 0))
    pts = mu["pts"].copy()
    pts["date"] = pts["date"].dt.date
    pts["confirmed"] = pts["confirmed"].dt.date
    pts["kind"] = pts["kind"].map({"peak": "꼭대기", "trough": "바닥"})
    pts.columns = ["날짜", "종류", "월말 종가", "확정된 달"]
    pts["월말 종가"] = pts["월말 종가"].map(lambda v: f"{v:.2f}")
    dd = cyc["하락폭"]
    summary = [f"- 꼭대기 → 다음 꼭대기 평균 {cyc['사이클 길이(개월)'].mean():.1f}개월 (중앙값 {cyc['사이클 길이(개월)'].median():.0f}, 완결 {cyc['사이클 길이(개월)'].notna().sum()}개)",
               f"- 하락폭: 평균 {fmt_pct(dd.mean())}, 중앙값 {fmt_pct(dd.median())}, 가장 얕음 {fmt_pct(dd.max())}, 가장 깊음 {fmt_pct(dd.min())} (사이클 {len(dd)}개)",
               f"- 하락 기간 평균 {cyc['하락 기간(개월)'].mean():.1f}개월, 상승 구간 평균 상승폭 {fmt_pct(cyc['상승폭'].mean())}",
               f"- 벤치(1999-03까지 ^IXIC, 그 뒤 QQQ) 대비: 하락 구간 평균 {cyc['하락 구간 벤치 대비(%p)'].mean():+.0f}%p, 상승 구간 평균 {cyc['상승 구간 벤치 대비(%p)'].mean():+.0f}%p"]
    # 지표 표
    t = a_mu.copy()
    t["바닥 선행(사이클별)"] = t["바닥 선행 개월"].map(leads_str)
    t["꼭대기 선행(사이클별)"] = t["꼭대기 선행 개월"].map(leads_str)
    show = pd.DataFrame({
        "지표": t["지표"], "형태": t["형태"], "바닥 선행(사이클별)": t["바닥 선행(사이클별)"],
        "바닥 중앙값": t["바닥 선행 중앙값"].map(lambda v: fmt_num(v, 1)),
        "꼭대기 선행(사이클별)": t["꼭대기 선행(사이클별)"], "꼭대기 중앙값": t["꼭대기 선행 중앙값"].map(lambda v: fmt_num(v, 1)),
        "선행 양수 비율": t["선행 양수 비율"].map(lambda v: fmt_pct(v, 0)),
        "신호": t["신호 수"], "적중률": [f"{fmt_pct(r, 0)} ({h}/{n})" for r, h, n in zip(t["적중률"], t["적중"], t["신호 수"])],
        "놓침": [f"{m}/{j}" for m, j in zip(t["놓침"], t["평가 바닥"])],
        "헛신호": [f"{f}/{n}" for f, n in zip(t["헛신호"], t["신호 수"])],
        "순위 상관(12개월 뒤)": t["12개월 뒤 수익률 순위 상관"].map(lambda v: fmt_num(v, 2))})
    # 동료 표
    prow = []
    for _, r in a_mu.iterrows():
        key = (r["지표"], r["형태"])
        row = {"지표": r["지표"], "형태": r["형태"], "MU 적중률": fmt_pct(r["적중률"], 0), "MU 놓침": f"{r['놓침']}/{r['평가 바닥']}",
               "MU 바닥 선행 중앙": fmt_num(r["바닥 선행 중앙값"], 1)}
        hits, leads, same = [], [], 0
        for tk in PEERS:
            p = a_peer[tk]
            pr = p[(p["지표"] == key[0]) & (p["형태"] == key[1])].iloc[0]
            row[f"{tk} 적중률"] = fmt_pct(pr["적중률"], 0)
            if not pd.isna(pr["적중률"]):
                hits.append(pr["적중률"])
            ls = [x for x in pr["바닥 선행 개월"] if not np.isnan(x)]
            leads += ls
            if ls and not pd.isna(r["바닥 선행 중앙값"]) and np.sign(np.median(ls)) == np.sign(r["바닥 선행 중앙값"]):
                same += 1
        row["동료 평균 적중률"] = fmt_pct(np.mean(hits), 0) if hits else "—"
        row["동료 바닥 선행 중앙(전체)"] = fmt_num(np.median(leads), 1) if leads else "—"
        row["선행 방향 MU와 같은 동료"] = f"{same}/5"
        prow.append(row)
    peer_tbl = pd.DataFrame(prow)
    peer_cyc = pd.DataFrame([{"종목": tk, "자료 시작": peers[tk]["start"].date(), "바닥 수": len(peers[tk]["troughs"]),
                              "꼭대기 수": len(peers[tk]["peaks"]),
                              "바닥 날짜": ", ".join(d.strftime("%Y-%m") for d in peers[tk]["troughs"])} for tk in PEERS])
    # 단계·PER
    stg = stages.copy()
    agg = stg.groupby("단계", sort=False).agg(변동성=("변동성", "mean"), 거래량변화=("거래량 변화", "mean"), 평균PER=("평균 PER", "mean"),
                                             PER창=("평균 PER", lambda s: int(s.notna().sum())), 적자달=("적자 달", "sum"), 표본=("변동성", lambda s: int(s.notna().sum())))
    agg = agg.reset_index()
    agg_show = pd.DataFrame({"단계": agg["단계"], "평균 연율 변동성": agg["변동성"].map(fmt_pct),
                             "평균 거래량 변화(앞 6개월 대비)": agg["거래량변화"].map(fmt_pct), "표본(창)": agg["표본"],
                             "평균 PER": [("자료 없음" if pd.isna(v) else f"{v:.1f}") + f" (창 {n}개)" for v, n in zip(agg["평균PER"], agg["PER창"])],
                             "적자 달 수": agg["적자달"]})
    stg_show = stg.copy()
    stg_show["변동성"] = stg_show["변동성"].map(fmt_pct)
    stg_show["거래량 변화"] = stg_show["거래량 변화"].map(fmt_pct)
    stg_show["평균 PER"] = stg_show["평균 PER"].map(lambda v: "자료 없음" if pd.isna(v) else f"{v:.1f}")
    pe = pe_pts.copy()
    pe["종가"] = pe["종가"].map(lambda v: f"{v:.2f}")
    pe["TTM 희석 EPS"] = pe["TTM 희석 EPS"].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}")
    pe["PER"] = pe["PER"].map(lambda v: "—" if pd.isna(v) else f"{v:.1f}")
    pe_known = pe_pts.dropna(subset=["TTM 희석 EPS"])
    if pe_known.empty:
        pe_verdict = "EDGAR 희석 EPS가 있는 꼭대기·바닥이 없어 확인할 수 없음(자료 없음)."
    else:
        per = ["적자" if r["TTM 희석 EPS"] <= 0 else f"{r['PER']:.1f}" for _, r in pe_known.iterrows()]
        pe_verdict = "확인 가능한 점: " + "; ".join(f"{r['점']} {r['날짜']} PER {p}" for (_, r), p in zip(pe_known.iterrows(), per))
    # 후보
    cand = []
    for i, (name, form) in enumerate(top3, 1):
        r = ranked[(ranked["지표"] == name) & (ranked["형태"] == form)].iloc[0]
        pr = peer_tbl[(peer_tbl["지표"] == name) & (peer_tbl["형태"] == form)].iloc[0]
        cand.append(f"{i}. **{name} · {form}** — 점수 {r['점수']:.2f}. 바닥 선행 {leads_str(r['바닥 선행 개월'])}개월(중앙 {fmt_num(r['바닥 선행 중앙값'])}, IQR {fmt_num(r['바닥 선행 IQR'])}), "
                    f"꼭대기 선행 {leads_str(r['꼭대기 선행 개월'])}개월, 선행 양수 비율 {fmt_pct(r['선행 양수 비율'], 0)}; "
                    f"신호 {r['신호 수']}번 중 적중 {r['적중']}·헛신호 {r['헛신호']}, 놓침 {r['놓침']}/{r['평가 바닥']}. "
                    f"동료 평균 적중률 {pr['동료 평균 적중률']}, 선행 방향 같은 동료 {pr['선행 방향 MU와 같은 동료']}.")
    eps_range = f"{q['end'].min().date()} ~ {q['end'].max().date()} 분기 {len(q)}개" if not q.empty else "없음"
    lines = [
        "# 마이크론(MU) 사이클 해부 1단계 — 분석 보고서 (판정 없음)", "",
        f"> 발견 구간 1990-01 ~ 2012-12만 사용. 2013년 이후 값은 받지도 읽지도 않았다(요청 끝 날짜 2012-12-31 + 읽을 때 날짜 차단). 매매 전략 수익률은 계산하지 않았다.",
        f"> 코드 커밋 `{commit}` · 실행 `python scripts/mu_cycle_report.py` → `{out.relative_to(ROOT).as_posix()}`", "",
        "## 0. 데이터와 정의", "",
        "- 주가: yfinance auto_adjust=False 일봉 `Close`(분할 반영, 배당 미반영 — 트레이딩뷰 기본 차트와 같은 기준). 사이클은 월말(그달 마지막 거래일) 종가.",
        "- FRED(키 없는 fredgraph.csv, 1985-01 ~ 2012-12 요청): " + ", ".join(f"{s}({n})" for s, n, _, _ in INDICATORS) + ". 달러는 DTWEXBGS가 2006년부터라 TWEXB(1995-01 ~)를 전 구간에 씀 — 두 지수를 잇지 않음.",
        "- 자료 시작: A34SNO 1992-02, PCOPPUSDM 1992-01, TWEXB 1995-01 → 그 전 사이클은 해당 지표 \"자료 부족\". 1985~1989 값은 1990년 전년비·12개월 창 계산에만 씀.",
        "- 공개 시차: 월간 = 관측 월 + 2개월째 1일, 주간(ICSA·TWEXB) = 관측일 + 7일, 일간(T10Y2Y) = 다음 날. FRED 현재 값(수정 반영)이며 당시 발표본이 아님.",
        "- 두 형태: 전년비(12개월)·6개월 변화. 금리·금리차(T10Y2Y·FEDFUNDS)는 %p 차이, 나머지는 증가율 %.",
        "- 지그재그(사전 고정): 월말 종가가 직전 꼭대기에서 −40% 이상이면 그 꼭대기 확정, 바닥에서 +60% 이상이면 그 바닥 확정. 사후에 확정되는 표시라 신호로 쓰지 않음. "
        "창 첫 달(MU 1990-01, MCHP 1993-03)에 찍힌 점은 그 전을 볼 수 없어 빼고 셈.",
        "- 선행 개월: 주가 바닥(꼭대기) 달 ±12개월 안에서 지표 최저(최고)가 온 달까지 거리. + = 지표가 먼저. 창에 빈 값이 있으면 \"—\".",
        "- 바닥 전환 신호(사전 정의): 지표(각 형태)가 최근 12개월(그달 포함) 최저보다 +2 이상(%p 또는 %) 높아진 첫 달. 적중 = 신호 달 ~ 12개월 뒤 안에 MU 바닥. "
        "헛신호 = 신호 뒤 12개월 안 월말 종가 최저가 신호 달 종가보다 −20% 이상. 놓침 = 바닥 앞 12개월(바닥 달 포함)에 신호 없음. 뒤 12개월이 없는 2012년 신호는 평가 제외.",
        "- 순위 상관: 지표와 MU 12개월 뒤 수익률(월말)의 스피어만 상관, 1990-01 ~ 2011-12.",
        f"- 재무: EDGAR companyfacts 희석 EPS 중 기간 끝·공시일이 2012-12-31 이하인 사실만 저장(원본 저장 안 함). 분기 {eps_range}. TTM = 그날까지 알려진 마지막 4분기 합. 알게 된 날 = min(처음 공시일, 분기 끝 + 60일) — 2010년 초기 XBRL 공시가 companyfacts에 빠져 그 분기들이 1년 뒤 비교 값으로만 있어서. "
        "FY2009 2~4분기는 XBRL에 없어 TTM은 2010-11부터.", "",
        "## 1. MU 사이클", "", "### 꼭대기·바닥", "", md(pts), "", "### 사이클별", "", md(c), "", *summary, "",
        "## 2. 선행 지표 (MU)", "",
        "괄호 안 숫자는 사이클 순서(1990년부터). 적중률 = 적중/평가 신호, 놓침 = 놓친 바닥/평가 바닥, 헛신호 = 헛신호/평가 신호.", "",
        f"**적중률 기준선 {fmt_pct(base, 0)}**: 평가 가능한 아무 달에 신호를 냈어도 12개월 안에 MU 바닥이 있었을 비율(바닥이 자주 와서 높다). "
        "적중률은 이 값과 비교해 읽는다. 선행 개월 ±12는 찾는 창의 끝이라 \"창 안에 뚜렷한 극값 없음\"에 가깝다.", "",
        md(show), "",
        "## 3. 경기순환주 특징 (MU, 단계별)", "",
        "변동성 = 창 안 일간 종가 수익률 표준편차 × √252. 거래량 변화 = 창 평균 거래량 ÷ 그 앞 6개월 평균 − 1. PER = 월말 종가 ÷ TTM 희석 EPS(흑자 달만), EDGAR 2009~2012 범위.", "",
        md(agg_show), "", "### 창별 값", "", md(stg_show), "",
        "### 꼭대기·바닥 PER (\"꼭대기에서 PER 낮고 바닥에서 PER 높다\" 확인)", "", md(pe), "", pe_verdict, "",
        "## 4. 확인용 동료 5곳 (같은 2·3번 적용, 요약)", "", md(peer_cyc), "", md(peer_tbl), "",
        "## 5. 다음 단계 사전 등록 후보", "",
        "순위 규칙(결과 전 고정): 평가 신호 3번 이상·바닥 선행 표본 3개 이상인 지표·형태 중 점수 = 선행 양수 비율 + 적중률 − 놓침 비율 − 헛신호 비율이 큰 순, 같으면 바닥 선행 IQR이 작은 순. "
        "표본이 사이클 몇 개뿐이라 순위는 다음 단계에서 시험할 가설 후보일 뿐 판정이 아니다.", "",
        *cand, "",
        "![MU 사이클과 상위 지표 2개](mu_cycle.png)", "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
