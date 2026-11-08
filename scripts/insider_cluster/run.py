"""내부자 집단 매수 — 실행기 (docs/prereg/insider_cluster.md, 태그 insider-cluster-prereg-r2 — 편차 절 참고).

실행:
  python -m scripts.insider_cluster.run --gap-check             # 태그 전: 공시·메타데이터만으로 2절 점검(가격 본문 없음)
  python -m scripts.insider_cluster.run --run --pushed-at <시각> # 태그 push 뒤: 가격 본문을 받아 1회 계산
출력: docs/results/insider_cluster/v1/ — 이미 있으면 덮어쓰지 않고 멈춘다. 재실행·규칙 변경 없음.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import smallmid_prices_check as spc  # noqa: E402
from scripts.insider_cluster import book as bk  # noqa: E402
from scripts.insider_cluster import data as dm  # noqa: E402
from scripts.insider_cluster import signals as sg  # noqa: E402
from scripts.insider_cluster.gate import CUTOFF, assert_cutoff  # noqa: E402

OUT = ROOT / "docs" / "results" / "insider_cluster" / "v1"
TAG = "insider-cluster-prereg-r2"
GAP_JSON = dm.CACHE / "gap_check.json"
START = pd.Timestamp("2017-01-03")
GAP_LIMIT = 0.10
MIN_SIGNALS = 30


def log(msg: str) -> None:
    print(msg, flush=True)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout.strip()


def check_order() -> None:
    """태그가 원격에 있고 HEAD가 태그 커밋이며 작업 트리가 깨끗할 때만 통과. 아니면 SystemExit."""
    if OUT.exists():
        raise SystemExit(f"BLOCKED — {OUT.relative_to(ROOT)} 이미 있음(재실행 금지)")
    if not git("ls-remote", "origin", f"refs/tags/{TAG}"):
        raise SystemExit(f"BLOCKED — 원격에 태그 {TAG} 없음")
    if git("rev-list", "-n", "1", TAG) != git("rev-parse", "HEAD"):
        raise SystemExit("BLOCKED — HEAD가 태그 커밋이 아님")
    if git("status", "--porcelain"):
        raise SystemExit("BLOCKED — 작업 트리가 깨끗하지 않음")


# ── 신호 → 구역 ───────────────────────────────────────────────────────────────


def zone_signals(sig: pd.DataFrame, zone: dm.Zone, days: pd.DatetimeIndex) -> pd.DataFrame:
    """구역 ① 신호만 남기고 체결일을 붙인다. 입력: cluster_signals 표. 출력: 열 exec_day 추가(달력을 넘으면 NaT)."""
    keep = [zone.contains(int(r.cik), r.confirm) for r in sig.itertuples()]
    z = sig[keep].copy()
    ex = []
    for d in z["confirm"]:
        _, e = sg.use_and_exec(d, days)
        ex.append(days[e] if e is not None else pd.NaT)
    z["exec_day"] = ex
    return z.reset_index(drop=True)


def build_signals(min_people: int) -> pd.DataFrame:
    """캐시 공시 → 구역 ① 신호."""
    filings, owners, trades = dm.load_sec()
    pur, stats = sg.eligible_purchases(filings, owners, trades)
    days = sg.nyse_days()
    sig = sg.cluster_signals(pur, days, min_people)
    z = zone_signals(sig, dm.load_zone(), days)
    z.attrs["stats"] = stats
    z.attrs["all_signals"] = len(sig)
    return z


# ── 2절 점검 ──────────────────────────────────────────────────────────────────


def gap_status(z: pd.DataFrame, check: pd.DataFrame, tickers: pd.DataFrame, progress: dict, meta: dict, symbols: dict | None = None) -> pd.DataFrame:
    """신호별 가격 출처(메타데이터). symbols: Form 4 신고 티커. 출력: z + source(캐시/Tiingo 메타/없음), tickers(Tiingo 후보). 순수."""
    src, tks = [], []
    for r in z.itertuples():
        cik = int(r.cik)
        row = check.loc[cik].to_dict() if cik in check.index else None
        ok_day = pd.notna(r.exec_day)
        if row and ok_day and dm.covered_by_cache(row, r.confirm, r.exec_day):
            src.append("캐시")
            tks.append("")
            continue
        names = str(row.get("names", "")) if row else ""
        if row and str(row.get("price")) == "없음" and ok_day:
            cands = [t for t in dm.candidate_tickers_for(cik, tickers, progress, symbols) if dm.meta_covers(meta.get(t, {}), names, r.confirm, r.exec_day)]
            if cands:
                src.append("Tiingo 메타")
                tks.append(";".join(cands))
                continue
        src.append("없음")
        tks.append("")
    return z.assign(source=src, tiingo=tks)


def gap_check() -> dict:
    """2절 점검: 공시 수신 → 신호 → 메타데이터. 가격 본문을 받지 않는다. 출력: 요약 dict(gap_check.json)."""
    fails = dm.fetch_sec_all()
    if fails:
        raise SystemExit("BLOCKED — SEC 수신 실패: " + "; ".join(fails))
    z = build_signals(sg.MIN_PEOPLE)
    log(f"전체 신호 {z.attrs['all_signals']}개 · 구역 ① 신호 {len(z)}개 · 거래 날짜 오류 버림 {z.attrs['stats']['bad_trans_date']}행 / P {z.attrs['stats']['p_rows']}행")
    check = dm.load_price_check()
    tickers = pd.read_csv(dm.SM / "tickers.csv", encoding="utf-8-sig").drop_duplicates("cik").set_index("cik")
    progress = json.loads((dm.SM / "tiingo_progress.json").read_text(encoding="utf-8"))
    symbols = dm.issuer_symbols(dm.load_sec()[0])
    need: list[str] = []
    for r in z.itertuples():
        cik = int(r.cik)
        if cik in check.index and str(check.loc[cik, "price"]) == "없음":
            need += dm.candidate_tickers_for(cik, tickers, progress, symbols)
    need = list(dict.fromkeys(need))
    log(f"Tiingo 메타데이터 확인 대상 티커 {len(need)}개")
    try:
        meta = dm.fetch_tiingo_meta(need)
    except dm.TiingoLimit as exc:
        raise SystemExit(f"BLOCKED — Tiingo 한도로 메타데이터 확인 불가({exc})") from None
    st = gap_status(z, check, tickers, progress, meta, symbols)
    n = len(st)
    no = int((st["source"] == "없음").sum())
    rate = no / n if n else float("nan")
    by_year = st.groupby(st["confirm"].dt.year).size().to_dict()
    summary = {"signals": n, "by_year": {int(k): int(v) for k, v in by_year.items()}, "no_source": no, "gap_rate": rate,
               "from_cache": int((st["source"] == "캐시").sum()), "from_tiingo_meta": int((st["source"] == "Tiingo 메타").sum()),
               "tiingo_meta_tickers": len(need), "all_signals_before_zone": int(z.attrs["all_signals"]), "bad_trans_date": z.attrs["stats"]["bad_trans_date"],
               "p_rows": z.attrs["stats"]["p_rows"], "checked_utc": dm.now_utc()}
    GAP_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    log(json.dumps(summary, ensure_ascii=False))
    if n < MIN_SIGNALS:
        raise SystemExit(f"BLOCKED — 신호 수 {n} < {MIN_SIGNALS}")
    if rate > GAP_LIMIT:
        raise SystemExit(f"BLOCKED — 판정 불가 비율 {rate:.1%} > 10%")
    log("2절 점검 통과")
    return summary


# ── 시장 ──────────────────────────────────────────────────────────────────────


def build_market(prices: dict[int, pd.DataFrame], qqq: pd.Series, fx: pd.Series, zone: dm.Zone) -> bk.Market:
    """가격 → Market. 입력: {CIK: close·volume 표}, QQQ 종가, 일별 환율, Zone. 달력은 QQQ 거래일(시작 2017-01-03)."""
    assert_cutoff(qqq, fx, *prices.values())
    days = qqq.loc[START:].index
    q = qqq.reindex(days).to_numpy(float)
    f = fx.dropna().sort_index().reindex(days.union(fx.dropna().index)).ffill().reindex(days).to_numpy(float)
    if not (np.isfinite(q).all() and np.isfinite(f).all()):
        raise SystemExit("BLOCKED — QQQ·환율 결측")
    ciks = sorted(prices)
    T, N = len(days), len(ciks)
    px = np.full((T, N), np.nan)
    elig = np.zeros((T, N), bool)
    last_idx = np.full(N, -1, dtype=int)
    for j, c in enumerate(ciks):
        p = prices[c]
        cl = p["close"]
        s = cl.reindex(days.union(cl.index)).ffill().reindex(days)
        real_last = cl.index.max()
        s[days > real_last] = np.nan
        px[:, j] = s.to_numpy(float)
        inside = np.flatnonzero(days <= real_last)
        last_idx[j] = int(inside[-1]) if len(inside) else -1
        elig[:, j] = dm.member_flags(c, zone, days) & dm.liquid_flags(p, days)
    return bk.Market(days, q, f, px, last_idx, ciks, elig, bk.basket_returns(px, elig))


def confirm_position(confirm: pd.Timestamp, days: pd.DatetimeIndex) -> int:
    """신호 확인일 위치 — 휴장일이면 다음 거래일(해석 9)."""
    return int(np.searchsorted(days.values, np.datetime64(confirm), side="left"))


def classify(z: pd.DataFrame, m: bk.Market) -> tuple[pd.DataFrame, list[bk.Entry], list]:
    """구역 신호 → 상태 표·체결 후보. status: 후보/판정 불가/유동성 미달/기간 끝 체결 불가. 출력: (표, 후보, 판정 불가 신호 목록)."""
    entries, undecided, status = [], [], []
    for i, r in enumerate(z.itertuples()):
        cik = int(r.cik)
        _, ex = sg.use_and_exec(r.confirm, m.days)
        if ex is None:
            status.append("기간 끝 체결 불가")
            continue
        ci = confirm_position(r.confirm, m.days)
        j = m.col.get(cik)
        price_ok = False
        if j is not None:
            first = np.flatnonzero(np.isfinite(m.px[:, j]))
            price_ok = len(first) > 0 and first[0] <= ci and m.last_idx[j] >= ex
        if not price_ok:
            status.append("판정 불가")
            undecided.append({"row": i, "cik": cik, "confirm_idx": ci, "exec_idx": ex})
            continue
        if not m.eligible[ci, j]:
            status.append("유동성 미달")
            continue
        status.append("후보")
        entries.append(bk.Entry(ci, ex, j, key=i))
    return z.assign(status=status), entries, undecided


def with_synthetic(m: bk.Market, undecided: list, ret: float) -> tuple[bk.Market, list[bk.Entry]]:
    """판정 불가 신호마다 가상 가격 열을 붙인다(해석 20). 출력: (새 Market, 가상 진입)."""
    T = len(m.days)
    extra = np.full((T, len(undecided)), np.nan)
    last = np.zeros(len(undecided), dtype=int)
    entries = []
    for k, u in enumerate(undecided):
        e = u["exec_idx"]
        stop = min(e + bk.HOLD_DAYS, T - 1)
        extra[e:stop + 1, k] = (1.0 + ret) ** (np.arange(stop - e + 1) / bk.HOLD_DAYS)
        last[k] = stop
        entries.append(bk.Entry(u["confirm_idx"], e, len(m.ciks) + k, key=f"판정불가{k}"))
    px = np.hstack([m.px, extra]) if len(undecided) else m.px
    elig = np.hstack([m.eligible, np.zeros((T, len(undecided)), bool)]) if len(undecided) else m.eligible
    ciks = list(m.ciks) + [f"synthetic{k}" for k in range(len(undecided))]
    return bk.Market(m.days, m.qqq, m.fx, px, np.concatenate([m.last_idx, last]), ciks, elig, m.basket), entries


def order(entries: list[bk.Entry]) -> list[bk.Entry]:
    """체결일 → 확인일 → 열 순서."""
    return sorted(entries, key=lambda e: (e.exec_idx, e.confirm_idx, e.col))


def book_trio(entries: list[bk.Entry], m: bk.Market) -> tuple[dict, dict]:
    """전략과 그 비중 경로의 단순 보유 바스켓. 출력: (전략, 바스켓)."""
    s = bk.run_strategy(order(entries), m, want_path=True)
    return s, bk.run_basket(s["sat_weight"], s["cost_frac"], m)


def observed_p10(trades: list[dict]) -> float:
    """강제 청산이 아닌 매매 가격 수익률의 하위 10백분위(해석 20). 없으면 0."""
    r = [t["ret"] for t in trades if t["reason"] != "끝날 강제 청산"]
    return float(np.percentile(r, 10)) if r else 0.0


def excess_table(trades: list[dict], m: bk.Market) -> pd.DataFrame:
    """매매별 구역 ① 동일 가중 대비 초과수익(보고용). 출력: col, confirm_year, ret, basket_ret, excess."""
    cum = np.cumprod(1.0 + m.basket)
    rows = []
    for t in trades:
        b = cum[t["exit_idx"]] / cum[t["exec_idx"]] - 1.0
        rows.append({**t, "basket_ret": b, "excess": t["ret"] - b, "confirm_year": int(m.days[t["confirm_idx"]].year), "hold_days": t["exit_idx"] - t["exec_idx"]})
    return pd.DataFrame(rows)


def compute(z3: pd.DataFrame, z2: pd.DataFrame, z4: pd.DataFrame, prices: dict[int, pd.DataFrame], qqq: pd.Series, fx: pd.Series, zone: dm.Zone, n_random: int = bk.N_RANDOM) -> dict:
    """전체 계산. 입력: 3·2·4명 구역 신호, 가격, QQQ, 환율, Zone. 출력: 판정·장부·표 dict."""
    m = build_market(prices, qqq, fx, zone)
    t3, cand, und = classify(z3, m)
    main, basket = book_trio(cand, m)
    qqq_book = bk.run_strategy([], m)
    entered = main["trades"]
    done = [bk.Entry(t["confirm_idx"], t["exec_idx"], t["col"]) for t in entered]
    actual_by_conf: dict[int, set] = {}
    for e in cand:
        actual_by_conf.setdefault(e.confirm_idx, set()).add(e.col)
    rand = bk.random_control(done, actual_by_conf, m, n_random)
    p, r = bk.p_value(main["cagr"], rand)
    p10 = observed_p10(entered)
    m_c, syn = with_synthetic(m, und, p10)
    cons, cons_b = book_trio(cand + syn, m_c)
    var = {}
    for k, z in ((2, z2), (4, z4)):
        _, c, _ = classify(z, m)
        var[k] = book_trio(c, m)
    verdict = bk.judge(main, qqq_book, basket, p, cons, cons_b, var[2][0], var[2][1], var[4][0], var[4][1], len(entered))
    ex = excess_table(entered, m)
    status = t3["status"].tolist()
    for t in entered:
        status[t["key"]] = "진입"
    return {"market": m, "signals": t3.assign(status=status), "main": main, "basket": basket, "qqq": qqq_book, "cons": cons, "cons_basket": cons_b, "p10": p10, "var": var,
            "p": p, "r": r, "random": rand, "verdict": verdict, "excess": ex, "undecided": len(und), "n_cand": len(cand)}


# ── 산출물 ────────────────────────────────────────────────────────────────────


def pct(x, d=2):
    return "" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def md(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def ox(x) -> str:
    return "O" if x else "X"


def write_outputs(out: Path, res: dict, gap: dict, order_info: dict, manifest: list[dict], maxdates: dict, price_stats: dict) -> None:
    """산출물 5개. 덮어쓰기 금지(폴더가 있으면 FileExistsError)."""
    m = res["market"]
    out.mkdir(parents=True, exist_ok=False)
    sig = res["signals"].copy()
    sig["confirm"] = sig["confirm"].dt.strftime("%Y-%m-%d")
    sig["exec_day"] = pd.to_datetime(sig["exec_day"]).dt.strftime("%Y-%m-%d")
    for c in ("first_trade", "last_trade"):
        sig[c] = pd.to_datetime(sig[c]).dt.strftime("%Y-%m-%d")
    sig.to_csv(out / "signals.csv", index=False, encoding="utf-8-sig")

    ex = res["excess"]
    tr = pd.DataFrame(columns=["cik", "confirm", "buy", "sell", "reason", "hold_days", "ret", "basket_ret", "excess"])
    if len(ex):
        tr = pd.DataFrame({"cik": [m.ciks[c] for c in ex["col"]], "confirm": [m.days[i].strftime("%Y-%m-%d") for i in ex["confirm_idx"]],
                           "buy": [m.days[i].strftime("%Y-%m-%d") for i in ex["exec_idx"]], "sell": [m.days[i].strftime("%Y-%m-%d") for i in ex["exit_idx"]],
                           "reason": ex["reason"], "hold_days": ex["hold_days"], "ret": ex["ret"], "basket_ret": ex["basket_ret"], "excess": ex["excess"]})
    tr.to_csv(out / "trades.csv", index=False, encoding="utf-8-sig")

    label = ["시작"] + [m.days[d].strftime("%Y-%m") for d in res["main"]["curve_days"]]
    pd.DataFrame({"month": label, "strategy": res["main"]["curve"], "qqq": res["qqq"]["curve"], "same_exposure": res["basket"]["curve"],
                  "conservative": res["cons"]["curve"], "conservative_same_exposure": res["cons_basket"]["curve"]}).to_csv(out / "equity_curves.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(manifest).to_csv(out / "data_manifest.csv", index=False, encoding="utf-8-sig")

    v, c = res["verdict"], res["main"]["counts"]
    comp = pd.DataFrame([{"장부": n, "세후 연복리": pct(b["cagr"]), "끝 값(원)": f"{b['end_krw']:,.0f}", "MDD": pct(b["mdd"])} for n, b in [
        ("전략(주 결과)", res["main"]), ("QQQ 100%", res["qqq"]), ("같은 노출 단순 보유", res["basket"]), ("보수 결과", res["cons"]), ("보수의 같은 노출", res["cons_basket"]),
        ("변형 2명", res["var"][2][0]), ("변형 2명의 같은 노출", res["var"][2][1]), ("변형 4명", res["var"][4][0]), ("변형 4명의 같은 노출", res["var"][4][1])]])
    crit = pd.DataFrame([{"기준": n, "결과": ox(v[k])} for n, k in [("① >QQQ", "c1"), ("② >같은 노출", "c2"), (f"③ p ≤ 0.01 (p={res['p']:.4f}, r={res['r']})", "c3"), ("④ 낙폭", "c4"), ("⑤ 보수에서 ①·②", "c5"), ("⑥ 2·4명 변형 ②", "c6"), ("진입 ≥ 30", "enough")]])
    st_counts = res["signals"]["status"].value_counts().to_dict()
    years = "해당 없음"
    amount = "해당 없음"
    avg_ex = float("nan")
    avg_hold = float("nan")
    short = 0
    if len(ex):
        avg_ex, avg_hold, short = float(ex["excess"].mean()), float(ex["hold_days"].mean()), int((ex["hold_days"] < 42).sum())
        g = ex.groupby("confirm_year").agg(건수=("ret", "size"), 평균수익=("ret", "mean"), 평균초과=("excess", "mean")).reset_index()
        years = md(g.assign(평균수익=g["평균수익"].map(pct), 평균초과=g["평균초과"].map(pct)).rename(columns={"confirm_year": "신호 연도"}))
        amt = res["signals"].loc[[t["key"] for t in res["main"]["trades"]], "amount_usd"].to_numpy(float)
        hi = amt >= np.median(amt)
        exv = ex["excess"].to_numpy(float)
        amount = f"상위 절반 {pct(float(exv[hi].mean()))} ({int(hi.sum())}건), 하위 절반 {pct(float(exv[~hi].mean())) if (~hi).any() else '없음'} ({int((~hi).sum())}건)"
    w = res["main"]["sat_weight"]
    yrs = (m.days[-1] - m.days[0]).days / 365.25
    turnover = (len(ex) * bk.WEIGHT * 2) / yrs
    overall = max(pd.Timestamp(x) for x in maxdates.values())
    report = "\n".join([
        "# 내부자 집단 매수 전략 — 판정 보고서", "",
        "연구 작업이며 투자 조언이 아니다. 실제 돈 변경 제안 없음. 2022-01-01 이후 데이터는 받지도 계산하지도 않았다.", "",
        f"## 최종 판정: {v['result']}", "", "구역: 검증 ① 2/5 사용 (남은 기회 3/5).", "",
        "## 순서 증빙", "",
        f"- 사전 등록 커밋(태그 `{TAG}` = {order_info['tag_commit'][:7]}) 시각: {order_info['commit_time']}",
        f"- push 시각: {order_info['pushed_at']}", f"- 첫 가격 본문 수신 시각(UTC): {order_info['first_price']}",
        f"- 입력 최대 날짜 assert: 통과 — 전체 최대 {overall.date()} ≤ 2021-12-31 (입력 {len(maxdates)}개)", "",
        "## 2절 가격 공백 점검 (태그 전)", "",
        f"- 구역 ① 신호 {gap.get('signals')}개, 연도별 {gap.get('by_year')}, 판정 불가 비율 {pct(gap.get('gap_rate', float('nan')))}",
        f"- 출처: 캐시 {gap.get('from_cache')} · Tiingo 메타 {gap.get('from_tiingo_meta')} · 없음 {gap.get('no_source')} (메타 확인 티커 {gap.get('tiingo_meta_tickers')}개)", "",
        "## 합격 기준", "", md(crit), "", "## 비교", "", md(comp), "",
        f"보수 결과의 판정 불가 신호 수익률(하위 10백분위): {pct(res['p10'])}. 무작위 세후 연복리 중앙값 {pct(float(np.median(res['random'])))}, 95백분위 {pct(float(np.percentile(res['random'], 95)))}.", "",
        "## 건수", "",
        f"- 구역 ① 신호 {len(res['signals'])}, 상태별 {st_counts}",
        f"- 실제 진입 {len(ex)}, 보유 중 무시 {c['ignored_holding']}, 10종목 건너뜀 {c['skipped_full']}, 체결일 가격 없음 {c['no_price_at_exec']}",
        f"- 가격 끊김 매도 {c['delisted']}, 끝날 강제 청산 {c['forced_end']}, 판정 불가 {res['undecided']} (태그 뒤 실제 가격 기준)",
        f"- 가격 표 종목 {price_stats['ciks']}개 (기존 캐시 {price_stats['old']}, 태그 뒤 추가 {price_stats['new']})", "",
        "## 보고용 (판정에 쓰지 않음)", "",
        f"- 12개월 평균 초과수익(구역 ① 동일 가중 대비): {pct(avg_ex)}", "", years, "",
        f"- 평균 위성 비중 {pct(float(w.mean()))}, 연 회전율(한쪽 왕복 기준) {pct(turnover, 0)}, 평균 보유 거래일 {avg_hold:.1f}",
        f"- P8: 현금 >10% 아니오(항상 투자), 보유 <60일(42거래일 미만) {short}건, 회전 >100% {'예' if turnover > 1 else '아니오'}, 대형주 손절 아니오",
        f"- 내부자 매수 금액별 평균 초과수익: {amount}", "",
        "## 유의", "",
        "- 가격은 분할 반영·배당 미반영, 모든 비교 대상 같음. 원천징수 미적용.",
        "- 가격 공백은 인수·파산 종목에 몰려 있어 전략·바스켓·무작위 모두 공백 편향을 지닌다. 상장폐지 수익률 데이터가 없어 마지막 종가로 매도했다.",
        "- 사전 기대: 통과 가능성 10~15% (사전 등록 8절).", "",
        "## 파일", "", "`signals.csv`, `trades.csv`, `equity_curves.csv`, `data_manifest.csv`", ""])
    (out / "report.md").write_text(report, encoding="utf-8")


def load_prices_all(z3: pd.DataFrame, check: pd.DataFrame, tickers: pd.DataFrame, progress: dict, meta: dict, symbols: dict) -> tuple[dict, dict]:
    """가격 본문: 같은 회사 캐시 + 메타 통과 티커 추가 수신. 태그 push 뒤에만 부른다. 출력: ({CIK: 표}, 통계)."""
    st = gap_status(z3, check, tickers, progress, meta, symbols)
    new = 0
    for cik, g in st[st["source"] == "Tiingo 메타"].groupby("cik"):
        cik = int(cik)
        row = check.loc[cik].to_dict()
        tks = list(dict.fromkeys(t for s in g["tiingo"] for t in s.split(";")))
        try:
            fl = spc.filings_table(spc.sub_jsons(cik))
            ff = fl.loc[fl["form"].isin(["10-K", "10-Q"]), "filing_date"].min()
        except Exception as exc:  # noqa: BLE001 — 첫 정기 보고를 몰라도 날짜 대조는 한다(상장 첫 달 예외만 빠짐)
            log(f"CIK {cik} EDGAR 목록 실패 {type(exc).__name__}")
            ff = None
        if dm.fetch_tiingo_body(cik, row, tks, ff if isinstance(ff, str) else None):
            new += 1
    same = [int(c) for c in check.index[check["same_company"].astype("boolean").fillna(False)]]
    new_ciks = [int(p.stem) for p in dm.NEW_PRICES.glob("*.csv")] if dm.NEW_PRICES.exists() else []
    prices = {}
    for c in sorted(set(same) | set(new_ciks)):
        p = dm.read_cached_price(c)
        if p is not None and len(p):
            prices[c] = p
    return prices, {"ciks": len(prices), "new": new, "old": len(prices) - new}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-check", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--pushed-at", default="")
    args = ap.parse_args()
    if args.gap_check:
        gap_check()
        return
    if not args.run:
        ap.error("--gap-check 또는 --run")
    check_order()
    gap = json.loads(GAP_JSON.read_text(encoding="utf-8"))
    z3, z2, z4 = build_signals(3), build_signals(2), build_signals(4)
    if len(z3) != gap["signals"]:
        raise SystemExit(f"BLOCKED — 신호 수가 2절 점검({gap['signals']})과 다름({len(z3)})")
    first_price = dm.now_utc()
    log(f"가격 본문 수신 시작 {first_price}")
    failures = []
    for name, fn in (("QQQ", dm.fetch_qqq), ("DEXKOUS", dm.fetch_fx)):
        try:
            meta = fn()
            log(f"받음 {name}: {meta['first']} ~ {meta['last']} ({meta['rows']}행)")
        except Exception as exc:  # noqa: BLE001 — 실패 목록을 모아 보고
            failures.append(f"{name}: {type(exc).__name__}")
    if failures:
        raise SystemExit("BLOCKED — 수신 실패: " + "; ".join(failures))
    qqq, fx = dm.load_qqq(), dm.load_fx()
    cal = sg.nyse_days("2017-01-03", "2021-12-31")
    if not cal.equals(qqq.loc[START:].index):
        raise SystemExit("BLOCKED — NYSE 달력과 QQQ 거래일이 다름")
    check = dm.load_price_check()
    tickers = pd.read_csv(dm.SM / "tickers.csv", encoding="utf-8-sig").drop_duplicates("cik").set_index("cik")
    progress = json.loads((dm.SM / "tiingo_progress.json").read_text(encoding="utf-8"))
    tmeta = json.loads(dm.META_JSON.read_text(encoding="utf-8")) if dm.META_JSON.exists() else {}
    try:
        prices, stats = load_prices_all(z3, check, tickers, progress, tmeta, dm.issuer_symbols(dm.load_sec()[0]))
    except dm.TiingoLimit as exc:
        raise SystemExit(f"BLOCKED — Tiingo 한도로 가격 본문 수신 불가({exc})") from None
    maxdates = {"QQQ": qqq.index.max(), "DEXKOUS": fx.index.max(), "prices": max(p.index.max() for p in prices.values())}
    maxdates["공시(신호 확인일)"] = z2["confirm"].max()
    bad = {k: v for k, v in maxdates.items() if pd.Timestamp(v) > CUTOFF}
    assert not bad, f"2021-12-31 뒤 날짜: {bad}"
    log("입력 최대 날짜 assert 통과")
    res = compute(z3, z2, z4, prices, qqq, fx, dm.load_zone())
    info = {"tag_commit": git("rev-list", "-n", "1", TAG), "commit_time": git("log", "-1", "--format=%cI", TAG), "pushed_at": args.pushed_at or "(인자 없음)", "first_price": first_price}
    write_outputs(OUT, res, gap, info, dm.manifest_rows(), maxdates, stats)
    log(json.dumps({**res["verdict"], "p": res["p"], "cagr": res["main"]["cagr"], "qqq": res["qqq"]["cagr"], "basket": res["basket"]["cagr"], "entries": len(res["excess"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
