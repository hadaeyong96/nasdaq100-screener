"""실적급등 장부 검증 보강 — 과거 구간 1회 계산(참고, 선택·수정에 쓰지 않음).

정의: docs/prereg/pead_watch_reporting.md(포트폴리오·다음 발표·무작위 10,000회), 생존자 편향 점검은 읽기·계산만.
신호·가격·비용·세금은 타이밍 탐색(B1·90)과 같은 데이터·함수(core/pead_timing, core/sw1)를 쓰고, 기준 재현이 틀리면 멈춘다.

실행:
  python -u -m scripts.pead_validation survivorship   # 2012~2021 명단 중 가격 없던 티커 표 (EDGAR submissions 읽기, 무료 가격 확인만)
  python -u -m scripts.pead_validation history        # 포트폴리오·다음 발표 전후·무작위 10,000회 (2012~2021, 2022~2026-09)
결과: docs/research/pead_validation_results.md (두 단계가 각자 자기 절을 쓴다)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from core import pead_timing as pt  # noqa: E402
from core import pead_validation as pv  # noqa: E402
from core import pead_watch as pw  # noqa: E402
from core import sealed_a as sa  # noqa: E402
from core import sw1  # noqa: E402

OUT_DIR = ROOT / "outputs"
RESULT_MD = ROOT / "docs" / "research" / "pead_validation_results.md"
SURV_JSON = OUT_DIR / "pead_validation_survivorship.json"
HIST_JSON = OUT_DIR / "pead_validation_history.json"
RATE, HOLD = 0.22, 90
SEED_OLD, SEED_NEW = 20261111, 20261112  # 타이밍 탐색 시드(재현) · 장부 시드(10,000회 보고)
BASE_KNOWN = {"main": (549, 0.67), "sealed": (336, -0.69)}
COSTS = sw1.Costs()

# CIK가 SW1 cik.csv에 없던 티커 — EDGAR 이름으로 확인한 값(survivorship 단계가 submissions 이름을 함께 출력한다)
CIK_EXTRA = {"BBBY": 886158, "CA": 356028, "CTRX": 1363851, "EMC": 790070, "QRTEA": 1355096, "RIMM": 1070235,
             "YHOO": 1011006}
# 가격 확인할 다른 기호(티커 변경·후속 회사). 확인만 하고 계산에 쓰지 않는다.
ALT_SYMBOLS = {"RIMM": ["BB", "BBRY"], "DISCA": ["WBD"], "QRTEA": ["QVCGA", "QRTEA"], "YHOO": ["AABA"],
               "LMCK": ["FWONK", "LSXMK"], "SNDK": ["SNDK"], "ALTR": ["ALTR"], "MON": ["MON"], "BBBY": ["BBBYQ"],
               "SHLD": ["SHLDQ"], "ENDP": ["ENDPQ", "NDOI"], "STRZA": ["STRZA"], "CA": ["CA"], "EMC": ["EMC"]}
EXIT_FORMS = ("25", "25-NSE", "15-12B", "15-12G", "15-15D", "DEFM14A", "SC TO-T", "S-4")
# 상태 요약 — 위 표의 근거(위키 사유·EDGAR 서식·마지막 공시)를 읽고 적은 것. 표에 근거를 같이 둔다.
STATUS = {
    "ALTR": "인수 합병(인텔), 상장 폐지 2015-12", "APC": "인수 합병(옥시덴탈), 상장 폐지 2019-08",
    "APOL": "지수 탈락 2012-12 → 비상장 전환 2017-02", "BBBY": "지수 탈락 2016-12 → 파산·상장 폐지 2023",
    "BMC": "비상장 전환 2013-09", "CA": "인수 합병(브로드컴), 상장 폐지 2018-11", "CTRX": "인수(유나이티드헬스), 상장 폐지 2015-07",
    "DISCA": "지수 탈락 2017-12, 상장 유지 → 2022 WBD로 합병·티커 변경", "DTV": "인수 합병(AT&T), 상장 폐지 2015-07",
    "EMC": "인수 합병(델), 상장 폐지 2016-09", "ENDP": "지수 탈락 2016-07 → 파산·상장 폐지 2022",
    "LIFE": "인수 합병(써모피셔), 상장 폐지 2014-02", "LMCK": "지수 탈락 2016-06, 2016-04 트래킹 주식으로 재편(기호 사라짐)",
    "MON": "인수 합병(바이엘), 상장 폐지 2018-06", "NYX": "인수 합병(ICE), 상장 폐지 2013-11",
    "QRTEA": "지수 탈락 2018-12, 상장 유지(기호 변경 QVCGA 2025)", "RIMM": "지수 탈락 2012-12, 상장 유지(블랙베리 BB, 외국 발행인 6-K)",
    "SHLD": "지수 탈락 2013-12 → 파산·상장 폐지 2018~19", "SNDK": "인수 합병(웨스턴디지털), 상장 폐지 2016-05",
    "SPLS": "지수 탈락 2015-12 → 비상장 전환 2017-09", "STRZA": "지수 탈락 2013-03 → 인수(라이온스게이트) 2016-12",
    "YHOO": "지수 탈락 2017-06(버라이즌 사업 매각 → 알타바 펀드, 2019 청산)",
}


def write_section(tag: str, lines: list[str]) -> None:
    """결과 문서의 한 절(<!-- tag --> ~ <!-- /tag -->)만 바꾼다."""
    head = ["# 실적급등 장부 검증 보강 — 과거 구간 결과 (참고)", "",
            "> 정의: `docs/prereg/pead_watch_reporting.md`, `docs/prereg/pead_watch_tags.md`. 원 사전 등록(`pead-watch-prereg`)·판정 규칙은 그대로다.",
            "> 과거 계산은 각 1회, 결과만 보고한다. 신호 선택·규칙 수정에 쓰지 않는다.", ""]
    txt = RESULT_MD.read_text(encoding="utf-8") if RESULT_MD.exists() else "\n".join(head)
    a, b = f"<!-- {tag} -->", f"<!-- /{tag} -->"
    block = "\n".join([a, *lines, b])
    if a in txt:
        txt = txt[:txt.index(a)] + block + txt[txt.index(b) + len(b):]
    else:
        txt = txt.rstrip("\n") + "\n\n" + block + "\n"
    RESULT_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULT_MD.write_text(txt.rstrip("\n") + "\n", encoding="utf-8", newline="\n")


def p(x, d: int = 2) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:+.{d}f}%"


# ── 1. 생존자 편향 점검 ──────────────────────────────────────────────────────


def wiki_removals(txt: str) -> dict[str, list[str]]:
    """나스닥100 변경표 위키 원문 → 빠진 티커 → [날짜 · 사유]."""
    out: dict[str, list[str]] = {}
    for row in txt.split("\n|-"):
        cells = [c.strip() for c in row.split("\n|")[1:]]
        if len(cells) < 6:
            continue
        date, removed, reason = cells[0], cells[3], cells[5]
        reason = pd.Series([reason]).str.replace(r"<ref.*?(</ref>|/>)", "", regex=True).str.replace(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", regex=True)[0]
        if removed:
            out.setdefault(removed, []).append(f"{date} · {reason.strip()}")
    return out


def free_price_check(syms: list[str], lo: pd.Timestamp, hi: pd.Timestamp, tiingo: pd.DataFrame | None) -> list[str]:
    """무료 출처에서 그 기간 가격이 있는지 확인만(받아 두지 않음). 출력: 사람이 읽는 줄 목록."""
    import yfinance as yf

    notes = []
    for s in syms:
        try:
            h = yf.Ticker(s).history(start=(lo - pd.Timedelta(days=40)).date().isoformat(),
                                     end=(hi + pd.Timedelta(days=40)).date().isoformat(), auto_adjust=False)
        except Exception as exc:  # noqa: BLE001 — 확인 결과로 적는다
            notes.append(f"yfinance {s}: 실패({type(exc).__name__})")
            continue
        if h is None or h.empty:
            notes.append(f"yfinance {s}: 없음")
        else:
            notes.append(f"yfinance {s}: {h.index[0].date()}~{h.index[-1].date()} {len(h)}행")
        if tiingo is not None:
            t = tiingo[tiingo["ticker"].str.upper() == s.upper()]
            for _, r in t.iterrows():
                notes.append(f"Tiingo 목록 {s}: {r['startDate']}~{r['endDate']} ({r['exchange']})")
    return notes


def tiingo_list() -> pd.DataFrame | None:
    """Tiingo 공개 지원 종목 목록(무료, 키 없음). 실패하면 None."""
    import io
    import zipfile

    import requests

    try:
        r = requests.get("https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip", timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            return pd.read_csv(z.open(z.namelist()[0]), dtype=str).fillna("")
    except Exception as exc:  # noqa: BLE001
        print(f"[survivorship] Tiingo 목록 실패({type(exc).__name__})")
        return None


def step_survivorship() -> None:
    from data import edgar
    from scripts import sw1_data_check as dc
    from scripts import sw1_experiments as sx

    u = sx.universe_tickers()
    miss = [t for t in u if sx.load_ohlc(t) is None]
    print(f"[survivorship] 명단 {len(u)} · 가격 없음 {len(miss)}", flush=True)
    nd, sp = dc.member_periods(dc.ndx_members()), dc.member_periods(dc.sp100_members())
    cik = pd.read_csv(dc.CACHE / "cik.csv").set_index("ticker")
    wiki = wiki_removals(dc.NDX_WIKITEXT.read_text(encoding="utf-8"))
    ua = edgar.get_user_agent()
    ti = tiingo_list()
    ann = pd.read_csv(dc.CACHE / "announcements.csv")
    pc = pd.read_csv(dc.PRICE_CHECK).set_index("ticker")
    rows = []
    for t in miss:
        c = cik.at[t, "cik"] if t in cik.index and pd.notna(cik.at[t, "cik"]) else CIK_EXTRA.get(t)
        per = [x for x in (nd.get(t), sp.get(t)) if x]
        lo = min(a for a, _ in per)
        hi = max(b for _, b in per)
        w_lo = max(lo, pd.Timestamp("2011-12-31"))
        w_hi = min((hi + pd.offsets.MonthEnd(1)).normalize(), pd.Timestamp("2021-12-31"))
        cand = [x for x in wiki.get(t, []) if (pd.Timestamp(x.split(" · ")[0]) - hi).days >= -5]
        wk = sorted(cand, key=lambda x: pd.Timestamp(x.split(" · ")[0]))[:1]
        rec = {"ticker": t, "status": STATUS.get(t, ""),
               "sw1_tiingo": str(pc.at[t, "status"]) + (f"·{pc.at[t, 'date_check']}" if pd.notna(pc.at[t, "date_check"]) else "")
               if t in pc.index else "—", "ndx": [str(x.date()) for x in nd[t]] if t in nd else None,
               "sp100": [str(x.date()) for x in sp[t]] if t in sp else None, "cik": int(c) if c else None,
               "wiki": wk, "sw1_ann": int((ann["ticker"] == t).sum())}
        if c:
            subs = dc.fetch_submissions(int(c), ua)
            name = subs[0].get("name")
            former = [f.get("name") for f in subs[0].get("formerNames", [])]
            r = dc.submissions_rows(subs)
            r["filing_date"] = r["filing_date"].astype(str)
            k8 = [{"form": f, "items": it, "filing_date": d, "accn": a} for f, it, d, a in
                  zip(r["form"], r["items"].fillna(""), r["filing_date"], r["accn"]) if f == "8-K"]
            inwin = [f for f in k8 if str(w_lo.date()) < f["filing_date"] <= str(w_hi.date())]
            raw202 = [f for f in inwin if pw.is_8k_202(f["form"], f["items"])]
            counted = [f for f in pw.counted_announcements(k8, 60) if str(w_lo.date()) < f["filing_date"] <= str(w_hi.date())]
            exits = r[r["form"].isin(EXIT_FORMS) & (r["filing_date"] >= str(lo.date()))]
            rec.update(edgar_name=name, former=former, n202=len(raw202), n202_counted=len(counted),
                       last_filing=str(r["filing_date"].max()), last_10q10k=str(r.loc[r["form"].isin(["10-Q", "10-K"]), "filing_date"].max()),
                       exit_forms=[f"{f} {d}" for f, d in zip(exits["form"], exits["filing_date"])][-4:])
        rec["prices"] = free_price_check(sorted({t, *ALT_SYMBOLS.get(t, [])}), w_lo, w_hi, ti)
        rows.append(rec)
        print(f"[survivorship] {t} 끝", flush=True)
    OUT_DIR.mkdir(exist_ok=True)
    SURV_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    tot = len(ann)
    n_c = sum(r.get("n202_counted", 0) for r in rows)
    L = ["## 1. 생존자 편향 점검 — 2012~2021 명단 중 가격이 없던 티커", "",
         f"- 명단(2011-12 ~ 2021-12 매달 말 나스닥100 ∪ S&P100) {len(u)}개 중 가격 없음 **{len(miss)}개** (타이밍 탐색 '279/301'과 같음).",
         f"- 이 티커들의 명단 기간 8-K 2.02(직전 발표 60일 규칙으로 센 것) 합 **{n_c}건** — SW1 실적 발표 표 {tot:,}건 대비 {n_c / tot:.1%}. "
         "이 발표들은 가격이 없어 신호 판정 자체를 못 했다(신호 비율이 같다고 보면 신호 636건 중 약 "
         f"{636 * n_c / tot:.0f}건 규모).",
         "- 기간 = 첫 달 말 ~ 마지막 달 말 + 1달(전 달 말 명단 규칙), 2012~2021로 자름. 8-K 2.02 = 정정 제외 8-K 중 Item 2.02.",
         "- 사유는 위키 나스닥100 변경표(빠질 때 적힌 문구)와 EDGAR 마지막 공시·상장 폐지 관련 서식(25·15·DEFM14A·S-4 등). S&P100만 있던 종목은 위키 사유 칸이 없다.", "",
         "| 티커 | EDGAR 이름 | 나스닥100 | S&P100 | 8-K 2.02 (센 것/전체) | 상태 | 마지막 10-Q·10-K | 근거 (위키 사유 · EDGAR 서식) | 무료 가격 확인 (SW1 때 Tiingo API) |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        nd_s = "~".join(r["ndx"]) if r["ndx"] else "—"
        sp_s = "~".join(r["sp100"]) if r["sp100"] else "—"
        why = "<br>".join(r["wiki"] + (r.get("exit_forms") or [])) or "—"
        px_notes = "<br>".join(r["prices"] + [f"(SW1 Tiingo: {r['sw1_tiingo']})"])
        L.append(f"| {r['ticker']} | {r.get('edgar_name') or '—'} | {nd_s} | {sp_s} | {r.get('n202_counted', '—')}/{r.get('n202', '—')} | "
                 f"{r['status']} | {r.get('last_10q10k', '—')} | {why} | {px_notes} |")
    L += ["", "- 가격 확인은 무료 출처(yfinance, Tiingo 공개 지원 목록)만 봤고, 받아 두거나 계산에 넣지 않았다. 결제 없음."]
    write_section("survivorship", L)
    print(f"[survivorship] 완료 → {RESULT_MD.relative_to(ROOT)}", flush=True)


# ── 2·3·5. 과거 구간 ─────────────────────────────────────────────────────────


class World:
    """한 구간의 달력·가격·환율·무작위 후보·실적 발표 반응일."""

    def __init__(self, days, al, bench, fx, pool_rows, first_year, n_years, ann_days, i_first, i_last):
        self.days, self.al, self.bench, self.fx = days, al, bench, fx
        names = sorted(al)
        self.M = (np.vstack([al[t].open for t in names]), np.vstack([al[t].close for t in names]),
                  np.vstack([al[t].lastv for t in names]))
        self.pool_rows, self.first_year, self.n_years = pool_rows, first_year, n_years
        self.years = days.year.to_numpy()
        self.ann_days, self.i_first, self.i_last = ann_days, i_first, i_last


def main_world():
    from scripts import sw1_data_check as dc
    from scripts import sw1_experiments as sx

    qqq = pd.read_parquet(dc.BT_CACHE / "QQQ.parquet")[["open", "close"]]
    if qqq.index[-1] > sx.SEAL:
        raise SystemExit("[pead-validation] QQQ 봉인일 뒤 행 — 멈춤")
    days = qqq.index
    k = sx.rebuild_announcements(days, qqq["close"])
    ref = pd.read_csv(dc.CACHE / "announcements.csv")
    sig = k[k["signal"]].copy()
    if len(sig) != 636 or set(sig["accn"]) != set(ref.loc[ref["signal"], "accn"]):
        raise SystemExit(f"[pead-validation] 신호 재현 불일치 {len(sig)}건 — 멈춤")
    sig["d0"] = pd.to_datetime(sig["d0"])
    sig = sig[["ticker", "d0", "accn", "in_ndx", "in_sp100"]]
    raw = {t: sx.load_ohlc(t) for t in sx.universe_tickers()}
    al = {t: pt.align(v, days) for t, v in raw.items() if v is not None}
    ndx, sp = dc.ndx_members(), dc.sp100_members()
    row_of = {t: i for i, t in enumerate(sorted(al))}

    def pool_rows(b, t):
        me = (pd.Timestamp(b) - pd.offsets.MonthEnd(1)).normalize()
        return np.array(sorted(row_of[x] for x in (ndx.get(me, set()) | sp.get(me, set())) - {t} if x in row_of), dtype=int)

    ann = k.dropna(subset=["d0"]).groupby("ticker")["d0"].apply(lambda s: sorted(pd.to_datetime(s))).to_dict()
    i_first = int(days.searchsorted(pd.Timestamp("2012-01-01")))
    i_last = int(days.searchsorted(pd.Timestamp("2021-12-31"), side="right")) - 1
    W = World(days, al, pt.align(qqq, days), sx.load_fx(days).to_numpy(), pool_rows, 2012, 11, ann, i_first, i_last)
    return sig, W


def sealed_world():
    from scripts import sealed_a_b1 as b1
    from scripts import sealed_a_sw1 as sas
    from scripts import sealed_a_sw1_data as sd

    b1.verify_preregistration()
    qq = sas.load_bench()
    days = qq.index
    k = sd.classify(days, qq["close"])
    v = k[k["status"] != "기간 밖"]
    prices, pc = sas.load_prices()
    fx, _ = sas.load_fx(days)
    sig = v[(v["status"] == "판정") & v["signal"]].copy()
    sig["d0"] = pd.to_datetime(sig["d0"])
    sig["sym"] = sig["ticker"]
    sig["ticker"] = sig["cik"].astype(int).astype(str)
    sig = sig.drop_duplicates(["ticker", "d0"])[["ticker", "d0", "sym", "in_ndx", "in_sp100"]]
    al = {c: pt.align(px[["open", "close"]], days) for c, px in prices.items()}
    m = sd.members()
    cik_of = {r["ticker"]: str(int(r["cik"])) for _, r in pc[pc["use"]].iterrows()}
    row_of = {t: i for i, t in enumerate(sorted(al))}

    def pool_rows(b, t):
        me = (pd.Timestamp(b) - pd.offsets.MonthEnd(1)).normalize()
        ts = m["ndx"].get(me, set()) | m["sp100"].get(me, set())
        return np.array(sorted(row_of[c] for c in {cik_of[x] for x in ts if x in cik_of} - {t} if c in row_of), dtype=int)

    kk = k.dropna(subset=["d0"]).copy()
    kk["c"] = kk["cik"].astype(int).astype(str)
    ann = kk.groupby("c")["d0"].apply(lambda s: sorted(pd.to_datetime(s))).to_dict()
    i_first = int(days.searchsorted(pd.Timestamp("2022-01-01")))
    i_last = int(days.searchsorted(sa.SEAL_LAST, side="right")) - 1
    W = World(days, al, pt.align(qq, days), fx.to_numpy(), pool_rows, 2022, 5, ann, i_first, i_last)
    return sig, W


def period_report(name: str, sig: pd.DataFrame, W: World) -> dict:
    """B1·90 매매 → 기준 재현 확인 → 무작위(1,000·10,000)·포트폴리오·다음 발표 전후."""
    tr, sk = pt.make_trades(sig, W.al, W.days, "B1", HOLD)
    T = pt.price_trades(tr, W.bench, W.fx, "B1", COSTS, RATE)
    n, mean = len(T), float(T["ex_after"].mean() * 100)
    if (n, round(mean, 2)) != BASE_KNOWN[name]:
        raise SystemExit(f"[pead-validation] {name} B1·90 재현 불일치 ({n}건 {mean:+.2f}%) — 멈춤")
    print(f"[pead-validation] {name} B1·90 재현 {n}건 {mean:+.2f}%", flush=True)
    from scripts import sw1_experiments as sx

    pools = [W.pool_rows(b, t) for b, t in zip(T["buy_day"], T["ticker"])]
    O, C, L = W.M
    cands = pt.candidates(T, pools, O, C, L, W.bench, W.fx, W.years, "B1", COSTS)
    v = float(T["ex_after"].mean())
    r1 = pv.random_summary(v, sx.random_means(cands, 1000, SEED_OLD, RATE, W.first_year, W.n_years))
    r10 = pv.random_summary(v, sx.random_means(cands, 10_000, SEED_NEW, RATE, W.first_year, W.n_years))
    bs = pv.bootstrap_ci(T["ex_after"].to_numpy(), 10_000, SEED_NEW)
    print(f"[pead-validation] {name} 무작위 끝", flush=True)

    trades = pv.trade_dicts(T, W.days, W.bench)
    rows, ev = pv.simulate(trades, W.al, W.bench, W.days, W.i_first, W.i_last, float(W.fx[W.i_first]))
    S = pv.pf_summary(rows, float(W.fx[W.i_last]))
    yearly = rows.assign(y=pd.to_datetime(rows["date"]).dt.year).groupby("y").agg(v=("value_usd", "last"), q=("qqq100_usd", "last"))
    prev = pd.concat([pd.Series({"v": rows["value_usd"].iloc[0], "q": rows["qqq100_usd"].iloc[0]}).to_frame().T, yearly.iloc[:-1]])
    yr = pd.DataFrame({"pf": (yearly["v"].to_numpy() / prev["v"].to_numpy() - 1) * 100,
                       "qqq": (yearly["q"].to_numpy() / prev["q"].to_numpy() - 1) * 100}, index=yearly.index)
    S.update(n_buy=int((ev["action"] == "buy").sum()), n_skip=int((ev["action"] == "skip").sum()),
             skip_why=ev.loc[ev["action"] == "skip", "note"].value_counts().to_dict(),
             yearly={int(k): {"pf": float(a), "qqq": float(b)} for k, (a, b) in yr.iterrows()})
    print(f"[pead-validation] {name} 포트폴리오 끝", flush=True)

    sp = []
    no_next = 0
    for r in T.itertuples(index=False):
        dn = pv.next_announcement(W.ann_days.get(r.ticker, []), W.days[int(r.ib)], W.days[int(r.ie)])
        if dn is None:
            no_next += 1
            continue
        i_n = int(W.days.get_loc(dn))
        sp.append(pv.split_at_next(W.al[r.ticker], W.bench, int(r.ib), int(r.ie), float(r.buy_px), float(r.sell_px), i_n))
    D = pd.DataFrame(sp)
    split = {"n": int(len(D)), "no_next": no_next}
    for c in ("ex_a", "ex_b", "ex_day"):
        x = D[c] * 100
        split[c] = {"mean": float(x.mean()), "median": float(x.median()), "win": float((x > 0).mean() * 100)}
    split["len_a"], split["len_b"] = float(D["len_a"].mean()), float(D["len_b"].mean())
    return {"n": n, "mean": mean, "r1000": r1, "r10000": r10, "boot": bs, "pf": S, "split": split,
            "skip": sk["skip"].value_counts().to_dict()}


def step_history() -> None:
    out = {}
    for name, fn in (("main", main_world), ("sealed", sealed_world)):
        sig, W = fn()
        out[name] = period_report(name, sig, W)
    OUT_DIR.mkdir(exist_ok=True)
    HIST_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    lab = {"main": "2012~2021", "sealed": "2022~2026-09"}
    L = ["## 2. 포트폴리오 성적 (가상 1억, 신호마다 남은 현금 1/10, 남는 현금 = QQQ) — 참고", "",
         "- 주 장부 규칙 B1·90 매매(2012~2021 549건 +0.67%, 2022~2026-09 336건 −0.69% 재현 확인)를 한 계좌에. 세금 전, 수수료 0.07%·시작 환전 0.1%.",
         "- 원화 = 달러 × 마지막 날 환율(나갈 때 환전 비용 없음). QQQ 100%도 같은 시작 달러·같은 환율. 가격은 분할만 반영(배당 제외 — 종목·QQQ 같음), 2026은 9월 말까지.", "",
         "| 구간 | 최종 평가액 | QQQ 100% | QQQ 대비 | 연 환산 (QQQ) | 최대 낙폭 (QQQ) | 매수 / 건너뜀 | 평균 보유 종목 · 주식 비중 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, r in out.items():
        s = r["pf"]
        L.append(f"| {lab[k]} | {s['end_krw'] / 1e8:.2f}억 원 | {s['qqq100_krw'] / 1e8:.2f}억 원 | {p(s['vs_qqq_pct'])} | "
                 f"{p(s['cagr'])} ({p(s['qqq_cagr'])}) | {p(s['mdd'], 1)} ({p(s['qqq_mdd'], 1)}) | {s['n_buy']} / {s['n_skip']} | "
                 f"{s['avg_n_pos']:.1f}개 · {s['avg_stock_share']:.0f}% |")
    for k, r in out.items():
        s = r["pf"]
        L += ["", f"{lab[k]} 연도별 (포트폴리오 / QQQ 100%): " + " · ".join(
            f"{y} {p(v['pf'], 1)} / {p(v['qqq'], 1)}" for y, v in s["yearly"].items())]
        if s["skip_why"]:
            L.append(f"- 건너뜀 사유: {s['skip_why']}")
    L += ["", "## 3. 다음 실적 발표 전·후 (B1·90 매매, QQQ 대비 초과, 달러·비용 전) — 참고", "",
          "- A = 매수일 시가 → 다음 발표 반응일 전날 종가 · B = 그 종가 → 매도일 종가(다음 발표 반응일 포함) · 반응일 하루 = B 중 그날만.",
          "- 다음 발표 = 각 구간 데이터 조사의 실적 발표 반응일(매수일 < 반응일 ≤ 실제 매도일 중 첫째).", "",
          "| 구간 | 다음 발표 있음 / 없음 | A 평균 · 중앙값 · 승률 (평균 길이) | B 평균 · 중앙값 · 승률 (평균 길이) | 반응일 하루 평균 · 중앙값 · 승률 |",
          "| --- | --- | --- | --- | --- |"]
    for k, r in out.items():
        s = r["split"]
        f = lambda c: f"{p(s[c]['mean'])} · {p(s[c]['median'])} · {s[c]['win']:.0f}%"  # noqa: E731
        L.append(f"| {lab[k]} | {s['n']} / {s['no_next']} | {f('ex_a')} ({s['len_a']:.0f}일) | {f('ex_b')} ({s['len_b']:.0f}일) | {f('ex_day')} |")
    L += ["", "## 5. 무작위 비교 1,000회 vs 10,000회 · 95% 구간 (B1·90, 세금 뒤) — 참고", "",
          f"- 1,000회 = 타이밍 탐색과 같은 시드 {SEED_OLD}(재현), 10,000회 = 장부 시드 {SEED_NEW}. 무작위 = 같은 매수일·예정 매도일, 전 달 말 명단 1종목.",
          "- 신호 95% 구간 = 매매 10,000번 복원 추출 평균의 2.5~97.5 백분위수.", "",
          "| 구간 | 매매 | 신호 평균 (95% 구간) | 1,000회 백분위 · 무작위 중앙값 · 95% 구간 | 10,000회 백분위 · 무작위 중앙값 · 95% 구간 |",
          "| --- | --- | --- | --- | --- |"]
    for k, r in out.items():
        a, b = r["r1000"], r["r10000"]
        L.append(f"| {lab[k]} | {r['n']} | {p(r['mean'])} ({p(r['boot'][0] * 100)} ~ {p(r['boot'][1] * 100)}) | "
                 f"{a['pct']:.1f} · {p(a['median'] * 100)} · {p(a['lo'] * 100)} ~ {p(a['hi'] * 100)} | "
                 f"{b['pct']:.1f} · {p(b['median'] * 100)} · {p(b['lo'] * 100)} ~ {p(b['hi'] * 100)} |")
    write_section("history", L)
    print(f"[pead-validation] 완료 → {RESULT_MD.relative_to(ROOT)}", flush=True)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "survivorship":
        step_survivorship()
    elif cmd == "history":
        step_history()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
