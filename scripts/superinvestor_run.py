"""슈퍼투자자 최고 확신 종목 전략(superinvestor) 검증 구역 ③ 판정 실행 — 사전 등록 태그 뒤에만 수익을 계산한다.

    python scripts/superinvestor_run.py run           # 판정 + 참고 보고 + 엑셀 (무작위 1,000회)
    python scripts/superinvestor_run.py run --quick   # 무작위 20회만(실행기 점검용, 판정 아님)

시작할 때 (1) 태그 superinvestor-prereg가 있는지, (2) 잠근 파일이 태그 내용과 같은지, (3) 연결표·확보율 표(git 제외 캐시)의 SHA-256이
사전 등록 문서에 적은 값과 같은지 확인하고 아니면 멈춘다. 규칙·해석은 docs/prereg/superinvestor.md 1장·3장 그대로다.
장부(원화 계좌·비용·세금·12월 공제 소진)는 mcap-growth 실행기(scripts/mcap_growth_run.Book·run_book, 잠긴 파일)를 시작·끝 날짜만 바꿔 쓴다.
결과 숫자는 data/cache/superinvestor/result.json(git 제외), 엑셀은 exports/superinvestor/(git 제외).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import mcap_growth_check as mc  # noqa: E402
import mcap_growth_run as mr  # noqa: E402
import superinvestor_check as sc  # noqa: E402
from core import superinvestor as si  # noqa: E402

TAG = "superinvestor-prereg"
PREREG = ROOT / "docs" / "prereg" / "superinvestor.md"
LOCKED = ["docs/prereg/superinvestor.md", "core/superinvestor.py", "scripts/superinvestor_check.py", "scripts/mcap_growth_run.py",
          "scripts/mcap_growth_check.py", "scripts/value_score_zone3_data.py", "scripts/value_score_zone3_check.py",
          "scripts/rs26_zone3_check.py"]
START = pd.Timestamp("2014-01-02")
END = pd.Timestamp("2021-12-31")
MID = pd.Timestamp("2017-12-29")
SEED, N_RANDOM = 20261030, 1000
RESULT = sc.OUT / "result.json"
EXPORT = ROOT / "exports" / "superinvestor"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_prereg() -> str:
    r = subprocess.run(["git", "rev-parse", "--verify", f"{TAG}^{{commit}}"], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"태그 {TAG}가 없습니다 — 사전 등록 전에는 수익을 계산하지 않습니다")
    for f in LOCKED:
        if subprocess.run(["git", "diff", "--quiet", TAG, "--", f], cwd=ROOT).returncode != 0:
            raise SystemExit(f"{f}가 태그 {TAG} 뒤에 바뀌었습니다 — 멈춥니다")
    doc = PREREG.read_text(encoding="utf-8")
    for p in (sc.LINK, sc.COVER):
        if sha256(p) not in doc:
            raise SystemExit(f"{p.name}의 SHA-256이 사전 등록 문서와 다릅니다 — 멈춥니다")
    return r.stdout.strip()


# ── 분기 입력 ────────────────────────────────────────────────────────────────


@dataclass
class Base:
    b: pd.Timestamp          # 분기 기준일(신호)
    i: int                   # 체결일 위치(첫 분기는 시작일 0)
    pool: list               # 대상: 기준일 직전 월말 구역 ③ 명단 중 그날 실제 종가 있는 종목
    sector: dict
    orders: dict             # 판 이름 → 순위 목록(대상만)
    tables: dict             # 판 이름 → 표 DataFrame
    managers: int
    judgeable: bool


def bases(data, m) -> list[Base]:
    """사전 등록 3-1·3-2: 보고 기준일 2013-09-30 ~ 2021-09-30 → 기준일. 첫 기준일(2013-11-15)의 선택은 시작일(2014-01-02) 종가에 체결."""
    subs = sc.load_subs()
    link = pd.read_csv(sc.LINK, dtype=str)
    c2t = dict(zip(link["cusip"], link["ticker"]))
    cov = pd.read_csv(sc.COVER, encoding="utf-8-sig", parse_dates=["period"]).set_index("period")
    gi = mc.Inputs.__new__(mc.Inputs)  # 섹터 규칙만 재사용(그 분기 위키 GICS → 전 판 GICS → SIC 대분류)
    gi.gics, gi.data = mc.load_gics(), data
    pos = pd.Series(np.arange(len(m.days)), index=m.days)
    pts = sc.periods()
    out = []
    for prev, p in zip(pts[:-1], pts[1:]):
        b = sc.base_date(p)
        if b > END:
            break
        i = 0 if b < START else int(pos[b])
        me = max(x for x in data.mes if x < b)
        pool = sorted(t for t in data.mem[me] if data.price_on(t, b) is not None)
        sector = {t: gi.sector(b, t)[0] for t in pool}
        hold = sc.period_holdings(p, prev)
        now, old, st = sc.qualified(subs, hold, p, b, prev)
        mem = set(pool)
        tables = {"main": si.votes(now, old, c2t, mem, k=5), "top3": si.votes(now, old, c2t, mem, k=3),
                  "noinc": si.votes(now, old, c2t, mem, k=5, require_increase=False), "cons": si.consensus(now, c2t, mem)}
        orders = {k: si.rank_order(v) for k, v in tables.items()}
        out.append(Base(b, i, pool, sector, orders, tables, st["managers"], bool(cov.at[p, "judgeable"])))
        print(f"[bases] {p.date()} → {b.date()} 체결 {m.days[i].date()} · 대상 {len(pool)} · 매니저 {st['managers']} · "
              f"표 받은 종목 {len(orders['main'])}", flush=True)
    if out[0].i != 0 or any(B.i == 0 for B in out[1:]):
        raise SystemExit("첫 기준일만 시작일에 체결돼야 합니다")
    return out


# ── 계획 ─────────────────────────────────────────────────────────────────────


def plan_strategy(bs: list[Base], key: str = "main") -> tuple[dict, list[dict]]:
    held: list[str] = []
    plan, log = {}, []
    for B in bs:
        if not B.judgeable:
            continue  # 판정 불가 분기: 새로 고르지 않고 보유 그대로
        sel = si.select(B.orders[key], B.sector, held)
        rank = {t: k + 1 for k, t in enumerate(B.orders[key])}
        tb = B.tables[key]
        for t in sel:
            log.append({"기준일": B.b.date(), "종목": t, "섹터": B.sector.get(t), "표 수": int(tb.at[t, "votes"]),
                        "표 준 매니저 수": int(tb.at[t, "votes"]), "표 준 매니저 비중 합": float(tb.at[t, "weight_sum"]),
                        "순위": rank[t], "유지/새로": "유지" if t in held else "새로", "조건 통과 매니저": B.managers,
                        "대상 수": len(B.pool)})
        plan[B.i] = sel
        held = sel
    return plan, log


def plan_ew(data, m, bs: list[Base]) -> dict:
    """구역 ③ 동일비중: 기준일마다(시작일 포함) 그날 직전 월말 명단 중 그날 종가 있는 전 종목 1/n."""
    out = {}
    for B in bs:
        d = m.days[B.i]
        me = max(x for x in data.mes if x < d)
        out[B.i] = sorted(t for t in data.mem[me] if data.price_on(t, d) is not None)
    return out


def plan_random(bs: list[Base], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {B.i: si.select_random(B.pool, B.sector, rng) for B in bs if B.judgeable}


_POOL: dict = {}


def _init(m, bs):
    mr.START, mr.END = START, END
    _POOL.update(m=m, bs=bs)


def _rand(i: int) -> float:
    return mr.run_book(_POOL["m"], plan_random(_POOL["bs"], SEED + i))["cagr"]


def main(quick: bool = False) -> None:
    import value_score_zone3_data as vd

    tag_commit = verify_prereg()
    t0 = time.time()
    mr.START, mr.END = START, END  # 장부 날짜만 바꾼다(Market·run_book이 모듈 값을 읽음)
    data = vd.Data()
    m = mr.Market(data, [])
    if m.days[0] != START or m.days[-1] != END:
        raise SystemExit("거래일 범위가 시작·끝과 다릅니다")
    bs = bases(data, m)
    res = {}
    plan_main, log = plan_strategy(bs)
    res["strategy"] = mr.run_book(m, plan_main, record=True)
    res["qqq"] = mr.run_book(m, {i: [] for i in plan_main}, record=True, qqq=True)
    ew = plan_ew(data, m, bs)
    res["ew"] = mr.run_book(m, ew, record=True)
    res["strategy_top3"] = mr.run_book(m, plan_strategy(bs, "top3")[0], record=True)
    res["ref_noinc"] = mr.run_book(m, plan_strategy(bs, "noinc")[0], record=True)
    res["ref_cons"] = mr.run_book(m, plan_strategy(bs, "cons")[0], record=True)
    res["strategy_zero"] = mr.run_book(m, plan_main, cost=0.0, tax=False)
    res["strategy_notax"] = mr.run_book(m, plan_main, tax=False)
    res["qqq_zero"] = mr.run_book(m, {i: [] for i in plan_main}, cost=0.0, tax=False, qqq=True)
    res["qqq_notax"] = mr.run_book(m, {i: [] for i in plan_main}, tax=False, qqq=True)
    res["ew_zero"] = mr.run_book(m, ew, cost=0.0, tax=False)
    res["ew_notax"] = mr.run_book(m, ew, tax=False)
    print(f"[run] 장부 계산 {time.time() - t0:.0f}초", flush=True)
    n_rand = 20 if quick else N_RANDOM
    with ProcessPoolExecutor(max_workers=8, initializer=_init, initargs=(m, bs)) as ex:
        rand = list(ex.map(_rand, range(n_rand), chunksize=10))
    print(f"[run] 무작위 {n_rand}회 {time.time() - t0:.0f}초", flush=True)

    S, Qq, E = res["strategy"], res["qqq"], res["ew"]
    r = np.array(rand)
    pct = 100.0 * (float((r < S["cagr"]).sum()) + 0.5 * float((r == S["cagr"]).sum())) / len(r)
    ms, mq = mr.mdd(S["curve"]), mr.mdd(Qq["curve"])
    segs = {"2014~17": (mr.seg_cagr(S["curve"], START, MID), mr.seg_cagr(E["curve"], START, MID)),
            "2018~21": (mr.seg_cagr(S["curve"], MID, END), mr.seg_cagr(E["curve"], MID, END))}
    T3 = res["strategy_top3"]
    crit = {
        "1": S["cagr"] > Qq["cagr"] and S["cagr"] > E["cagr"],
        "2": pct >= 95.0,
        "3": ms >= mq - 0.05,
        "4": T3["cagr"] > Qq["cagr"] and T3["cagr"] > E["cagr"],
        "5": all(a > b for a, b in segs.values()),
        "6": pct >= 99.0,
    }
    verdict = "PASS" if all(crit.values()) else "FAIL"
    windows = {"2018-10-01~12-24 하락": ("2018-10-01", "2018-12-24"), "2020-02-19~03-23 급락": ("2020-02-19", "2020-03-23"),
               "2020-03-23~06-30 반등": ("2020-03-23", "2020-06-30")}
    win = {k: (mr.seg_ret(S["curve"], a, b), mr.seg_ret(Qq["curve"], a, b)) for k, (a, b) in windows.items()}
    sec_max = [pd.Series([B.sector.get(t) or "미상" for t in plan_main[B.i]]).value_counts().iloc[0] / len(plan_main[B.i])
               for B in bs if plan_main.get(B.i)]
    pnl = pd.Series({m.tickers[j]: v for j, v in S["book"].pnl.items()})
    top5 = pnl.sort_values(ascending=False).head(5)
    out = {
        "tag_commit": tag_commit, "verdict": verdict, "criteria": crit, "n_random": n_rand,
        "cagr": {k: v["cagr"] for k, v in res.items()}, "final": {k: v["final"] for k, v in res.items()},
        "mdd": {k: mr.mdd(v["curve"]) for k, v in res.items() if v.get("curve") is not None},
        "random": {"pct": pct, "p50": float(np.percentile(r, 50)), "p95": float(np.percentile(r, 95)), "p99": float(np.percentile(r, 99))},
        "segments": segs, "windows": win,
        "concentration": {"sector_max_mean": float(np.mean(sec_max)), "sector_max_max": float(np.max(sec_max)),
                          "top5": {k: float(v) for k, v in top5.items()}, "top5_share": float(top5.sum() / pnl.sum()) if pnl.sum() else None,
                          "pnl_total": float(pnl.sum())},
        "taxes": {int(k): v for k, v in S["book"].taxes.items()}, "costs_usd": S["book"].costs_usd,
        "n_names": {str(B.b.date()): len(plan_main.get(B.i, [])) for B in bs},
        "turnover_new": {str(B.b.date()): sum(1 for x in log if x["기준일"] == B.b.date() and x["유지/새로"] == "새로") for B in bs},
        "managers": {str(B.b.date()): B.managers for B in bs},
        "voted": {str(B.b.date()): len(B.orders["main"]) for B in bs},
        "pool_size": {str(B.b.date()): len(B.pool) for B in bs},
        "yearly": {int(y): tuple(float(c[c.index.year == y].iloc[-1] / c[c.index <= f"{y - 1}-12-31"].iloc[-1] - 1)
                                 if (c.index <= f"{y - 1}-12-31").any() else None for c in (S["curve"], Qq["curve"], E["curve"]))
                   for y in range(2015, 2022)},
    }
    RESULT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    if not quick:
        EXPORT.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(EXPORT / "superinvestor_selections.xlsx") as w:
            pd.DataFrame(log).to_excel(w, sheet_name="분기별 15종목", index=False)
            pd.DataFrame([{"판": k, "세후 연복리": v["cagr"], "세후 청산액(원)": v["final"]} for k, v in res.items()]).to_excel(
                w, sheet_name="요약", index=False)
    print(json.dumps({k: out[k] for k in ("verdict", "criteria", "cagr", "mdd", "random", "segments")}, ensure_ascii=False,
                     indent=1, default=float))


if __name__ == "__main__":
    for st in (sys.stdout, sys.stderr):
        st.reconfigure(encoding="utf-8")
    main(quick="--quick" in sys.argv)
