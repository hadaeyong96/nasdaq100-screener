"""봉인 ⓐ 개봉 — 후보 ② 실적급등 다음날 매수·3달 보유(대형) 데이터 확인 (사전 등록: docs/sealed_a_plan.md S-1~S-9).

수익 계산 없음. 명단 확보율·발표 시각·가격 확보율·판정 불가 비율·신호 건수만 본다.
코드는 SW1 데이터 조사(scripts/sw1_data_check.py)의 순수 함수를 그대로 쓰고, 캐시는 data/cache/sealed_a/sw1/에 따로 둔다
(기존 SW1 캐시는 2021-12-31 봉인 기준이라 섞지 않는다). 가격·지수에 2026-09-30 뒤 행이 있으면 멈춘다.

단계:
  --step members   나스닥100(시점별) · S&P 100(위키 옛 판) 매달 말 명단 2021-12 ~ 2026-08
  --step events    명단 티커 → CIK → EDGAR 제출 기록(새로 받음) → 실적 발표 표
  --step accepted  실적 발표마다 공시 색인 Accepted(동부시간)
  --step prices    yfinance auto_adjust=False 시가·종가·거래량 2021-01-01 ~ 2026-09-30, 같은 회사 기간 확인
  --step check     반응일·판정/판정 불가·신호 건수, 판정 불가 비율(⑦)과 공백 목록
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from core import sealed_a as sa  # noqa: E402
from scripts import sw1_data_check as dc  # noqa: E402

CACHE = ROOT / "data" / "cache" / "sealed_a" / "sw1"
SUB_CACHE = CACHE / "submissions"
PRICE_CACHE = CACHE / "prices"
ACCEPTED = CACHE / "accepted.json"
FIRST_MONTH, LAST_MONTH = "2021-12", "2026-08"
EV_FIRST, EV_LAST = "2021-12-01", "2026-09-30"
D0_FIRST = pd.Timestamp("2022-01-01")
SEAL_LAST = sa.SEAL_LAST
QQQ_CSV = ROOT / "data" / "cache" / "sealed_a" / "QQQ.csv"
UNDECIDED_STATUS = ["CIK 없음", "가격 없음·다른 회사", "반응일 가격 없음·자료 부족", "발표 시각 없음"]


def month_ends() -> list[pd.Timestamp]:
    return [p.to_timestamp(how="end").normalize() for p in pd.period_range(FIRST_MONTH, LAST_MONTH, freq="M")]


# ── 1) 명단 ──────────────────────────────────────────────────────────────────


def step_members() -> None:
    """매달 말 나스닥100·S&P 100 명단 → members.csv (month_end, index, tickers)."""
    from data import universe_history as uh

    CACHE.mkdir(parents=True, exist_ok=True)
    cur = set(pd.read_csv(ROOT / "data" / "universe_fallback.csv", comment="#")["ticker"])
    rows = [{"month_end": me.date(), "index": "ndx", "n": len(s), "tickers": ";".join(sorted(s))}
            for me in month_ends() for s in [set(uh.point_in_time_universe(me.date(), cur))]]

    revs, cont = [], {}
    while True:
        r = dc._wiki_get({"action": "query", "prop": "revisions", "titles": "S&P 100", "rvlimit": 500,
                          "rvstart": "2026-10-01T00:00:00Z", "rvend": "2021-01-01T00:00:00Z", "rvdir": "older",
                          "rvprop": "ids|timestamp", **cont})
        revs += r["query"]["pages"][0]["revisions"]
        if "continue" not in r:
            break
        cont = r["continue"]
    rv = pd.DataFrame(revs)
    rv["ts"] = pd.to_datetime(rv["timestamp"]).dt.tz_localize(None)
    rv = rv.sort_values("ts")
    dc.WIKI_CACHE.mkdir(parents=True, exist_ok=True)
    for me in month_ends():
        r = rv[rv["ts"] < me + pd.Timedelta(days=1)].iloc[-1]
        path = dc.WIKI_CACHE / f"{r['revid']}.txt"
        if not path.exists():
            j = dc._wiki_get({"action": "query", "prop": "revisions", "revids": int(r["revid"]),
                              "rvprop": "content", "rvslots": "main"})
            path.write_text(j["query"]["pages"][0]["revisions"][0]["slots"]["main"]["content"], encoding="utf-8")
            time.sleep(1.5)
        tick = dc.parse_sp100_wikitext(path.read_text(encoding="utf-8"))
        rows.append({"month_end": me.date(), "index": "sp100", "n": len(tick), "tickers": ";".join(sorted(tick)),
                     "revid": r["revid"], "rev_age_days": (me - r["ts"].normalize()).days})
    out = pd.DataFrame(rows)
    out.to_csv(CACHE / "members.csv", index=False, encoding="utf-8-sig")
    for k, g in out.groupby("index"):
        allt = set(";".join(g["tickers"]).split(";"))
        print(f"[members] {k}: 달 {len(g)} · 명단 수 {g['n'].min()}~{g['n'].max()} · 한 번이라도 {len(allt)}")
    sp = out[out["index"] == "sp100"]
    print(f"[members] S&P 100 옛 판 나이 중앙값 {sp['rev_age_days'].median():.0f}일, 최대 {sp['rev_age_days'].max()}일")
    u = set(";".join(out["tickers"]).split(";"))
    print(f"[members] 합집합 티커 {len(u)}")


def members() -> dict[str, dict[pd.Timestamp, set[str]]]:
    m = pd.read_csv(CACHE / "members.csv", parse_dates=["month_end"])
    return {k: {r["month_end"]: set(str(r["tickers"]).split(";")) for _, r in g.iterrows()} for k, g in m.groupby("index")}


def member_months() -> pd.DataFrame:
    """티커 → 처음·마지막 달 말, 대상 달 수(합집합), 나스닥100·S&P100 여부."""
    m = members()
    rec: dict[str, dict] = {}
    for k, mm in m.items():
        for me, ts in mm.items():
            for t in ts:
                r = rec.setdefault(t, {"months": set(), "ndx": False, "sp100": False})
                r["months"].add(me)
                r[k] = True
    return pd.DataFrame([{"ticker": t, "first": min(r["months"]), "last": max(r["months"]), "n_months": len(r["months"]),
                          "in_ndx": r["ndx"], "in_sp100": r["sp100"]} for t, r in sorted(rec.items())])


# ── 2) CIK·제출 기록 ─────────────────────────────────────────────────────────


def fetch_submissions(cik: int, ua: str) -> list[dict]:
    """submissions 본문 + 2021-01 이후와 겹치는 추가 파일 (새로 받아 sealed_a 캐시에)."""
    import requests

    SUB_CACHE.mkdir(parents=True, exist_ok=True)
    out, names = [], [f"CIK{cik:010d}.json"]
    while names:
        name = names.pop(0)
        path = SUB_CACHE / name
        if path.exists():
            j = json.loads(path.read_text(encoding="utf-8"))
        else:
            r = requests.get(dc.SUB_URL.format(name=name), headers={"User-Agent": ua}, timeout=60)
            time.sleep(0.12)
            r.raise_for_status()
            j = r.json()
            path.write_text(json.dumps(j), encoding="utf-8")
        out.append(j)
        if "filings" in j:
            names += [f["name"] for f in j["filings"].get("files", []) if f.get("filingTo", "9999") >= "2021-01-01"]
    return out


def step_events() -> None:
    """명단 티커 → CIK(후보: SW1 cik.csv·S&P 500 identity·SEC 현재 목록·바뀐 티커·이름 검색) → events.csv, cik.csv."""
    from dotenv import load_dotenv

    from data import edgar

    load_dotenv(ROOT / ".env")
    ua = edgar.get_user_agent()
    print(f"[events] SEC_USER_AGENT 있음: {bool(ua)}")
    mm = member_months()
    cmap = edgar.fetch_ticker_to_cik(force_refresh=True)
    old = pd.read_csv(dc.CACHE / "cik.csv").dropna(subset=["cik"]).set_index("ticker")["cik"].astype(int).to_dict()
    ident = pd.read_csv(dc.SP500_IDENTITY, index_col=0) if dc.SP500_IDENTITY.exists() else pd.DataFrame()
    rn = dc.renames()
    names = dc.expected_names()
    rows, ev, failed = [], [], []
    for i, r in mm.iterrows():
        t, a, b = r["ticker"], r["first"], r["last"]
        cands = []

        def add(c, src):
            if c is not None and not pd.isna(c) and all(int(c) != x for x, _ in cands):
                cands.append((int(c), src))

        add(cmap.get(t.upper().replace("-", "."), cmap.get(t.upper())), "SEC 티커 목록(현재)")
        add(old.get(t), "SW1 cik.csv")
        if t in ident.index:
            add(ident.loc[t, "cik"], "S&P 500 점검 identity.csv")
        if rn.get(t):
            add(cmap.get(rn[t].replace("-", ".")), f"SEC 티커 목록(바뀐 티커 {rn[t]})")
        pick = None
        for cik, src in cands:
            try:
                er = dc.pick_events(dc.submissions_rows(fetch_submissions(cik, ua)), "2021-07-01", EV_LAST)
            except Exception as exc:  # noqa: BLE001 — 실패는 모아서 보고
                failed.append((t, f"{src} CIK {cik} 받기 실패: {type(exc).__name__}"))
                continue
            q = er[er["kind"].isin(["10-Q", "10-K"])]
            n_in = ((q["filing_date"] >= (a - pd.Timedelta(days=150)).strftime("%Y-%m-%d"))
                    & (q["filing_date"] <= (b + pd.Timedelta(days=120)).strftime("%Y-%m-%d"))).sum()
            if n_in >= 2:  # SW1과 같은 확인: 명단 기간(앞뒤 여유)에 10-Q·10-K 2건 이상
                pick = (cik, src, er)
                break
        if pick is None and t in names:
            for nm in sorted(names[t]):
                for cik, _ in dc.name_search(nm, ua):
                    if any(cik == x for x, _ in cands):
                        continue
                    try:
                        js = fetch_submissions(cik, ua)
                    except Exception:  # noqa: BLE001
                        continue
                    er = dc.pick_events(dc.submissions_rows(js), "2021-07-01", EV_LAST)
                    q = er[er["kind"].isin(["10-Q", "10-K"])]
                    n_in = ((q["filing_date"] >= (a - pd.Timedelta(days=150)).strftime("%Y-%m-%d"))
                            & (q["filing_date"] <= (b + pd.Timedelta(days=120)).strftime("%Y-%m-%d"))).sum()
                    ed = [js[0].get("name", "")] + [f.get("name", "") for f in js[0].get("formerNames", [])]
                    if n_in >= 2 and max(dc.name_sim(nm, x) for x in ed) >= dc.NAME_SIM_MIN:
                        pick = (cik, f"이름 검색({nm})", er)
                        break
                if pick:
                    break
        base = {"ticker": t, "first": a.date(), "last": b.date(), "n_months": r["n_months"],
                "in_ndx": r["in_ndx"], "in_sp100": r["in_sp100"]}
        if pick is None:
            failed.append((t, "CIK 없음 또는 명단 기간 10-Q·10-K 없음(외국 기업 등)"))
            rows.append({**base, "cik": None, "cik_source": "", "edgar_name": "", "edgar_tickers": ""})
            continue
        cik, src, er = pick
        j0 = fetch_submissions(cik, ua)[0]
        rows.append({**base, "cik": cik, "cik_source": src, "edgar_name": j0.get("name", ""),
                     "edgar_tickers": ";".join(j0.get("tickers", []))})
        ev.append(er.assign(ticker=t, cik=cik))
        if (i + 1) % 50 == 0:
            print(f"[events] {i + 1}/{len(mm)}", flush=True)
    pd.DataFrame(rows).to_csv(CACHE / "cik.csv", index=False, encoding="utf-8-sig")
    pd.concat(ev, ignore_index=True).to_csv(CACHE / "events.csv", index=False, encoding="utf-8-sig")
    ok = sum(x["cik"] is not None for x in rows)
    print(f"[events] 대상 티커 {len(rows)} · CIK 확정 {ok} · 실패 {len(failed)}")
    for t, why in failed:
        print(f"  실패 {t}: {why}")


def earnings_table() -> pd.DataFrame:
    """CIK 있는 티커의 실적 발표(SW1 earnings_dates 그대로), 제출일 2021-12-01 ~ 2026-09-30, 전 달 말 명단에 있던 것만.

    같은 CIK를 두 티커가 쓰면(티커 변경 등) 같은 발표는 1건만 (CIK·accn 기준).
    """
    ev = pd.read_csv(CACHE / "events.csv", dtype={"report_date": str})
    m = members()
    rows = []
    for t, g in ev.groupby("ticker"):
        e = dc.earnings_dates(g)
        e = e[(e["filing_date"] >= EV_FIRST) & (e["filing_date"] <= EV_LAST)]
        rows.append(e.assign(ticker=t, cik=int(g["cik"].iloc[0])))
    e = pd.concat(rows, ignore_index=True)
    me = (pd.to_datetime(e["filing_date"]) - pd.offsets.MonthEnd(1)).dt.normalize()
    e["in_ndx"] = [t in m["ndx"].get(x, set()) for t, x in zip(e["ticker"], me)]
    e["in_sp100"] = [t in m["sp100"].get(x, set()) for t, x in zip(e["ticker"], me)]
    e = e[e["in_ndx"] | e["in_sp100"]]
    return e.sort_values(["cik", "accn", "ticker"]).drop_duplicates(["cik", "accn"]).reset_index(drop=True)


# ── 3) 발표 시각 ────────────────────────────────────────────────────────────


def step_accepted() -> None:
    """실적 발표마다 색인 페이지 Accepted. SW1 accepted.json에 같은 accn이 있으면 그 값을 옮겨 쓴다."""
    import requests
    from dotenv import load_dotenv

    from data import edgar

    load_dotenv(ROOT / ".env")
    ua = edgar.get_user_agent()
    e = earnings_table()
    cache = json.loads(ACCEPTED.read_text(encoding="utf-8")) if ACCEPTED.exists() else {}
    old = json.loads(dc.ACCEPTED_CACHE.read_text(encoding="utf-8")) if dc.ACCEPTED_CACHE.exists() else {}
    for a in e["accn"]:
        if a not in cache and a in old:
            cache[a] = old[a]
    todo = [(c, a) for c, a in zip(e["cik"], e["accn"]) if a not in cache]
    print(f"[accepted] 실적 발표 {len(e)} · 있음 {len(e) - len(todo)} · 받을 것 {len(todo)}", flush=True)
    failed = []
    for i, (cik, acc) in enumerate(todo):
        u = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{acc}-index.htm"
        try:
            r = requests.get(u, headers={"User-Agent": ua}, timeout=30)
            v = dc.parse_index_accepted(r.text) if r.status_code == 200 else None
            why = "" if v else f"HTTP {r.status_code}"
        except Exception as exc:  # noqa: BLE001 — 실패는 모아 보고
            v, why = None, type(exc).__name__
        time.sleep(0.12)
        if v:
            cache[acc] = v
        else:
            failed.append((acc, why))
        if (i + 1) % 200 == 0:
            ACCEPTED.write_text(json.dumps(cache), encoding="utf-8")
            print(f"[accepted] {i + 1}/{len(todo)} · 실패 {len(failed)}", flush=True)
    ACCEPTED.write_text(json.dumps(cache), encoding="utf-8")
    print(f"[accepted] 완료 · 실패 {len(failed)}")
    for a, why in failed[:30]:
        print(f"  실패 {a}: {why}")


# ── 4) 가격 ─────────────────────────────────────────────────────────────────


def price_tickers(r: pd.Series) -> list[str]:
    """yfinance로 시도할 티커: 명단 티커, 바뀐 티커, EDGAR 현재 티커 (순서대로, 중복 제외)."""
    out = [r["ticker"]]
    rn = dc.renames().get(r["ticker"])
    if rn:
        out.append(rn)
    for x in str(r.get("edgar_tickers") or "").split(";"):
        x = x.strip().replace(".", "-")
        if x and x != "nan":
            out.append(x)
    return list(dict.fromkeys(out))


def fetch_yf(sym: str) -> pd.DataFrame | None:
    """yfinance auto_adjust=False 시가·종가·거래량 2021-01-01 ~ 2026-09-30 (캐시). 없으면 빈 파일."""
    PRICE_CACHE.mkdir(parents=True, exist_ok=True)
    path = PRICE_CACHE / f"{sym}.csv"
    if not path.exists():
        import yfinance as yf

        try:
            h = yf.Ticker(sym).history(start="2021-01-01", end="2026-10-01", auto_adjust=False)
        except Exception:  # noqa: BLE001 — 실패는 공백으로 보고
            h = pd.DataFrame()
        if h is None or h.empty:
            pd.DataFrame(columns=["open", "close", "volume"]).to_csv(path)
        else:
            h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
            pd.DataFrame({"open": h["Open"], "close": h["Close"], "volume": h["Volume"]}).to_csv(path)
    d = pd.read_csv(path, index_col=0, parse_dates=True).dropna(subset=["close"])
    if d.empty:
        return None
    sa.check_seal(d, sym)
    return d


def judge(first: pd.Timestamp, last: pd.Timestamp, p_first: pd.Timestamp, p_last: pd.Timestamp, tol: int = 10) -> str:
    """가격 기간 vs 명단 기간 (SW1 judge_period와 같은 기준, 봉인 끝만 2026-09-30, 시작 기준 2021-01)."""
    month_start = first - pd.offsets.MonthBegin(1)
    if p_first > last or p_last < month_start:
        return "다른 회사 의심"
    if p_first > max(month_start, pd.Timestamp("2021-01-04")) + pd.Timedelta(days=tol):
        return "앞이 빔"
    if p_last < min(last, SEAL_LAST) - pd.Timedelta(days=tol + 31):
        return "끝이 이름"
    return "맞음"


def step_prices() -> None:
    """CIK 있는 티커마다 yfinance 가격 → price_check.csv (사용 티커·기간 판정). 실패는 모아 보고."""
    cik = pd.read_csv(CACHE / "cik.csv")
    rows = []
    for _, r in cik.iterrows():
        row = {"ticker": r["ticker"], "cik": r["cik"], "first": r["first"], "last": r["last"]}
        if pd.isna(r["cik"]):
            rows.append({**row, "status": "CIK 없음", "use": False})
            continue
        best = None
        for sym in price_tickers(r):
            d = fetch_yf(sym)
            if d is None:
                continue
            real = d.index[d["volume"].fillna(0) > 0]
            p_first, p_last = d.index.min(), (real[-1] if len(real) else d.index[-1])
            jd = judge(pd.Timestamp(r["first"]), pd.Timestamp(r["last"]) + pd.offsets.MonthEnd(0), p_first, p_last)
            cand = {"price_symbol": sym, "price_first": p_first.date(), "price_last": p_last.date(), "date_check": jd}
            if jd == "맞음":
                best = cand
                break
            best = best or cand
        if best is None:
            rows.append({**row, "status": "yfinance 자료 없음", "use": False})
        else:
            rows.append({**row, **best, "status": "판정", "use": best["date_check"] == "맞음"})
    out = pd.DataFrame(rows)
    out.to_csv(CACHE / "price_check.csv", index=False, encoding="utf-8-sig")
    print(f"[prices] 티커 {len(out)} · 사용 {int(out['use'].sum())}")
    print(out.groupby(["status", out.get("date_check", pd.Series(dtype=str)).fillna("-")]).size().to_string())
    bad = out[~out["use"]]
    if len(bad):
        print(bad.to_string(index=False))


# ── 5) 점검 ─────────────────────────────────────────────────────────────────


def load_qqq() -> pd.Series:
    d = pd.read_csv(QQQ_CSV, index_col=0, parse_dates=True)
    sa.check_seal(d, "QQQ")
    return d["Close"].astype(float)


def classify(days: pd.DatetimeIndex, qqq: pd.Series) -> pd.DataFrame:
    """실적 발표 → 반응일 → 상태(판정 / 판정 불가 사유 / 기간 밖) + 신호 여부. 반응일 뒤 가격은 읽지 않는다."""
    e = earnings_table()
    acc = json.loads(ACCEPTED.read_text(encoding="utf-8")) if ACCEPTED.exists() else {}
    pc = pd.read_csv(CACHE / "price_check.csv").set_index("ticker")
    pxs: dict[str, pd.DataFrame | None] = {}
    res = []
    for _, r in e.iterrows():
        t = r["ticker"]
        a = acc.get(r["accn"])
        d0 = dc.reaction_day(pd.Timestamp(a), days) if a else None
        if t not in pxs:
            ok = t in pc.index and bool(pc.at[t, "use"])
            pxs[t] = fetch_yf(pc.at[t, "price_symbol"]) if ok else None
        rec = {"d0": d0, "signal": False}
        if a is None:
            # 발표 시각이 없으면 제출일 기준으로 기간 안인지만 본다
            rec["status"] = "발표 시각 없음" if r["filing_date"] >= str(D0_FIRST.date()) else "기간 밖"
        elif d0 is None or d0 < D0_FIRST or d0 > SEAL_LAST:
            rec["status"] = "기간 밖"
        elif pxs[t] is None:
            rec["status"] = "가격 없음·다른 회사"
        else:
            s = dc.signal_on(pxs[t], qqq, d0)
            if s is None:
                rec["status"] = "반응일 가격 없음·자료 부족"
            else:
                rec.update(status="판정", signal=s["signal"])
        res.append(rec)
    return pd.concat([e.reset_index(drop=True), pd.DataFrame(res)], axis=1)


def step_check() -> None:
    """판정 불가 비율(⑦)·공백 목록·신호 건수 → check.csv, gaps.csv. 수익 계산 없음."""
    pd.set_option("display.width", 250)
    qqq = load_qqq()
    days = qqq.index
    k = classify(days, qqq)
    k.to_csv(CACHE / "check.csv", index=False, encoding="utf-8-sig")
    cik = pd.read_csv(CACHE / "cik.csv")
    no_cik = cik[cik["cik"].isna()].copy()
    no_cik["est_announcements"] = no_cik["n_months"].map(lambda n: math.ceil(n / 3))
    v = k[k["status"] != "기간 밖"]
    judged = int((v["status"] == "판정").sum())
    und_k = v[v["status"].isin(UNDECIDED_STATUS)]
    und = len(und_k) + int(no_cik["est_announcements"].sum())
    rate = und / (judged + und) if judged + und else float("nan")
    print(f"[check] 대상 발표(반응일 2022-01-01 ~ 2026-09-30) 판정 {judged} · 판정 불가 {und} "
          f"(발표 {len(und_k)} + CIK 없음 추정 {int(no_cik['est_announcements'].sum())}) → 판정 불가 비율 {rate * 100:.2f}%")
    print(v["status"].value_counts().to_string())
    j = v[v["status"] == "판정"].copy()
    j["year"] = pd.DatetimeIndex(j["d0"]).year
    print("[check] 연도별 판정·신호")
    print(j.groupby("year").agg(judged=("signal", "size"), signals=("signal", "sum")).to_string())
    print(f"[check] 신호 합계 {int(j['signal'].sum())} (나스닥100 {int(j.loc[j['in_ndx'], 'signal'].sum())} · "
          f"S&P100 {int(j.loc[j['in_sp100'], 'signal'].sum())})")
    gaps = (und_k.groupby(["ticker", "status"]).size().rename("announcements").reset_index())
    gaps = pd.concat([gaps, no_cik.assign(status="CIK 없음")[["ticker", "status", "est_announcements"]]
                      .rename(columns={"est_announcements": "announcements"})], ignore_index=True)
    gaps.to_csv(CACHE / "gaps.csv", index=False, encoding="utf-8-sig")
    print("[check] 공백 종목")
    print(gaps.sort_values("announcements", ascending=False).to_string(index=False))
    summary = {"judged": judged, "undecided": und, "undecided_announcements": len(und_k),
               "undecided_no_cik_est": int(no_cik["est_announcements"].sum()), "undecided_rate": rate,
               "signals": int(j["signal"].sum())}
    (CACHE / "check_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["members", "events", "accepted", "prices", "check"])
    args = ap.parse_args()
    {"members": step_members, "events": step_events, "accepted": step_accepted, "prices": step_prices,
     "check": step_check}[args.step]()


if __name__ == "__main__":
    main()
