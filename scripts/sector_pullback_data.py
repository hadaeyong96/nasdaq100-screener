"""탐색 B 데이터 층 — 가격 받기·가드 읽기·회사 단위로 묶기 (configs/sector_pullback_preregistration.yaml step1_interpretations.data).

사용:
  python scripts/sector_pullback_data.py --step prices    시점별 명단 종목 가격 출처 정하기 + yfinance 받기 → 출처 표 보고
  python scripts/sector_pullback_data.py --step current   "현재 명단"(나스닥100 ∪ S&P 100 현재) 종목의 CIK·SIC·가격 받기

가격 출처 (CIK가 같은 것만):
  (나) SEC 현재 티커 목록에서 그 티커(또는 SW1 renames의 바뀐 티커)가 같은 CIK → yfinance auto_adjust=False Close·Dividends
  (가) S&P 500 점검 final_gap.csv에서 같은 CIK·status 있음 → 그 점검의 Tiingo 캐시(div_cash 포함)
  (다) SW1 price_check.csv use=True·같은 CIK → SW1 Tiingo 캐시(분배금 없음)
가격 파일은 read_guarded로만 읽는다: 2021-12-31 넘는 줄에서 읽기를 멈춘다(2022년 이후 데이터 금지 가드).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import sw1_data_check as sw1  # noqa: E402

CACHE = ROOT / "data" / "cache" / "sector_pullback"
YF_CACHE = CACHE / "yf"
SOURCES = CACHE / "price_sources.csv"
CURRENT = CACHE / "current_list.csv"
TICKERS = ROOT / "docs" / "factory" / "explore" / "sector_pullback_tickers.csv"
LISTING = ROOT / "configs" / "sector_pullback_listing_dates.csv"
EXCLUDED = {"CTRX", "LILA", "LILAK", "CTS"}  # 0단계 뒤 사용자 결정 2
READ_FIRST, LAST = pd.Timestamp("2010-01-01"), pd.Timestamp("2021-12-31")
YF_START, YF_END = "2010-01-01", "2022-01-01"  # yfinance end는 그날 미포함 → 2021-12-31까지


class GuardError(RuntimeError):
    """2022년 이후 데이터에 닿으려 할 때."""


# ── 가드 읽기 ─────────────────────────────────────────────────────────────


def read_guarded(path: Path) -> pd.DataFrame:
    """가격 CSV → DataFrame(index 날짜, close 분할 반영 종가, div 주당 분배금(close와 같은 분할 기준)).

    yfinance 저장분(date, close, div): close·div가 이미 분할 반영. Tiingo 저장분(date, raw_close, split_factor[, div_cash]):
    raw를 2021-12-31까지의 분할 곱으로 맞춘다(sw1.split_adjust와 같은 방식). 날짜가 2021-12-31을 넘는 줄을 만나면
    그 자리에서 읽기를 멈춘다. 날짜가 오름차순이 아니면 GuardError.
    """
    rows, prev = [], None
    with open(path, encoding="utf-8-sig") as f:
        rd = csv.reader(f)
        head = [h.strip().lower() for h in next(rd)]
        tiingo = "raw_close" in head
        i_d = head.index("date") if "date" in head else 0
        i_c = head.index("raw_close" if tiingo else "close")
        i_s = head.index("split_factor") if tiingo else None
        i_v = head.index("div_cash") if "div_cash" in head else (head.index("div") if "div" in head else None)
        for row in rd:
            d = pd.Timestamp(row[i_d][:10])
            if prev is not None and d < prev:
                raise GuardError(f"{path.name}: 날짜가 오름차순이 아님")
            prev = d
            if d > LAST:
                break
            if d < READ_FIRST or row[i_c] in ("", "nan"):
                continue
            sp = float(row[i_s]) if tiingo and row[i_s] not in ("", "0", "0.0", "nan") else 1.0
            dv = float(row[i_v]) if i_v is not None and row[i_v] not in ("", "nan") else 0.0
            rows.append((d, float(row[i_c]), sp, dv))
    df = pd.DataFrame(rows, columns=["date", "px", "split", "dv"]).drop_duplicates("date", keep="last").set_index("date")
    if df.empty:
        return pd.DataFrame(columns=["close", "div"])
    if tiingo:
        cum = df["split"].cumprod()
        k = cum / cum.iloc[-1]
        df["close"], df["div"] = df["px"] * k, df["dv"] * k
    else:
        df["close"], df["div"] = df["px"], df["dv"]
    if df.index.max() > LAST:
        raise GuardError(f"{path.name}: 2021-12-31 뒤 날짜")
    return df[["close", "div"]]


# ── 받기 ─────────────────────────────────────────────────────────────────


def yf_path(t: str) -> Path:
    return YF_CACHE / f"{t}.csv"


def fetch_yf(t: str) -> bool:
    """yfinance auto_adjust=False Close·Dividends (2010-01-01 ~ 2021-12-31) → 캐시. 출력: 데이터가 있으면 True."""
    import yfinance as yf

    p = yf_path(t)
    if p.exists():
        return p.stat().st_size > 30
    YF_CACHE.mkdir(parents=True, exist_ok=True)
    try:
        h = yf.Ticker(t).history(start=YF_START, end=YF_END, auto_adjust=False)
    except Exception:  # noqa: BLE001 — 실패는 출처 표에 남긴다
        h = pd.DataFrame()
    time.sleep(0.2)
    if h is None or h.empty:
        pd.DataFrame(columns=["date", "close", "div"]).to_csv(p, index=False)
        return False
    h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
    h = h[h.index <= LAST]
    pd.DataFrame({"date": h.index.strftime("%Y-%m-%d"), "close": h["Close"].values,
                  "div": h["Dividends"].values}).to_csv(p, index=False)
    return len(h) > 0


def choose_source(t: str, cik: int, cmap: dict, rn: dict, gap: pd.DataFrame, pchk: pd.DataFrame,
                  expect: set[str], ua: str) -> tuple[str, Path | None, str]:
    """티커 하나의 가격 출처 (같은 회사로 확인된 것만). expect: 기대 회사 이름들. 출력: (출처 이름, 파일 경로 또는 None, 메모)."""
    for x, label in ((t, "yfinance"), (rn.get(t), "yfinance(바뀐 티커 {x})")):
        if not x:
            continue
        c = cmap.get(x.upper().replace("-", "."), cmap.get(x.upper()))
        if c is not None and int(c) == cik:
            if fetch_yf(x):
                return label.format(x=x), yf_path(x), ""
    # (나') 같은 CIK가 지금 다른 티커로 거래되면 그 티커 (RIMM→BB 등, 사전 등록 보완 2)
    now = sorted(x for x, cc in cmap.items() if int(cc) == cik)
    for x in now:
        if fetch_yf(x):
            return f"yfinance(같은 CIK 지금 티커 {x})", yf_path(x), ""
    # (라) 지주회사 개편: 지금 같은 티커의 CIK가 다르지만 그 새 CIK의 EDGAR 이름이 기대 이름과 유사도 0.6 이상,
    #     그리고 옛 CIK는 지금 거래되는 티커가 없을 때 (옛 회사가 다른 티커로 살아 있으면 분사·티커 재배정이라 제외, 사전 등록 보완 1)
    c = cmap.get(t.upper().replace("-", "."), cmap.get(t.upper()))
    if c is not None and int(c) != cik and expect and not now:
        try:
            js = sw1.fetch_submissions(int(c), ua)
            ed = [js[0].get("name", "")] + [f.get("name", "") for f in js[0].get("formerNames", [])]
            sim = max(sw1.name_sim(n, x) for n in expect for x in ed)
        except Exception:  # noqa: BLE001
            sim = 0.0
        if sim >= sw1.NAME_SIM_MIN and fetch_yf(t):
            return f"yfinance(지주회사 개편 새 CIK {int(c)})", yf_path(t), f"{ed[0]} 유사도 {sim:.2f}"
    for gt in (t, rn.get(t)):
        if not gt or gt not in gap.index:
            continue
        r = gap.loc[gt]
        # (가) 그 점검이 같은 회사로 확인(status 있음)했고 그 점검의 CIK가 비었거나 같을 때 (사전 등록 보완 1)
        if str(r["status"]) == "있음" and (pd.isna(r["cik"]) or int(r["cik"]) == cik):
            if r["price_source"] == "tiingo":
                p = sw1.SP500_TIINGO / f"{gt}.csv"
                if p.exists():
                    return f"tiingo(S&P 500 점검 {gt})", p, ""
            elif str(r["price_source"]).startswith("yfinance"):
                x = rn.get(gt, gt) if r["price_source"] != "yfinance" else gt
                if fetch_yf(x):
                    return f"yfinance(S&P 500 점검 {x})", yf_path(x), ""
    if t in pchk.index:
        r = pchk.loc[t]
        if bool(r["use"]) and not pd.isna(r["cik"]) and int(r["cik"]) == cik:
            p = sw1.TIINGO_CACHE / f"{t}.csv"
            if p.exists():
                return "tiingo(SW1, 분배금 없음)", p, ""
    return "없음", None, "CIK가 같은 가격 출처 없음"


def step_prices() -> None:
    """시점별 명단 종목(0단계 표, 제외 4종목 뺌)의 가격 출처 → data/cache/sector_pullback/price_sources.csv, 보고."""
    from data import edgar

    tab = pd.read_csv(TICKERS)
    tab = tab[~tab["ticker"].isin(EXCLUDED) & tab["cik"].notna()]
    cmap = edgar.fetch_ticker_to_cik()
    rn = sw1.renames()
    gap = pd.read_csv(sw1.SP500_FINAL_GAP, encoding="utf-8-sig").drop_duplicates("ticker").set_index("ticker")
    pchk = pd.read_csv(sw1.PRICE_CHECK).drop_duplicates("ticker").set_index("ticker")
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    ua = edgar.get_user_agent()
    names = sw1.expected_names()
    rows = []
    for i, r in enumerate(tab.itertuples()):
        expect = set(names.get(r.ticker, set())) | ({r.edgar_name} if isinstance(r.edgar_name, str) and r.edgar_name else set())
        src, path, note = choose_source(r.ticker, int(r.cik), cmap, rn, gap, pchk, expect, ua)
        first = last = None
        n = 0
        if path is not None:
            df = read_guarded(path)
            if len(df):
                first, last, n = df.index.min().date(), df.index.max().date(), len(df)
        rows.append({"ticker": r.ticker, "cik": int(r.cik), "sector": r.sector, "months": r.months_in_universe,
                     "source": src, "path": str(path.relative_to(ROOT)) if path else "", "first": first, "last": last,
                     "n_days": n, "note": note})
        if (i + 1) % 50 == 0:
            print(f"[prices] {i + 1}/{len(tab)}", flush=True)
    fetch_yf("QQQ")
    out = pd.DataFrame(rows)
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_csv(SOURCES, index=False, encoding="utf-8-sig")
    print(out["source"].value_counts().to_string())
    miss = out[out["source"] == "없음"]
    print(f"\n[prices] 가격 없음 {len(miss)}종목 (명단 달 합 {miss['months'].sum()}):")
    print(miss[["ticker", "cik", "sector", "months"]].to_string(index=False))
    q = read_guarded(yf_path("QQQ"))
    print(f"\n[prices] QQQ {q.index.min().date()} ~ {q.index.max().date()} ({len(q)}일)")


def step_current() -> None:
    """현재 명단(나스닥100 data/universe_fallback.csv ∪ S&P 100 위키 현재 판) → CIK·SIC·업종·가격 받기 → current_list.csv."""
    from dotenv import load_dotenv

    from core import sector_map as sm
    from data import edgar

    load_dotenv(ROOT / ".env")
    ua = edgar.get_user_agent()
    ndx = set(pd.read_csv(ROOT / "data" / "universe_fallback.csv", comment="#")["ticker"])
    r = sw1._wiki_get({"action": "query", "prop": "revisions", "titles": "S&P 100", "rvlimit": 1,
                       "rvprop": "ids|timestamp|content", "rvslots": "main", "format": "json", "formatversion": 2})
    rev = r["query"]["pages"][0]["revisions"][0]
    sp = sw1.parse_sp100_wikitext(rev["slots"]["main"]["content"])
    print(f"[current] 나스닥100 {len(ndx)} · S&P 100 {len(sp)} (위키 판 {rev['revid']}, {rev['timestamp']})")
    rules = pd.read_csv(ROOT / "configs" / "sector_pullback_sic_map.csv", comment="#")
    cmap = edgar.fetch_ticker_to_cik()
    rows = []
    for t in sorted(ndx | sp):
        c = cmap.get(t.upper().replace("-", "."), cmap.get(t.upper()))
        sic = sector = None
        if c is not None:
            try:
                js = sw1.fetch_submissions(int(c), ua)
                sic = js[0].get("sic") or None
            except Exception:  # noqa: BLE001 — 실패는 표에 남긴다
                pass
            sector = sm.sic_to_sector(sic, rules)
        ok = fetch_yf(t) if c is not None else False
        rows.append({"ticker": t, "cik": int(c) if c is not None else None, "sic": sic, "sector": sector or sm.UNMAPPED,
                     "in_ndx": t in ndx, "in_sp100": t in sp, "path": str(yf_path(t).relative_to(ROOT)) if ok else ""})
    out = pd.DataFrame(rows)
    out.to_csv(CURRENT, index=False, encoding="utf-8-sig")
    (CACHE / "current_list_wiki_rev.json").write_text(json.dumps({"revid": rev["revid"], "timestamp": rev["timestamp"]}))
    print(f"[current] 종목 {len(out)}, CIK 없음 {out['cik'].isna().sum()}, 매핑 실패 {(out['sector'] == sm.UNMAPPED).sum()}, "
          f"가격 없음 {(out['path'] == '').sum()}")
    print(out[(out["cik"].isna()) | (out["sector"] == sm.UNMAPPED) | (out["path"] == "")].to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="탐색 B 데이터")
    p.add_argument("--step", choices=["prices", "current"], required=True)
    a = p.parse_args(argv)
    {"prices": step_prices, "current": step_current}[a.step]()


if __name__ == "__main__":
    main()
