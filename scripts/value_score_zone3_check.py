"""가치평가 전략(value-score) 검증 구역 ③ 0단계 — 허용 조회만 (수익 계산 없음, docs/validation_zones.md 2장 4번).

    python scripts/value_score_zone3_check.py ids      # CIK·SIC·회사명 (SEC submissions, 이름·SIC만 저장)
    python scripts/value_score_zone3_check.py facts    # EDGAR companyfacts → 10-K 연간 값만, 접수일 2021-12-31 이하만 저장
    python scripts/value_score_zone3_check.py ksplit   # yfinance 분할 기록 → 2022년 이후 분할 배수 K (사용자 결정 2026-10-17)
    python scripts/value_score_zone3_check.py check    # 월말 점수 완전 종목 수·항목별 N/A·시작일 (수익 계산 없음)

종목 × 날짜 가드:
- 가격은 scripts/rs26_zone3_check.read_prices_guarded(2021-12-31 넘는 줄에서 읽기 멈춤, 구역 ③ 밖 종목 거부)로만 읽는다.
- EDGAR 사실은 접수일(filed) ≤ 2021-12-31인 10-K·10-K/A만 꺼내 저장한다. 저장 뒤 다시 확인해 넘는 줄이 있으면 멈춘다.
- yfinance 분할 기록은 [사용자 결정 2026-10-17] 예외: 2022년 이후 분할의 날짜·비율만 받아(가격 없음) 곱 K만 저장한다.
  yfinance 캐시 종가 = 그날 실제 종가 ÷ (그날 뒤 ~ 받은 날 분할 곱)이므로, 그날 실제 종가 = 캐시 종가 × (그날 뒤 ~ 2021-12-31 분할 곱) × K.

출력: data/cache/value_score/ (git 제외) + 화면 요약
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import rs26_zone3_check as z  # noqa: E402
from core import value_score as vs  # noqa: E402
from data.market_calendar import trading_days_between  # noqa: E402

OUT = ROOT / "data" / "cache" / "value_score"
META = OUT / "sec_meta.json"
FACTS = OUT / "facts.parquet"
KSPLIT = OUT / "ksplit.csv"
FILED_LAST = "2021-12-31"
EXCLUDED = {"HSH", "WYND", "EP"}  # rs26 결정 3 (가격 점프 점검) 그대로
# 나스닥 100 CMCSA와 같은 회사(컴캐스트 K주) — rs26 이름 변경 짝에 없어 구역 ③에 남아 있던 것 (구현 해석)
ALIAS_EXCLUDED = {"CMCSK"}
# final_gap.csv에 CIK가 없는 종목: renames.csv의 new_cik(같은 회사 확인됨) 또는 수동 표 — ids 단계에서 SEC 이름으로 확인
MANUAL_CIK = {
    "AABA": [1011006], "APA": [6769, 1841666], "BLK": [1364742], "ANRZQ": [1301063], "BCR": [9892], "BMS": [11199], "CCEP": [1491675], "CHK": [895126],
    "DNR": [945764], "FII": [1056288], "FLIR": [354908], "GAS": [1004155], "GGP": [1496048], "HRS": [202058],
    "KORS": [1530721], "LO": [1424847], "LSI": [703360], "LVLT": [794323], "MWV": [1159297], "MYL": [1623613],
    "PBCT": [1378946], "PX": [884905], "SBNY": [1288784], "SYMC": [849399], "TEG": [916863], "TSS": [721683],
    "TWC": [1377013],
}

# SEC submissions에 SIC가 비어 있는 종목 (구현 해석): 알타바(옛 야후, 투자회사 전환 전 7370), 블랙록(금융 6282)
SIC_OVERRIDE = {"AABA": "7370", "BLK": "6282"}

# 꺼낼 태그 (core/value_score.TAGS의 모든 태그)
FORMS = ("10-K", "10-K/A")


def zone_tickers() -> tuple[list[str], pd.DataFrame, dict[str, str]]:
    """구역 ③에 한 번이라도 속하고 가격 파일이 있는 종목 (rs26 0단계 표 + 제외 반영)."""
    tk = pd.read_csv(z.OUT / "zone3_tickers.csv", encoding="utf-8-sig")
    gap = pd.read_csv(z.SP500 / "final_gap.csv", encoding="utf-8-sig").set_index("ticker")
    renames = {z.norm(r["old"]): z.norm(r["final_ticker"])
               for _, r in pd.read_csv(z.SP500 / "renames.csv", encoding="utf-8-sig").iterrows()}
    out = [t for t in tk.loc[tk["price_loaded"], "ticker"] if t not in EXCLUDED and t not in ALIAS_EXCLUDED]
    return out, gap, renames


def ciks_for(t: str, gap: pd.DataFrame) -> list[int]:
    if t in MANUAL_CIK:
        return MANUAL_CIK[t]
    if pd.notna(gap.at[t, "cik"]):
        return [int(gap.at[t, "cik"])]
    rn = pd.read_csv(z.SP500 / "renames.csv", encoding="utf-8-sig")
    r = rn[rn["old"] == t]
    if len(r) and pd.notna(r.iloc[0]["new_cik"]):
        return [int(r.iloc[0]["new_cik"])]
    return []


def step_ids() -> None:
    """종목별 CIK·SEC 회사명·SIC (submissions JSON에서 name·sic·sicDescription만 저장, 공시 목록은 저장하지 않음)."""
    from data import edgar

    tickers, gap, _ = zone_tickers()
    ua = edgar.get_user_agent()
    meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}
    failed = []
    for t in tickers:
        if t in meta:
            continue
        ciks = ciks_for(t, gap)
        if not ciks:
            meta[t] = {"ciks": [], "name": None, "sic": None}
            continue
        try:
            r = edgar._get(f"https://data.sec.gov/submissions/CIK{ciks[0]:010d}.json", ua)
            j = r.json()
            meta[t] = {"ciks": ciks, "name": j.get("name"), "sic": j.get("sic") or None, "sic_desc": j.get("sicDescription")}
        except Exception as exc:  # noqa: BLE001 — 실패 종목은 모아서 보고
            failed.append(f"{t}({str(exc)[:60]})")
    for t, sic in SIC_OVERRIDE.items():
        if t in meta:
            meta[t]["sic"], meta[t]["sic_source"] = sic, "수동"
    OUT.mkdir(parents=True, exist_ok=True)
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    no_cik = [t for t in tickers if not meta.get(t, {}).get("ciks")]
    no_sic = [t for t in tickers if meta.get(t, {}).get("ciks") and not meta[t].get("sic")]
    print(f"[ids] 종목 {len(tickers)} · CIK 없음 {len(no_cik)} {no_cik} · SIC 없음 {len(no_sic)} {no_sic}")
    print(f"[ids] 실패 {len(failed)} {failed}")
    for t in sorted(MANUAL_CIK):
        if t in meta:
            print(f"  수동 CIK {t:6s} {meta[t]['ciks']} → {meta[t]['name']} (SIC {meta[t]['sic']})")


def step_facts() -> None:
    """companyfacts(캐시 우선, 없으면 SEC에서 받음) → 필요한 태그의 10-K·10-K/A 사실 중 접수일 ≤ 2021-12-31만 저장."""
    from data import edgar

    tickers, _, _ = zone_tickers()
    meta = json.loads(META.read_text(encoding="utf-8"))
    rows, failed, dropped_late = [], [], 0
    for t in tickers:
        for cik in meta.get(t, {}).get("ciks", []):
            try:
                cf = edgar.fetch_company_facts(cik)
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{t}/{cik}({str(exc)[:60]})")
                continue
            for tax in ("us-gaap", "dei"):
                for tag in vs.ALL_TAGS:
                    for e in edgar.extract_fact_entries(cf, tax, tag):
                        if e.get("form") not in FORMS or not e.get("filed") or not e.get("end"):
                            continue
                        if e["filed"] > FILED_LAST:
                            dropped_late += 1
                            continue  # 2022년 이후 공시는 저장하지 않는다
                        rows.append({"ticker": t, "cik": cik, "tag": tag, "start": e.get("start"), "end": e["end"],
                                     "val": float(e["val"]), "filed": e["filed"], "form": e["form"], "unit": e["unit"]})
    df = pd.DataFrame(rows)
    if (df["filed"] > FILED_LAST).any():
        raise z.ZoneGuardError("2022년 이후 공시가 섞였습니다")
    df.to_parquet(FACTS, index=False)
    have = df["ticker"].nunique()
    print(f"[facts] 종목 {len(tickers)} 중 10-K 사실 있음 {have} · 사실 {len(df):,}줄 · 2022년 이후 접수라 버린 사실 {dropped_late:,}")
    print(f"[facts] 실패 {len(failed)} {failed}")
    print(f"[facts] 사실 없음: {sorted(set(tickers) - set(df['ticker']))}")


def yf_symbol(t: str, gap: pd.DataFrame, renames: dict[str, str]) -> str | None:
    src = str(gap.at[t, "price_source"])
    if src == "yfinance":
        return t
    if src.startswith("yfinance(") and t in renames:
        return renames[t]
    return None


def read_yf_guarded(t: str, path: Path, allowed: set[str]) -> pd.DataFrame:
    """yfinance 캐시를 줄 단위로 읽는다(2021-12-31 넘는 줄에서 멈춤, 2011-06-01 전 줄은 버림).

    출력: DataFrame(index=날짜, close=캐시 종가(받은 날 기준 분할·분사 반영), div, factor=그날 split 열(없으면 1)).
    rs26 read_prices_guarded는 yfinance split 열을 쓰지 않아(수익 비율만 필요) 여기서 따로 읽는다.
    """
    import csv

    if t not in allowed:
        raise z.ZoneGuardError(f"{t}: 구역 ③ 후보가 아닌 종목의 가격을 열려고 했습니다")
    rows, prev = [], None
    with open(path, encoding="utf-8-sig") as f:
        rd = csv.reader(f)
        head = [h.strip().lower() for h in next(rd)]
        i_c, i_v, i_s = head.index("close"), head.index("div"), head.index("split")
        for row in rd:
            d = pd.Timestamp(row[0][:10])
            if prev is not None and d < prev:
                raise z.ZoneGuardError(f"{t}: 날짜가 오름차순이 아닙니다")
            prev = d
            if d > z.ZONE_LAST:
                break
            if d < z.ZONE_FIRST or row[i_c] == "":
                continue
            sp = float(row[i_s]) if row[i_s] not in ("", "0", "0.0") else 1.0
            rows.append((d, float(row[i_c]), float(row[i_v] or 0.0), sp))
    df = pd.DataFrame(rows, columns=["date", "close", "div", "factor"]).drop_duplicates("date", keep="last")
    return df.set_index("date")


def step_ksplit() -> None:
    """yfinance 종목의 2022년 이후 분할·분사 비율 곱 K (가격은 받지 않는다, 사용자 결정 2026-10-17).
    캐시 파일을 받은 날 뒤 기록은 빼고(캐시 종가에 없음), 2021-12-31 이하 기록은 캐시 split 열과 맞는지 점검만 한다."""
    import os

    import yfinance as yf

    tickers, gap, renames = zone_tickers()
    rows, failed = [], []
    for t in tickers:
        sym = yf_symbol(t, gap, renames)
        if sym is None:
            continue
        path = z.SP500 / "prices" / f"{sym}.csv"
        got = pd.Timestamp(date.fromtimestamp(os.path.getmtime(path)))
        try:
            sp = yf.Ticker(sym).splits
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{t}({str(exc)[:60]})")
            continue
        if sp is None:
            sp = pd.Series(dtype=float)
        sp.index = pd.DatetimeIndex(sp.index).tz_localize(None).normalize()
        post = sp[(sp.index > z.ZONE_LAST) & (sp.index <= got)]
        pre = sp[(sp.index >= z.ZONE_FIRST) & (sp.index <= z.ZONE_LAST)]
        df = read_yf_guarded(t, path, {t})
        fp = df.loc[df["factor"] != 1.0, "factor"]
        a = sorted((d.date(), round(float(v), 4)) for d, v in pre.items())
        b = sorted((d.date(), round(float(v), 4)) for d, v in fp.items() if d >= df.index.min())
        a = [x for x in a if pd.Timestamp(x[0]) >= df.index.min()]
        rows.append({"ticker": t, "yf_symbol": sym, "k": float(np.prod(post.to_numpy(dtype=float))) if len(post) else 1.0,
                     "n_post": len(post), "file_date": got.date(), "pre2021_match": a == b,
                     "api_pre": ";".join(f"{d}:{v}" for d, v in a), "file_pre": ";".join(f"{d}:{v}" for d, v in b)})
    out = pd.DataFrame(rows)
    out.to_csv(KSPLIT, index=False, encoding="utf-8-sig")
    print(f"[ksplit] yfinance 종목 {len(out)} · 2022년 이후 분할·분사 기록 있음 {int((out['k'] != 1).sum())} · "
          f"2021 이하 기록 캐시와 불일치 {int((~out['pre2021_match']).sum())}")
    print(out.loc[~out["pre2021_match"], ["ticker", "api_pre", "file_pre"]].to_string(index=False))
    print(f"[ksplit] 실패 {len(failed)} {failed}")


def start_date(sc: pd.DataFrame, mes: list[pd.Timestamp], days: pd.DatetimeIndex) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    """시작일: 2012-04 이후, 그달 첫 거래일에 쓸 점수(전달 말)의 완전 종목(대상 중 N/A ≤ 2)이 30개 이상인 첫 달의 첫 거래일.
    출력: (시작일, 점수 기준 월말, 그 완전 종목 수)"""
    comp = sc[sc["universe"] & sc["total"].notna()].groupby("date").size()
    for me in mes:
        if me < pd.Timestamp("2012-03-01"):
            continue
        first = days[days > me][0]
        if comp.get(me, 0) >= 30:
            return first, me, int(comp.get(me, 0))
    raise SystemExit("완전 종목 30개 이상인 달이 없습니다")


def step_check() -> None:
    """허용 조회만: 대상 종목 수, 점수 완전 종목 수, 항목별 N/A 비율, 가격 공백, 시작일. 수익·점수 분포는 보지 않는다."""
    import value_score_zone3_data as vd

    data = vd.Data()
    sc = vd.load_scores(data, rebuild=True)
    days = vd.nyse_days()
    rows = []
    for d in data.mes:
        g = sc[sc["date"] == d]
        u = g[g["universe"]]
        comp = u[u["total"].notna()]
        r = {"date": d.date(), "members": len(data.mem_all[d]), "members_with_file": len(data.mem[d]),
             "members_priced": int(g["member"].sum()), "nonfin_priced": int((g["member"] & ~g["financial"]).sum()),
             "ge5": int((g["member"] & ~g["financial"] & (g["price_raw"] >= vd.MIN_PRICE)).sum()),
             "universe": len(u), "complete": len(comp)}
        for k in vs.ITEMS:
            r[f"na_{k}"] = float(u[k].isna().mean() * 100) if len(u) else np.nan
        rows.append(r)
    m = pd.DataFrame(rows)
    m.to_csv(OUT / "monthly_check.csv", index=False, encoding="utf-8-sig")
    st, me, n = start_date(sc, data.mes, days)
    u_all = sc[sc["universe"]]
    comp_all = u_all[u_all["total"].notna()]
    print(f"[명단] 월말 {len(m)}개 · 구역 ③ 종목 수 {m['members'].min()}~{m['members'].max()} (중앙값 {m['members'].median():.0f})")
    gap_pct = 100 - m["members_priced"].sum() / m["members"].sum() * 100
    print(f"[가격] 종목×월말 가격 있음 {m['members_priced'].sum()}/{m['members'].sum()} (공백 {gap_pct:.2f}%) · "
          f"가격 파일 없는 종목×월 {int((m['members'] - m['members_with_file']).sum())}")
    print(f"[대상] 금융 제외 {m['nonfin_priced'].min()}~{m['nonfin_priced'].max()} · 5달러 이상 {m['ge5'].min()}~{m['ge5'].max()} · "
          f"대상(3년 이상·최근 3년 흑자) {m['universe'].min()}~{m['universe'].max()} (평균 {m['universe'].mean():.1f})")
    print(f"[완전] 점수 완전(N/A ≤ 2) {m['complete'].min()}~{m['complete'].max()} (평균 {m['complete'].mean():.1f}) · "
          f"대상 대비 {comp_all.shape[0] / max(len(u_all), 1) * 100:.1f}%")
    na = {k: round(float(u_all[k].isna().mean() * 100), 1) for k in vs.ITEMS}
    print(f"[N/A] 대상 종목×월 항목별 N/A 비율(%) {na}")
    print(f"[시작일] {st.date()} (점수 기준 {me.date()} 완전 {n}개)")
    by_year = m.assign(y=pd.to_datetime(m["date"]).dt.year).groupby("y")[["members", "universe", "complete"]].mean().round(1)
    print("[연도별 월평균]\n" + by_year.to_string())


if __name__ == "__main__":
    for st in (sys.stdout, sys.stderr):
        st.reconfigure(encoding="utf-8")
    step = sys.argv[1] if len(sys.argv) > 1 else "check"
    {"ids": step_ids, "facts": step_facts, "ksplit": step_ksplit, "check": step_check}.get(step, lambda: None)()
