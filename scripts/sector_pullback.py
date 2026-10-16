"""탐색 B 「강한 업종 속 강한 종목, 첫 눌림 매수」 (configs/sector_pullback_preregistration.yaml, 판정 아님).

사용:
  python scripts/sector_pullback.py --step map   0단계: 대상 명단 × SIC → 업종 배정, 업종별 월 평균 종목 수·매핑 실패 보고
                                                 (가격·수익은 보지 않는다)

명단: 시점별 나스닥100 ∪ S&P 100 매달 말 (SW1 scripts/sw1_data_check.py와 같은 원천, 캐시).
      명단을 쓰는 달 = 그 다음 달 (신호일이 속한 달의 전 달 말 명단, 사전 등록 해석 1).
티커→CIK: SW1 cik.csv. 거기 없는 종목만 SW1과 같은 후보(S&P 500 점검 identity·SEC 현재 티커 목록·바뀐 티커·
      이름 검색)에서 고르되, 외국 기업 공시(20-F·40-F·6-K)도 인정하고 모든 후보에 이름 유사도 0.6 이상을 요구한다.
SIC: EDGAR submissions 본문의 sic (현재 값, 사전 등록 해석 2).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import sector_map as sm  # noqa: E402
from scripts import sw1_data_check as sw1  # noqa: E402

MAP_FILE = ROOT / "configs" / "sector_pullback_sic_map.csv"
CIK_FILE = sw1.CACHE / "cik.csv"
CIK_CANDIDATES = ROOT / "configs" / "sector_pullback_cik_candidates.csv"
SIC_OVERRIDES = ROOT / "configs" / "sector_pullback_sic_overrides.csv"
OUT_DIR = ROOT / "docs" / "factory" / "explore"
TICKER_OUT = OUT_DIR / "sector_pullback_tickers.csv"
FIRST_USE, LAST_USE = "2012-01", "2021-12"  # 명단을 쓰는 달 (사전 등록 기간)
MIN_MEMBERS = 3
FILING_FORMS = {"10-K", "10-Q", "20-F", "40-F", "6-K"}


def members_by_use_month() -> dict[pd.Timestamp, set[str]]:
    """명단을 쓰는 달(달 말 날짜) → 전 달 말 나스닥100 ∪ S&P 100 티커."""
    ndx, sp = sw1.ndx_members(), sw1.sp100_members()
    out = {}
    for p in pd.period_range(FIRST_USE, LAST_USE, freq="M"):
        prev = (p - 1).to_timestamp(how="end").normalize()
        if prev not in ndx or prev not in sp:
            raise RuntimeError(f"{prev.date()} 명단이 없습니다 (나스닥100 {prev in ndx}, S&P100 {prev in sp})")
        out[p.to_timestamp(how="end").normalize()] = ndx[prev] | sp[prev]
    return out


def _filings_in_period(js: list[dict], a: pd.Timestamp, b: pd.Timestamp) -> int:
    """명단 기간(앞 150일·뒤 120일 여유, SW1과 같음) 안 정기·외국 기업 공시 수."""
    r = sw1.submissions_rows(js)
    if r.empty:
        return 0
    lo, hi = (a - pd.Timedelta(days=150)).strftime("%Y-%m-%d"), (b + pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    return int((r["form"].isin(FILING_FORMS) & (r["filing_date"] >= lo) & (r["filing_date"] <= hi)).sum())


def resolve_missing_cik(t: str, a: pd.Timestamp, b: pd.Timestamp, ua: str, cmap: dict, ident: pd.DataFrame,
                        rn: dict, names: dict, manual: list[int] = ()) -> tuple[int | None, str, str]:
    """SW1 cik.csv에 CIK가 없는(또는 틀린) 종목 → (cik 또는 None, 출처, 실패 사유·EDGAR 이름).
    manual: configs/sector_pullback_cik_candidates.csv의 사람 후보 — 다른 후보와 같은 검증을 거친다."""
    exp = names.get(t, set())
    if not exp:
        return None, "", "기대 회사 이름 없음 — 같은 회사인지 확인할 수 없음"
    cands: list[tuple[int, str]] = [(int(c), "사람 후보(검증 통과)") for c in manual]
    if t in ident.index and not pd.isna(ident.loc[t, "cik"]):
        cands.append((int(ident.loc[t, "cik"]), "S&P 500 점검 identity.csv"))
    c = cmap.get(t.upper().replace("-", "."), cmap.get(t.upper()))
    if c is not None:
        cands.append((int(c), "SEC 티커 목록(현재)"))
    new = rn.get(t)
    c = cmap.get(new.replace("-", ".")) if new else None
    if c is not None:
        cands.append((int(c), f"SEC 티커 목록(바뀐 티커 {new})"))
    if t not in sw1.NAME_SEARCH_SKIP:
        for nm in sorted(exp):
            cands += [(cik, f"이름 검색({nm})") for cik, _ in sw1.name_search(nm, ua)]
    best, seen, why = None, set(), []
    for cik, src in cands:
        if cik in seen:
            continue
        seen.add(cik)
        try:
            js = sw1.fetch_submissions(cik, ua)
        except Exception as exc:  # noqa: BLE001 — 후보 하나 실패는 사유로 모은다
            why.append(f"CIK {cik} 받기 실패 {type(exc).__name__}")
            continue
        ed = [js[0].get("name", "")] + [f.get("name", "") for f in js[0].get("formerNames", [])]
        sim = max(sw1.name_sim(n, x) for n in exp for x in ed)
        n_in = _filings_in_period(js, a, b)
        if n_in >= 2 and sim >= sw1.NAME_SIM_MIN and (best is None or sim > best[3]):
            best = (cik, src, js[0].get("name", ""), sim)
    if best:
        return best[0], best[1], f"{best[2]} (유사도 {best[3]:.2f})"
    return None, "", "; ".join(why) or "후보 중 명단 기간 공시 2건 이상 + 이름 유사도 0.6 이상 없음"


def step_map() -> None:
    """0단계: 종목별 CIK·SIC·업종 → docs/factory/explore/sector_pullback_tickers.csv, 업종별 월 평균·매핑 실패 출력."""
    from dotenv import load_dotenv

    from data import edgar

    load_dotenv(ROOT / ".env")
    rules = pd.read_csv(MAP_FILE, comment="#")
    sm.validate_rules(rules)
    members = members_by_use_month()
    months_in = pd.Series([t for ts in members.values() for t in ts]).value_counts()
    tickers = sorted(months_in.index)
    cik_tab = pd.read_csv(CIK_FILE, index_col="ticker")
    missing_in_sw1 = [t for t in tickers if t not in cik_tab.index]
    if missing_in_sw1:
        print(f"[map] SW1 cik.csv에 줄 자체가 없는 종목 {len(missing_in_sw1)}: {missing_in_sw1}")

    ua = edgar.get_user_agent()
    cmap = edgar.fetch_ticker_to_cik()
    ident = pd.read_csv(sw1.SP500_IDENTITY, index_col=0) if sw1.SP500_IDENTITY.exists() else pd.DataFrame()
    rn, names = sw1.renames(), sw1.expected_names()
    manual = pd.read_csv(CIK_CANDIDATES, comment="#").set_index("ticker").to_dict("index")
    sic_over = pd.read_csv(SIC_OVERRIDES, comment="#").set_index("ticker").to_dict("index")

    rows = []
    for t in tickers:
        rec = cik_tab.loc[t] if t in cik_tab.index else None
        a = pd.Timestamp(rec["first"]) if rec is not None else min(m for m, ts in members.items() if t in ts)
        b = pd.Timestamp(rec["last"]) if rec is not None else max(m for m, ts in members.items() if t in ts)
        cik, src, note = None, "", ""
        if t in manual and pd.isna(manual[t]["cik"]):
            note = f"SW1 CIK 버림: {manual[t]['note']}"
        elif t in manual:
            cik, src, note = resolve_missing_cik(t, a, b, ua, cmap, ident, rn, names, [int(manual[t]["cik"])])
        elif rec is not None and not pd.isna(rec["cik"]):
            cik, src = int(rec["cik"]), f"SW1 cik.csv ({rec['cik_source']})"
        else:
            cik, src, note = resolve_missing_cik(t, a, b, ua, cmap, ident, rn, names)
        sic, sic_desc, edgar_name = None, "", ""
        if cik is not None:
            try:
                js = sw1.fetch_submissions(cik, ua)
                sic, sic_desc, edgar_name = js[0].get("sic") or None, js[0].get("sicDescription") or "", js[0].get("name", "")
            except Exception as exc:  # noqa: BLE001 — 실패 종목으로 보고
                note = f"submissions 받기 실패 {type(exc).__name__}"
        if not sic and t in sic_over:  # 현재 SIC가 빈 종목만 당시 공시 머리글 SIC (사용자 결정 2)
            sic, sic_desc = str(sic_over[t]["sic"]), f"당시 공시 머리글: {sic_over[t]['source']}"
        sector = sm.sic_to_sector(sic, rules)
        if sector is None and not note:
            note = "SIC 없음" if not sic else f"SIC {sic}가 매핑표에 없음"
        rows.append({"ticker": t, "cik": cik, "cik_source": src, "edgar_name": edgar_name, "sic": sic,
                     "sic_desc": sic_desc, "sector": sector or sm.UNMAPPED, "months_in_universe": int(months_in[t]),
                     "note": note if sector is None or src.startswith(("S&P", "SEC", "이름", "사람")) else ""})
    tab = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tab.to_csv(TICKER_OUT, index=False, encoding="utf-8-sig")

    sector_of = {r["ticker"]: (None if r["sector"] == sm.UNMAPPED else r["sector"]) for r in rows}
    counts = sm.monthly_sector_counts(members, sector_of)
    summ = sm.summarize_counts(counts, MIN_MEMBERS)
    summ["종목 수(기간 전체)"] = tab["sector"].value_counts().reindex(summ.index).fillna(0).astype(int)
    total = counts.sum(axis=1)
    print(f"[map] 대상 달 {len(members)} ({FIRST_USE} ~ {LAST_USE}), 기간 전체 티커 {len(tickers)}, "
          f"달마다 명단 크기 평균 {total.mean():.1f} (최소 {total.min()}, 최대 {total.max()})")
    print(summ.to_string())
    fail = tab[tab["sector"] == sm.UNMAPPED]
    print(f"\n[map] 매핑 실패 {len(fail)}종목 (대상 종목·달 중 {fail['months_in_universe'].sum() / total.sum() * 100:.1f}%)")
    print(fail[["ticker", "months_in_universe", "cik", "sic", "note"]].to_string(index=False))
    extra = tab[tab["cik_source"].str.startswith(("S&P", "SEC", "이름", "사람"))]
    print(f"\n[map] 이번에 새로 찾은 CIK {len(extra)}종목 (SW1 cik.csv에 없던 것)")
    print(extra[["ticker", "cik", "cik_source", "note", "sic", "sector"]].to_string(index=False))
    if sw1.SEARCH_FAILED:
        print(f"[map] 이름 검색 요청 실패 {len(sw1.SEARCH_FAILED)}: {sw1.SEARCH_FAILED}")
    print(f"\n[map] 종목별 표: {TICKER_OUT.relative_to(ROOT)}")


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="탐색 B 업종 속 강한 종목 첫 눌림 (판정 아님)")
    p.add_argument("--step", choices=["map"], required=True)
    args = p.parse_args(argv)
    {"map": step_map}[args.step]()


if __name__ == "__main__":
    main()
