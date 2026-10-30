"""슈퍼투자자 최고 확신 종목 전략(superinvestor) 검증 구역 ③ 1단계 — 데이터 확인만 (수익 계산 없음, docs/validation_zones.md 2장 4번).

    python scripts/superinvestor_check.py download   # SEC 「Form 13F 데이터셋」 분기 zip 2013Q3~2021Q4(접수 분기 기준) 받기
    python scripts/superinvestor_check.py parse      # zip → SUBMISSION·COVERPAGE·INFOTABLE 필요한 열만 parquet (접수일 ≤ 2021-12-31 확인)
    python scripts/superinvestor_check.py official   # SEC 공식 13(f) 증권 목록(분기 txt, ≤ 2021Q4) → CUSIP·회사명
    python scripts/superinvestor_check.py figi       # OpenFIGI(키 없음, 분당 25회 이하) CUSIP → 티커
    python scripts/superinvestor_check.py check      # 분기별 매니저 조건 통과 수·구역 ③ 연결률 표 — 수익은 보지 않는다

가드:
- 13F 데이터셋은 접수 분기 2013Q3~2021Q4 zip만 받는다(2022Q1 이후 zip은 받지 않음). 읽은 뒤 FILING_DATE > 2021-12-31 줄이 있으면 멈춘다.
- 공식 13(f) 증권 목록도 2021년 4분기 판까지만 받는다.
- 명단·가격·SIC·GICS는 scripts/value_score_zone3_data.Data·mcap_growth_check(구역 ③ 가드, 2011-06-01 ~ 2021-12-31)로만 읽는다.
- www.sec.gov 요청은 data.edgar 간격(초당 10회 이하)보다 넉넉히 1초에 1번. SEC_USER_AGENT 값은 출력하지 않는다.
- OpenFIGI는 키 없이 분당 25회·요청당 10건(공식 무키 한도) — 3초 간격.

출력: data/cache/superinvestor/ (git 제외) + 화면 요약
"""

from __future__ import annotations

import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "data" / "cache" / "superinvestor"
ZIPS = OUT / "zips"
PARSED = OUT / "parsed"
OFFICIAL = OUT / "official"
FIGI = OUT / "figi.csv"
LINK = OUT / "cusip_link.csv"
COVER = OUT / "coverage.csv"
FILED_LAST = "2021-12-31"
DATASET_URL = "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/{y}q{q}_form13f.zip"
QUARTERS = [(y, q) for y in range(2013, 2022) for q in range(1, 5) if (y, q) >= (2013, 3)]
FIGI_URL = "https://api.openfigi.com/v3/mapping"
MIN_LINK, MIN_MANAGERS = 0.80, 100
EXCLUDE_WORDS = ("ETF", "INDEX", "TRUST", "BANK")


def _sec_get(url: str, stream: bool = False):
    import requests

    from data import edgar

    r = requests.get(url, headers={"User-Agent": edgar.get_user_agent()}, timeout=120, stream=stream)
    time.sleep(1.0)
    return r


# ── 1. 13F 데이터셋 ──────────────────────────────────────────────────────────


def step_download() -> None:
    ZIPS.mkdir(parents=True, exist_ok=True)
    failed = []
    for y, q in QUARTERS:
        path = ZIPS / f"{y}q{q}_form13f.zip"
        if path.exists() and zipfile.is_zipfile(path):
            continue
        r = _sec_get(DATASET_URL.format(y=y, q=q), stream=True)
        if r.status_code != 200:
            failed.append(f"{y}q{q}({r.status_code})")
            continue
        tmp = path.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.replace(path)
        print(f"[download] {y}q{q} {path.stat().st_size / 1e6:.1f}MB", flush=True)
    print(f"[download] 실패 {len(failed)} {failed}")


def _tsv(z: zipfile.ZipFile, name: str, cols: list[str]) -> pd.DataFrame:
    with z.open(name) as f:
        return pd.read_csv(f, sep="\t", usecols=cols, dtype=str, quoting=3, keep_default_na=False, na_values=[""],
                           encoding="latin-1", on_bad_lines="warn")


def step_parse() -> None:
    """분기 zip마다 → parsed/subs_{yq}.parquet(접수 1건 1줄: 접수번호·접수일·종류·CIK·보고 기준일·정정 여부·정정 종류·이름)
    + parsed/hold_{yq}.parquet(접수 × CUSIP: 주식(SH)·옵션 아닌 줄의 VALUE 합(천 달러)·주식 수 합·발행사 이름·종류).
    13F-HR·13F-HR/A만(13F-NT 통지는 보유 표가 없다). 접수일이 2021-12-31을 넘는 줄이 있으면 멈춘다."""
    PARSED.mkdir(parents=True, exist_ok=True)
    for y, q in QUARTERS:
        yq = f"{y}q{q}"
        out_s, out_h = PARSED / f"subs_{yq}.parquet", PARSED / f"hold_{yq}.parquet"
        if out_s.exists() and out_h.exists():
            continue
        z = zipfile.ZipFile(ZIPS / f"{yq}_form13f.zip")
        s = _tsv(z, "SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"])
        s["filed"] = pd.to_datetime(s["FILING_DATE"], format="%d-%b-%Y")
        s["period"] = pd.to_datetime(s["PERIODOFREPORT"], format="%d-%b-%Y")
        if (s["filed"] > pd.Timestamp(FILED_LAST)).any():
            raise RuntimeError(f"{yq}: 2022년 이후 접수 줄이 있습니다 — 멈춥니다")
        s = s[s["SUBMISSIONTYPE"].isin(["13F-HR", "13F-HR/A"])]
        cv = _tsv(z, "COVERPAGE.tsv", ["ACCESSION_NUMBER", "ISAMENDMENT", "AMENDMENTTYPE", "FILINGMANAGER_NAME", "REPORTTYPE"])
        s = s.merge(cv, on="ACCESSION_NUMBER", how="left")
        s = s.rename(columns={"ACCESSION_NUMBER": "accn", "SUBMISSIONTYPE": "form", "CIK": "cik", "ISAMENDMENT": "is_amend",
                              "AMENDMENTTYPE": "amend_type", "FILINGMANAGER_NAME": "name", "REPORTTYPE": "report_type"})
        s["cik"] = s["cik"].astype(int)
        s = s[["accn", "filed", "form", "cik", "period", "is_amend", "amend_type", "name", "report_type"]]
        it = _tsv(z, "INFOTABLE.tsv", ["ACCESSION_NUMBER", "NAMEOFISSUER", "TITLEOFCLASS", "CUSIP", "VALUE", "SSHPRNAMT",
                                        "SSHPRNAMTTYPE", "PUTCALL"])
        it = it[it["ACCESSION_NUMBER"].isin(set(s["accn"])) & it["PUTCALL"].isna() & (it["SSHPRNAMTTYPE"] == "SH")]
        it["value"] = pd.to_numeric(it["VALUE"], errors="coerce")
        it["shares"] = pd.to_numeric(it["SSHPRNAMT"], errors="coerce")
        it["cusip"] = it["CUSIP"].str.strip().str.upper()
        h = it.groupby(["ACCESSION_NUMBER", "cusip"], sort=False).agg(
            value=("value", "sum"), shares=("shares", "sum"), issuer=("NAMEOFISSUER", "first"), cls=("TITLEOFCLASS", "first")).reset_index()
        h = h.rename(columns={"ACCESSION_NUMBER": "accn"})
        s.to_parquet(out_s, index=False)
        h.to_parquet(out_h, index=False)
        print(f"[parse] {yq} 접수 {len(s):,} (정정 {int((s['form'] == '13F-HR/A').sum()):,}) · 보유 줄 {len(h):,} · "
              f"접수일 {s['filed'].min().date()}~{s['filed'].max().date()}", flush=True)


def load_subs() -> pd.DataFrame:
    s = pd.concat([pd.read_parquet(PARSED / f"subs_{y}q{q}.parquet") for y, q in QUARTERS], ignore_index=True)
    if (s["filed"] > pd.Timestamp(FILED_LAST)).any():
        raise RuntimeError("접수 캐시에 2022년 이후 줄이 있습니다")
    return s


def load_holdings(accns: set[str] | None = None, files: list[tuple[int, int]] | None = None,
                  cols: list[str] | None = None) -> pd.DataFrame:
    parts = []
    for y, q in files or QUARTERS:
        h = pd.read_parquet(PARSED / f"hold_{y}q{q}.parquet", columns=cols)
        parts.append(h[h["accn"].isin(accns)] if accns is not None else h)
    return pd.concat(parts, ignore_index=True)


def file_of(p: pd.Timestamp) -> tuple[int, int]:
    """보고 기준일 p의 보고가 기준일(p + 46~50일) 전에 접수되는 곳 = p 다음 분기(접수 분기) zip."""
    n = p + pd.Timedelta(days=1)
    return n.year, (n.month - 1) // 3 + 1


def period_holdings(p: pd.Timestamp, prev: pd.Timestamp) -> pd.DataFrame:
    """보고 기준일 p·prev의 보고가 들어 있는 zip 두 개의 보유 줄(접수번호·CUSIP·value·shares)."""
    return load_holdings(files=sorted({file_of(p), file_of(prev)}), cols=["accn", "cusip", "value", "shares"])


# ── 2. CUSIP → 티커 ──────────────────────────────────────────────────────────

OFFICIAL_TXT = [(y, q) for y in (2020, 2021) for q in range(1, 5)]  # 2020Q1 전 판은 PDF뿐(읽을 도구 없음 — 외부 코드 설치 안 함)
OFFICIAL_URL = "https://www.sec.gov/files/investment/13flist{y}q{q}.txt"


def step_official() -> None:
    """공식 13(f) 증권 목록 txt(2020Q1~2021Q4) → official/names.csv(cusip, name, desc, list)."""
    OFFICIAL.mkdir(parents=True, exist_ok=True)
    rows = []
    for y, q in OFFICIAL_TXT:
        path = OFFICIAL / f"13flist{y}q{q}.txt"
        if not path.exists():
            r = _sec_get(OFFICIAL_URL.format(y=y, q=q))
            r.raise_for_status()
            path.write_bytes(r.content)
        for line in path.read_bytes().decode("latin-1").splitlines():
            if len(line) < 40 or not re.fullmatch(r"[0-9A-Z]{9}", line[:9]):
                continue
            rows.append({"cusip": line[:9], "name": line[10:39].strip(), "desc": line[39:67].strip(), "list": f"{y}q{q}"})
    df = pd.DataFrame(rows).drop_duplicates("cusip", keep="last")
    df.to_csv(OFFICIAL / "names.csv", index=False, encoding="utf-8-sig")
    print(f"[official] 판 {len(OFFICIAL_TXT)}개 · CUSIP {len(df):,}")


def periods() -> list[pd.Timestamp]:
    """13F 보고 기준일(분기 말) 2013-06-30 ~ 2021-09-30 — 2013-06-30은 직전 분기 비교용."""
    return list(pd.date_range("2013-06-30", "2021-09-30", freq="QE"))


def base_date(p: pd.Timestamp) -> pd.Timestamp:
    """분기 기준일 = 13F 마감일(분기 말 + 45일)의 다음 NYSE 거래일(마감일이 거래일이어도 그다음 날)."""
    import value_score_zone3_data as vd

    days = vd.nyse_days()
    return days[days > p + pd.Timedelta(days=45)][0]


def price_table() -> pd.DataFrame:
    """보고 기준일 × CUSIP → 13F가 말하는 분기 말 가격 중앙값(VALUE×1000 ÷ 주식 수, 원본 13F-HR만), 보고 매니저 수, 가장 흔한 발행사 이름.
    가격 연결 확인용(수익 계산 아님). OUT/price_table.parquet 캐시."""
    path = OUT / "price_table.parquet"
    if path.exists():
        return pd.read_parquet(path)
    s = load_subs()
    s = s[s["form"] == "13F-HR"]
    out = []
    for p in periods():
        acc = set(s.loc[s["period"] == p, "accn"])
        h = load_holdings(acc)
        h = h[(h["shares"] > 0) & (h["value"] > 0)].copy()
        h["px"] = h["value"] * 1000.0 / h["shares"]
        h["issuer"] = h["issuer"].str.upper().str.strip()
        g = h.groupby("cusip").agg(px=("px", "median"), n=("accn", "nunique"))
        g["issuer"] = h.groupby(["cusip", "issuer"]).size().reset_index().sort_values(0).drop_duplicates("cusip", keep="last").set_index("cusip")["issuer"]
        g["period"] = p
        out.append(g.reset_index())
        print(f"[price_table] {p.date()} CUSIP {len(g):,}", flush=True)
    df = pd.concat(out, ignore_index=True)
    df.to_parquet(path, index=False)
    return df


def price_candidates(data, tol: float = 0.02) -> pd.DataFrame:
    """가격 후보 (ticker, cusip, match, both, ratio): 13F 가격(분기 말, 매니저 5명 이상 보고)이 우리 캐시의 그날 실제 종가와 ±tol 안인 분기 수(match),
    둘 다 있는 분기 수(both). match ≥ 2 그리고 ratio ≥ 0.6만."""
    pt = price_table()
    pt = pt[pt["n"] >= 5]
    match: dict = {}
    has_px: dict = {}
    for p, g in pt.groupby("period"):
        p = pd.Timestamp(p)
        d = period_day(p)
        g = g.sort_values("px")
        pxs, cus = g["px"].to_numpy(), g["cusip"].to_numpy()
        has_px[p] = set()
        for t in data.raw:
            px = data.price_on(t, d)
            if px is None:
                continue
            has_px[p].add(t)
            lo, hi = np.searchsorted(pxs, px * (1 - tol)), np.searchsorted(pxs, px * (1 + tol), side="right")
            for cu in cus[lo:hi]:
                match[(t, cu)] = match.get((t, cu), 0) + 1
    cus_periods = pt.groupby("cusip")["period"].apply(lambda x: [pd.Timestamp(v) for v in x]).to_dict()
    rows = []
    for (t, cu), k in match.items():
        if k < 2:
            continue
        both = sum(1 for p in cus_periods[cu] if t in has_px[p])
        if both and k / both >= 0.6:
            rows.append({"ticker": t, "cusip": cu, "match": k, "both": both, "ratio": k / both})
    return pd.DataFrame(rows)


def step_figi() -> None:
    """OpenFIGI(키 없음) — 가격 후보(price_candidates) CUSIP만 → figi.csv(cusip, ticker, name, exch, sectype).
    한도: 분당 25회·요청당 10건 → 2.6초 간격, 429면 기다렸다 다시. 이미 받은 CUSIP은 건너뛴다."""
    import requests

    import value_score_zone3_data as vd

    cus = sorted(set(price_candidates(vd.Data())["cusip"]))
    done = pd.read_csv(FIGI, dtype=str) if FIGI.exists() else pd.DataFrame(columns=["cusip", "ticker", "name", "exch", "sectype"])
    todo = [c for c in cus if c not in set(done["cusip"])]
    print(f"[figi] 대상 CUSIP {len(cus):,} · 남은 {len(todo):,} (약 {len(todo) / 230:.0f}분)", flush=True)
    rows = done.to_dict("records")
    for k in range(0, len(todo), 10):
        batch = todo[k:k + 10]
        for _ in range(6):
            r = requests.post(FIGI_URL, json=[{"idType": "ID_CUSIP", "idValue": c} for c in batch], timeout=60)
            if r.status_code == 429:
                time.sleep(int(r.headers.get("ratelimit-reset", 30) or 30) + 1)
                continue
            break
        r.raise_for_status()
        for c, res in zip(batch, r.json()):
            d = [x for x in (res.get("data") or []) if x.get("exchCode") == "US"] or (res.get("data") or [])
            x = d[0] if d else {}
            rows.append({"cusip": c, "ticker": x.get("ticker"), "name": x.get("name"), "exch": x.get("exchCode"),
                         "sectype": x.get("securityType")})
        if (k // 10) % 25 == 24:
            pd.DataFrame(rows).to_csv(FIGI, index=False)
            print(f"[figi] {k + 10:,}/{len(todo):,}", flush=True)
        time.sleep(2.6)
    df = pd.DataFrame(rows)
    df.to_csv(FIGI, index=False)
    print(f"[figi] 끝 · 티커 찾음 {df['ticker'].notna().sum():,}/{len(df):,}")


STOP = {"INC", "CORP", "CORPORATION", "CO", "COMPANY", "COS", "LTD", "LIMITED", "PLC", "HLDGS", "HOLDINGS", "HOLDING", "HLDG", "GROUP",
        "GRP", "THE", "NEW", "DEL", "COM", "CL", "A", "B", "C", "NV", "N", "V", "SA", "AG", "LP", "L", "P", "LLC", "DE", "MD", "PA", "OF", "AND",
        "SHS", "ORD", "REIT", "INTL", "INTERNATIONAL"}


def name_tokens(s: str | None) -> list[str]:
    return [w for w in re.findall(r"[A-Z0-9]+", str(s or "").upper().replace("&", " ")) if w not in STOP]


GENERIC = {"ENERGY", "HEALTH", "HEALTHCARE", "RESOURCES", "TECHNOLOGIES", "TECHNOLOGY", "SYSTEMS", "FINANCIAL", "SERVICES", "INDUSTRIES",
           "PROPERTIES", "CAPITAL", "AMERICAN", "AMERICA", "NATIONAL", "FIRST", "GENERAL", "UNITED", "US", "TRUST", "BANCORP", "PARTNERS",
           "INVESTMENT", "REALTY", "PHARMACEUTICALS", "COMMUNICATIONS", "BRANDS", "ENTERPRISES", "SOLUTIONS", "GLOBAL", "GAS", "OIL",
           "POWER", "ELECTRIC", "INSURANCE", "MANAGEMENT", "PRODUCTS", "STORES", "MEDICAL", "DATA", "SOFTWARE", "MOTORS", "FOODS",
           "INDS", "TECH", "SVCS", "FINL", "RES", "PPTYS", "COMMUNICATION", "WORLDWIDE", "MATERIALS", "ENTERTAINMENT", "SCIENCES"}


def name_match(a: str | None, b: str | None) -> bool:
    """회사명 같음(느슨 — 가격 일치가 주 증거, 이름은 확인용): 불용어를 뺀 첫 낱말이 같거나(앞 4글자),
    흔하지 않은 낱말(3글자 이상, GENERIC 아님)을 하나 이상 공유하거나, 이어 붙인 글자 유사도 ≥ 0.7."""
    from difflib import SequenceMatcher

    ta, tb = name_tokens(a), name_tokens(b)
    if not ta or not tb:
        return False
    if ta[0] == tb[0] or (len(ta[0]) >= 4 and len(tb[0]) >= 4 and ta[0][:4] == tb[0][:4]):
        return True
    if {w for w in ta if len(w) >= 3 and w not in GENERIC} & {w for w in tb if len(w) >= 3 and w not in GENERIC}:
        return True
    return SequenceMatcher(None, "".join(ta), "".join(tb)).ratio() >= 0.7


def zone_names(data) -> dict[str, list[str]]:
    """구역 ③ 종목 → 회사명 후보: sec_meta 이름 + final_gap edgar_name + 이미 받아 둔 submissions의 옛 이름(2022년 전에 시작한 것만).
    submissions는 새로 받지 않는다(2022년 이후 공시 목록이 섞이므로)."""
    import value_score_zone3_check as c

    meta = json.loads(c.META.read_text(encoding="utf-8"))
    _, gap, _ = c.zone_tickers()
    sub_dir = ROOT / "data" / "cache" / "edgar" / "submissions"
    out = {}
    for t in data.tickers:
        names = [meta.get(t, {}).get("name")]
        if t in gap.index:
            names.append(gap.at[t, "edgar_name"])
        for cik in meta.get(t, {}).get("ciks", []):
            p = sub_dir / f"CIK{cik:010d}.json"
            if p.exists():
                j = json.loads(p.read_text(encoding="utf-8"))
                names += [x["name"] for x in j.get("formerNames") or [] if str(x.get("from", ""))[:4] < "2022"]
        out[t] = [n for n in dict.fromkeys(names) if isinstance(n, str) and n]
    return out


def period_day(p: pd.Timestamp) -> pd.Timestamp:
    """보고 기준일(분기 말) 당일 또는 그 전 마지막 NYSE 거래일."""
    import value_score_zone3_data as vd

    days = vd.nyse_days()
    return days[days <= p][-1]


def step_link(tol: float = 0.02) -> pd.DataFrame:
    """CUSIP → 구역 ③ 티커 연결(cusip_link.csv). 수익 계산 없음.

    후보: 13F 가격(분기 말, 매니저 5명 이상 보고)이 우리 캐시의 그날 실제 종가와 ±2% 안인 (티커, CUSIP) 짝.
    채택: 둘 다 있는 분기 중 60% 이상 맞고 2분기 이상 맞음 + 신원 확인 하나 — OpenFIGI 티커 = 그 티커(이름 변경 짝 포함) 또는 회사명 같음
    (공식 13(f) 목록 이름, 없으면 13F에 가장 흔한 발행사 이름). 한 CUSIP이 여러 티커에 붙으면 맞은 비율 → FIGI → 이름 순으로 하나.
    """
    import value_score_zone3_check as c
    import value_score_zone3_data as vd

    data = vd.Data()
    pt = price_table()
    figi = pd.read_csv(FIGI, dtype=str).set_index("cusip")["ticker"].to_dict() if FIGI.exists() else {}
    off = pd.read_csv(OFFICIAL / "names.csv", dtype=str).set_index("cusip")["name"].to_dict() if (OFFICIAL / "names.csv").exists() else {}
    _, _, renames = c.zone_tickers()
    alias: dict[str, set] = {}
    for old, new in renames.items():
        alias.setdefault(old, set()).add(new)
        alias.setdefault(new, set()).add(old)
    zn = zone_names(data)
    issuer = pt[pt["n"] >= 5].sort_values("period").groupby("cusip")["issuer"].last().to_dict()
    rows = []
    for r in price_candidates(data, tol).itertuples():
        t, cu, k, both, ratio = r.ticker, r.cusip, r.match, r.both, r.ratio
        ft = figi.get(cu)
        by_figi = isinstance(ft, str) and (ft.replace("/", ".") == t or ft in alias.get(t, set()))
        cname = off.get(cu) or issuer.get(cu)
        by_name = any(name_match(cname, n) for n in zn.get(t, []))
        if not (by_figi or by_name):
            continue
        rows.append({"cusip": cu, "ticker": t, "match": k, "both": both, "ratio": ratio, "by_figi": by_figi, "by_name": by_name,
                     "cusip_name": cname, "official": cu in off, "zone_name": (zn.get(t) or [""])[0]})
    df = pd.DataFrame(rows)
    df = df.sort_values(["cusip", "ratio", "by_figi", "by_name"], ascending=[True, False, False, False]).drop_duplicates("cusip")
    df.to_csv(LINK, index=False, encoding="utf-8-sig")
    print(f"[link] 연결 CUSIP {len(df):,} · 티커 {df['ticker'].nunique()} / 가격 있는 구역 종목 {len(data.raw)} · "
          f"FIGI 확인 {int(df['by_figi'].sum())} · 이름 확인 {int(df['by_name'].sum())} · 공식 목록 이름 {int(df['official'].sum())}")
    return df


# ── 3. 매니저 ────────────────────────────────────────────────────────────────


def effective_accns(subs: pd.DataFrame, p: pd.Timestamp, b: pd.Timestamp) -> dict[int, list[tuple[str, str]]]:
    """기준일 b 전에(접수일 < b) 접수된 보고 기준일 p의 보고 → 매니저(CIK)마다 [(접수번호, 'set'|'add')] 적용 순서.
    원본 13F-HR = 새로 놓기, 정정 RESTATEMENT = 새로 놓기, 정정 NEW HOLDINGS = 더하기. 접수일 순(같은 날은 접수번호 순)."""
    s = subs[(subs["period"] == p) & (subs["filed"] < b)].sort_values(["filed", "accn"])
    out: dict[int, list] = {}
    for r in s.itertuples():
        if r.form == "13F-HR" or str(r.amend_type).upper().startswith("RESTATEMENT"):
            out[r.cik] = [(r.accn, "set")]
        elif str(r.amend_type).upper().startswith("NEW HOLDINGS") and r.cik in out:
            out[r.cik].append((r.accn, "add"))
    return out


def portfolios(subs: pd.DataFrame, hold: pd.DataFrame, p: pd.Timestamp, b: pd.Timestamp) -> pd.DataFrame:
    """매니저 × CUSIP 보유(value 천 달러·shares) — effective_accns 적용."""
    eff = effective_accns(subs, p, b)
    pairs = [(cik, a) for cik, lst in eff.items() for a, _ in lst]
    if not pairs:
        return pd.DataFrame(columns=["cik", "cusip", "value", "shares"])
    m = pd.DataFrame(pairs, columns=["cik", "accn"])
    h = hold.merge(m, on="accn")
    return h.groupby(["cik", "cusip"], as_index=False)[["value", "shares"]].sum()


def manager_names(subs: pd.DataFrame, b: pd.Timestamp) -> dict[int, str]:
    s = subs[subs["filed"] < b].sort_values("filed")
    return s.groupby("cik")["name"].last().to_dict()


def qualified(subs: pd.DataFrame, hold: pd.DataFrame, p: pd.Timestamp, b: pd.Timestamp, prev: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """조건 통과 매니저의 이번·직전 분기 보유(weight 열 포함) + 단계별 수.
    조건: 보유 종목(주식 CUSIP) 10~60개, 13F 총액(주식 VALUE 합 × 1000) 2억~200억 달러, 직전 분기 보고도 기준일 전 접수,
    이름(가장 최근 접수의 매니저 이름)에 ETF·INDEX·TRUST·BANK 없음(대소문자 무시, 글자 포함)."""
    now = portfolios(subs, hold, p, b)
    now = now[now["value"] > 0]
    old = portfolios(subs, hold, prev, b)
    old = old[old["value"] > 0]
    names = manager_names(subs, b)
    g = now.groupby("cik").agg(n=("cusip", "nunique"), tot=("value", "sum"))
    st = {"filers": len(g)}
    g = g[(g["n"] >= 10) & (g["n"] <= 60)]
    st["n_10_60"] = len(g)
    g = g[(g["tot"] * 1000 >= 2e8) & (g["tot"] * 1000 <= 2e10)]
    st["size"] = len(g)
    g = g[g.index.isin(set(old["cik"]))]
    st["prev"] = len(g)
    g = g[[not any(w in str(names.get(c, "")).upper() for w in EXCLUDE_WORDS) for c in g.index]]
    st["managers"] = len(g)
    keep = set(g.index)
    now, old = now[now["cik"].isin(keep)].copy(), old[old["cik"].isin(keep)].copy()
    for df in (now, old):
        df["weight"] = df["value"] / df.groupby("cik")["value"].transform("sum")
    return now, old, st


def step_check() -> None:
    """분기별 표: 13F 보고 매니저 수 → 조건 단계별 수 → 통과 매니저, 구역 ③ 연결률(명단 전체 분모), 판정 가능 여부. 수익 없음."""
    import value_score_zone3_data as vd

    data = vd.Data()
    subs = load_subs()
    link = pd.read_csv(LINK, dtype=str)
    c2t = dict(zip(link["cusip"], link["ticker"]))
    pts = periods()
    rows = []
    for prev, p in zip(pts[:-1], pts[1:]):
        b = base_date(p)
        hold = period_holdings(p, prev)
        now, _, st = qualified(subs, hold, p, b, prev)
        me = max(m for m in data.mes if m < b)
        allm, priced = data.mem_all[me], data.mem[me]
        acc = {a for lst in effective_accns(subs, p, b).values() for a, _ in lst}
        held = {c2t[c] for c in hold.loc[hold["accn"].isin(acc), "cusip"].unique() if c in c2t}
        rows.append({"period": p.date(), "base": b.date(), **st, "members": len(allm), "priced": len(priced),
                     "linked": len(allm & held), "link_rate": len(allm & held) / len(allm),
                     "link_rate_priced": len(priced & held) / len(priced), "qual_cusips": now["cusip"].nunique()})
        print(f"[check] {p.date()} → {b.date()} 매니저 {st['managers']} · 연결 {len(allm & held)}/{len(allm)}", flush=True)
    m = pd.DataFrame(rows)
    m["judgeable"] = (m["link_rate"] >= MIN_LINK) & (m["managers"] >= MIN_MANAGERS)
    m.to_csv(COVER, index=False, encoding="utf-8-sig")
    show = m.copy()
    for k in ("link_rate", "link_rate_priced"):
        show[k] = (show[k] * 100).round(1)
    pd.set_option("display.width", 220)
    print(show.to_string(index=False))
    print(f"[판정 불가 분기] {[str(d) for d in m.loc[~m['judgeable'], 'period']]}")


if __name__ == "__main__":
    for st in (sys.stdout, sys.stderr):
        st.reconfigure(encoding="utf-8")
    step = sys.argv[1] if len(sys.argv) > 1 else "check"
    {"download": step_download, "parse": step_parse, "official": step_official, "figi": step_figi, "link": step_link,
     "check": step_check}[step]()
