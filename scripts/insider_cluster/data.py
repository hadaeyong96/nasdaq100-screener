"""내부자 집단 매수 — 데이터 입출력 (docs/prereg/insider_cluster.md 0·2절, 해석 19).

SEC 분기 파일은 메모리에서 풀어 Form 4·4/A에 필요한 열만 저장한다. 키·User-Agent 값은 출력하지 않는다.
가격 본문을 받는 함수(fetch_qqq, fetch_fx, fetch_tiingo_body)는 태그 push 뒤에만 run.py가 부른다.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import io
import json
import os
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data.edgar import get_user_agent
from data.fx import fetch_fred_series_range
from scripts import smallmid_prices_check as spc
from scripts.insider_cluster.gate import CUTOFF, CUTOFF_STR, assert_cutoff, gate, gate_filings

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache" / "insider_cluster"
SEC_DIR = CACHE / "sec"
NEW_PRICES = CACHE / "prices"
META_JSON = CACHE / "tiingo_meta.json"
SM = ROOT / "data" / "cache" / "smallmid"
SW1_CIK = ROOT / "data" / "cache" / "sw1" / "cik.csv"
SEC_URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip"
QUARTERS = [f"{y}q{k}" for y in range(2015, 2022) for k in range(1, 5) if f"{y}q{k}" >= "2015q4"]
PX_START = "2016-06-01"
LIQ_WINDOW = 20
SUFFIX = re.compile(r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|llc|lp|plc|holdings|holding|group|the|nv|sa|trust)\b\.?")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_meta(path: Path, source: str, first: str = "", last: str = "", rows: int = 0) -> dict:
    """캐시 파일의 출처·받은 시각·SHA256을 <이름>.meta.json으로 남긴다."""
    meta = {"name": path.name, "source": source, "fetched_utc": now_utc(), "sha256": sha(path), "first": first, "last": last, "rows": rows, "cutoff": CUTOFF_STR}
    path.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


# ── SEC 공시 ──────────────────────────────────────────────────────────────────


def parse_sec_zip(raw: bytes) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """SEC 분기 zip → (filings, owners, trades). Form 4·4/A만, 접수일 2021-12-31 뒤 행은 저장 전에 버린다. 날짜 형식 31-DEC-2015."""
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        def rd(name: str, cols: list[str]) -> pd.DataFrame:
            with z.open(name) as f:
                return pd.read_csv(f, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False, usecols=cols)

        s = rd("SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "PERIOD_OF_REPORT", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"])
        o = rd("REPORTINGOWNER.tsv", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"])
        t = rd("NONDERIV_TRANS.tsv", ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_ACQUIRED_DISP_CD", "TRANS_SHARES", "TRANS_PRICEPERSHARE"])
    s = s[s["DOCUMENT_TYPE"].isin(["4", "4/A"])]
    filings = pd.DataFrame({"accession": s["ACCESSION_NUMBER"], "filing_date": pd.to_datetime(s["FILING_DATE"], format="%d-%b-%Y", errors="coerce"),
                            "doc_type": s["DOCUMENT_TYPE"], "issuer_cik": pd.to_numeric(s["ISSUERCIK"], errors="coerce"), "period": s["PERIOD_OF_REPORT"],
                            "symbol": s["ISSUERTRADINGSYMBOL"]})
    filings = gate_filings(filings.dropna(subset=["issuer_cik"]))
    filings["issuer_cik"] = filings["issuer_cik"].astype(int)
    keep = set(filings["accession"])
    o = o[o["ACCESSION_NUMBER"].isin(keep)]
    owners = pd.DataFrame({"accession": o["ACCESSION_NUMBER"], "owner_cik": pd.to_numeric(o["RPTOWNERCIK"], errors="coerce"), "relationship": o["RPTOWNER_RELATIONSHIP"]}).dropna(subset=["owner_cik"])
    owners["owner_cik"] = owners["owner_cik"].astype(int)
    t = t[t["ACCESSION_NUMBER"].isin(keep) & (t["TRANS_CODE"] == "P")]
    trades = pd.DataFrame({"accession": t["ACCESSION_NUMBER"], "trans_date": pd.to_datetime(t["TRANS_DATE"], format="%d-%b-%Y", errors="coerce"),
                           "code": t["TRANS_CODE"], "ad": t["TRANS_ACQUIRED_DISP_CD"], "shares": pd.to_numeric(t["TRANS_SHARES"], errors="coerce"),
                           "price": pd.to_numeric(t["TRANS_PRICEPERSHARE"], errors="coerce")})
    return filings, owners, trades


def fetch_sec_all(cache: Path = SEC_DIR, quarters: list[str] | None = None, get=None, sleep=time.sleep) -> list[str]:
    """분기 파일을 받아 저장(이미 있으면 건너뜀). 2022 이후 분기는 요청하지 않는다. 출력: 실패 목록."""
    import requests

    get = get or requests.get
    ua = get_user_agent()
    cache.mkdir(parents=True, exist_ok=True)
    failures = []
    for q in quarters or QUARTERS:
        if q > "2021q4":
            raise ValueError(f"{q}: 2022 이후 분기는 요청하지 않습니다")
        path = cache / f"{q}_filings.csv"
        if path.exists() and path.with_suffix(".meta.json").exists():
            continue
        try:
            resp = get(SEC_URL.format(q=q), headers={"User-Agent": ua}, timeout=300)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")
            f, o, t = parse_sec_zip(resp.content)
        except Exception as exc:  # noqa: BLE001 — 실패 목록을 모아 보고
            failures.append(f"{q}: {type(exc).__name__}")
            continue
        o.to_csv(cache / f"{q}_owners.csv", index=False)
        t.to_csv(cache / f"{q}_trades.csv", index=False)
        f.to_csv(path, index=False)
        write_meta(path, SEC_URL.format(q=q), str(f["filing_date"].min().date()), str(f["filing_date"].max().date()), len(f))
        print(f"받음 SEC {q}: Form 4(/A) {len(f)}건, P 거래 {len(t)}행", flush=True)
        sleep(1.0)
    return failures


def load_sec(cache: Path = SEC_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """저장한 분기 파일 합치기. 접수일로 다시 자른다. 출력: (filings, owners, trades)."""
    fs, os_, ts = [], [], []
    for q in QUARTERS:
        p = cache / f"{q}_filings.csv"
        fs.append(pd.read_csv(p, parse_dates=["filing_date"], dtype={"period": str, "symbol": str}, keep_default_na=False))
        os_.append(pd.read_csv(cache / f"{q}_owners.csv", dtype={"relationship": str}, keep_default_na=False))
        ts.append(pd.read_csv(cache / f"{q}_trades.csv", parse_dates=["trans_date"], dtype={"ad": str, "code": str}))
    filings = gate_filings(pd.concat(fs, ignore_index=True))
    return filings, pd.concat(os_, ignore_index=True), pd.concat(ts, ignore_index=True)


# ── 구역 ①·가격 메타 ─────────────────────────────────────────────────────────


class Zone:
    """구역 ① 판정. 입력: intervals(CIK → member_on 구간 목록), big(CIK → (first, last))."""

    def __init__(self, intervals: dict, big: dict) -> None:
        self.intervals, self.big = intervals, big

    def contains(self, cik: int, d: pd.Timestamp) -> bool:
        """그날 편입 중(n_reports = 1 제외)이고 앞뒤 92일 대형 소속이 아니면 True."""
        return spc.member_on(self.intervals.get(cik, []), d)[0] and not spc.big_overlap(self.big.get(cik), d)


def load_zone() -> Zone:
    """명단·대형 소속 파일 → Zone."""
    big = pd.read_csv(SW1_CIK, parse_dates=["first", "last"]).dropna(subset=["cik"])
    return Zone(spc.intervals_by_cik(), {int(c): (a, b) for c, a, b in zip(big["cik"], big["first"], big["last"])})


def load_price_check() -> pd.DataFrame:
    """price_check.csv(메타데이터). cik 인덱스."""
    return pd.read_csv(SM / "price_check.csv", encoding="utf-8-sig").set_index("cik")


def clean_name(name: str) -> str:
    return re.sub(r"\s+", " ", SUFFIX.sub(" ", re.sub(r"[^a-z0-9 ]", " ", str(name).lower()))).strip()


def name_match(a: str, b: str) -> bool:
    """법인 접미사를 뺀 이름 유사도 ≥ 0.5 (difflib). a는 ;로 나뉜 여러 이름일 수 있다."""
    return any(difflib.SequenceMatcher(None, clean_name(x), clean_name(b)).ratio() >= 0.5 for x in str(a).split(";"))


def covered_by_cache(row, confirm: pd.Timestamp, exec_day: pd.Timestamp) -> bool:
    """기존 캐시 메타데이터로 보유 기간 가격 출처가 있는지(해석 2-ㄱ). row: price_check 행."""
    if str(row.get("same_company")) != "True":
        return False
    return pd.Timestamp(row["price_first"]) <= confirm and pd.Timestamp(row["price_last"]) >= exec_day


class TiingoLimit(RuntimeError):
    """Tiingo 월 한도·403 — 메타데이터를 더 확인할 수 없다."""


def tiingo_key() -> str:
    """TIINGO_API_KEY(.env). 값은 출력하지 않는다. 없으면 RuntimeError."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    key = (os.environ.get("TIINGO_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("TIINGO_API_KEY가 없습니다")
    return key


def fred_key() -> str:
    """FRED_API_KEY(.env 또는 환경변수). 값은 출력하지 않는다. 없으면 RuntimeError."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    key = (os.environ.get("FRED_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("FRED_API_KEY가 없습니다")
    return key


def _tiingo_get(url: str, params: dict | None, get, sleep) -> object:
    """Tiingo GET. 429는 다음 정시까지 기다린다. 월 한도·403이면 TiingoLimit. 오류 메시지에 키를 넣지 않는다."""
    import requests

    key = tiingo_key()
    while True:
        try:
            resp = get(url, params=params, headers={"Authorization": f"Token {key}"}, timeout=60)
        except requests.RequestException as exc:
            print(f"Tiingo 네트워크 실패 {type(exc).__name__} — 60초 뒤 다시", flush=True)
            sleep(60)
            continue
        if spc.is_hard_stop(resp.status_code, resp.text if resp.status_code != 200 else ""):
            raise TiingoLimit(f"HTTP {resp.status_code}")
        if resp.status_code == 429 or (resp.status_code == 200 and "run over" in resp.text[:200].lower()):
            if "month" in resp.text.lower():
                raise TiingoLimit("월 한도")
            w = spc.next_hour_wait_seconds(time.time())
            print(f"Tiingo 시간 한도 — {w}초 기다림", flush=True)
            sleep(w)
            continue
        return resp


def fetch_tiingo_meta(tickers: list[str], path: Path = META_JSON, get=None, sleep=time.sleep) -> dict:
    """티커별 메타데이터(시작일·종료일·이름·존재 여부)만 받는다. 가격 본문을 받지 않는다. 출력: {티커: {...}}."""
    import requests

    get = get or requests.get
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for tk in tickers:
        if tk in meta:
            continue
        resp = _tiingo_get(f"https://api.tiingo.com/tiingo/daily/{tk}", None, get, sleep)
        if resp.status_code == 200:
            js = resp.json()
            meta[tk] = {"exists": True, "start": js.get("startDate"), "end": js.get("endDate"), "name": js.get("name")}
        else:
            meta[tk] = {"exists": False, "status": resp.status_code}
        path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        sleep(0.5)
    return meta


def meta_covers(info: dict, names: str, confirm: pd.Timestamp, exec_day: pd.Timestamp) -> bool:
    """Tiingo 메타데이터로 보유 기간 가격 출처가 있는지(해석 2-ㄴ). 시작 ≤ 확인일, 종료 ≥ 체결일, 이름 유사."""
    if not info or not info.get("exists") or not info.get("start") or not info.get("end"):
        return False
    return pd.Timestamp(info["start"]) <= confirm and pd.Timestamp(info["end"]) >= exec_day and name_match(names, info.get("name") or "")


def normalize_symbols(raw: str) -> list[str]:
    """Form 4 ISSUERTRADINGSYMBOL → 티커 후보 목록. 쉼표 등으로 나뉜 값을 풀고, NONE·N/A와 보통주가 아닌 티커는 뺀다."""
    out: list[str] = []
    for tok in re.split(r"[,;/\s]+", str(raw).strip().upper()):
        if tok in ("", "NONE", "NA", "N", "A", "NAN") or not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,7}", tok):
            continue
        for t in spc.candidate_tickers({"ticker_hist": tok, "ticker_now": ""}):
            if t not in out:
                out.append(t)
    return out


def issuer_symbols(filings: pd.DataFrame) -> dict[int, list[str]]:
    """회사별 Form 4 신고 티커(데이터 경로 편차). 출력: {CIK: 티커 목록} — 신고 건수 많은 순, 같으면 이름 순."""
    counts: dict[int, dict[str, int]] = {}
    g = filings.groupby(["issuer_cik", "symbol"]).size()
    for (cik, raw), n in g.items():
        for t in normalize_symbols(raw):
            d = counts.setdefault(int(cik), {})
            d[t] = d.get(t, 0) + int(n)
    return {c: sorted(d, key=lambda t: (-d[t], t)) for c, d in counts.items()}


def candidate_tickers_for(cik: int, tickers: pd.DataFrame, progress: dict, symbols: dict[int, list[str]] | None = None) -> list[str]:
    """그 CIK의 후보 티커 중 지난번에 아직 받지 않은 것. tickers: tickers.csv(cik 인덱스), progress: tiingo_progress,
    symbols: Form 4 신고 티커(issuer_symbols). 기존 후보 다음에 붙이며 중복은 뺀다."""
    base = spc.candidate_tickers(tickers.loc[cik].to_dict()) if cik in tickers.index else []
    extra = (symbols or {}).get(cik, [])
    return [t for t in dict.fromkeys(base + extra) if t not in progress]


# ── 가격 본문 (태그 push 뒤에만) ──────────────────────────────────────────────


def read_cached_price(cik: int, new_dir: Path = NEW_PRICES, old_dir: Path = SM / "prices") -> pd.DataFrame | None:
    """CIK 가격(close, volume). 새 폴더 우선. 거래량 > 0인 마지막 날 뒤는 버린다. 2021-12-31 뒤는 잘린다."""
    p = new_dir / f"{cik}.csv"
    if not p.exists():
        p = old_dir / f"{cik}.csv"
    if not p.exists():
        return None
    px = gate(pd.read_csv(p, index_col=0, parse_dates=True).sort_index(), str(cik))
    px = px[~px.index.duplicated(keep="last")]
    real = px.index[px["volume"].fillna(0) > 0]
    return px.loc[:real[-1], ["close", "volume"]].dropna(subset=["close"]) if len(real) else None


def fetch_tiingo_body(cik: int, row: dict, tickers: list[str], first_periodic: str | None, new_dir: Path = NEW_PRICES, get=None, sleep=time.sleep) -> str | None:
    """후보 티커의 가격 본문을 받아 같은 회사면 저장(해석 19). 출력: 저장한 티커 또는 None."""
    import requests

    get = get or requests.get
    new_dir.mkdir(parents=True, exist_ok=True)
    for tk in tickers:
        resp = _tiingo_get(f"https://api.tiingo.com/tiingo/daily/{tk}/prices", {"startDate": PX_START, "endDate": CUTOFF_STR, "format": "json"}, get, sleep)
        recs = resp.json() if resp.status_code == 200 else None
        sleep(0.5)
        if not (isinstance(recs, list) and recs):
            continue
        raw = pd.DataFrame([{"date": x["date"][:10], "raw_open": x["open"], "raw_close": x["close"], "volume": x["volume"], "split_factor": x.get("splitFactor", 1.0)} for x in recs])
        raw = gate(raw.set_index(pd.to_datetime(raw["date"])).drop(columns="date").sort_index(), tk)
        adj = spc.tiingo_adjust(raw)
        real = adj.index[adj["volume"].fillna(0) > 0]
        if not len(real):
            continue
        first, last = str(row["first"]), str(row["last"])
        window_first = max(pd.Timestamp(first), pd.Timestamp(PX_START))
        dc = spc.judge_period_smallmid(str(window_first.date()), last, adj.index.min(), real[-1], first_periodic)
        gap = spc.max_gap_days(adj, str(window_first.date()), last)
        if dc in ("맞음", "맞음(상장 첫 달)", "끝이 이름") and gap <= 15:
            path = new_dir / f"{cik}.csv"
            adj.assign(ticker=tk, source="tiingo").to_csv(path)
            write_meta(path, f"Tiingo {tk} {PX_START} ~ {CUTOFF_STR}", str(adj.index.min().date()), str(real[-1].date()), len(adj))
            print(f"받음 Tiingo {tk} (CIK {cik})", flush=True)
            return tk
    return None


def fetch_qqq(cache: Path = CACHE) -> dict:
    """QQQ 일봉 auto_adjust=False Close(2016-12-01 ~ 2021-12-31). 출력: meta."""
    import yfinance as yf

    df = yf.download("QQQ", start="2016-12-01", end="2022-01-01", auto_adjust=False, actions=False, progress=False, threads=False)
    if df is None or df.empty:
        raise RuntimeError("QQQ: yfinance 빈 결과")
    c = df["Close"].iloc[:, 0] if isinstance(df["Close"], pd.DataFrame) else df["Close"]
    s = pd.Series(c.to_numpy(dtype=float), index=pd.DatetimeIndex(c.index).tz_localize(None), name="close")
    s = gate(s.dropna())
    s.index.name = "date"
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / "qqq.csv"
    s.to_frame().to_csv(p)
    return write_meta(p, "yfinance QQQ auto_adjust=False 2016-12-01 ~ 2021-12-31", str(s.index.min().date()), str(s.index.max().date()), len(s))


def fetch_fx(cache: Path = CACHE) -> dict:
    """DEXKOUS 일별(2016-12-01 ~ 2021-12-31). 출력: meta."""
    vals = fetch_fred_series_range("DEXKOUS", "2016-12-01", CUTOFF_STR, fred_key())
    if not vals:
        raise RuntimeError("DEXKOUS: 빈 결과")
    s = pd.Series(vals, name="value", dtype=float)
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index), name="date")
    s = gate(s.sort_index())
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / "fx.csv"
    s.to_frame().to_csv(p)
    return write_meta(p, "FRED DEXKOUS 2016-12-01 ~ 2021-12-31", str(s.index.min().date()), str(s.index.max().date()), len(s))


def load_qqq(cache: Path = CACHE) -> pd.Series:
    return gate(pd.read_csv(cache / "qqq.csv", index_col=0, parse_dates=True)["close"], "QQQ")


def load_fx(cache: Path = CACHE) -> pd.Series:
    return gate(pd.read_csv(cache / "fx.csv", index_col=0, parse_dates=True)["value"], "DEXKOUS")


def liquid_flags(px: pd.DataFrame, days: pd.DatetimeIndex) -> np.ndarray:
    """날짜별 유동성 표(길이 len(days)). liquid_before와 같은 판정: 그 종목 거래일 i에서 전일 종가 ≥ 5달러, 전 20거래일 평균 거래대금 ≥ 500만 달러."""
    assert_cutoff(px, days)
    prev_close = px["close"].shift(1)
    dv = (px["close"] * px["volume"]).shift(1).rolling(LIQ_WINDOW, min_periods=LIQ_WINDOW).mean()
    ok = (prev_close >= spc.LIQ_PRICE) & (dv >= spc.LIQ_DOLLAR_VOL)
    return ok.reindex(days, fill_value=False).to_numpy(dtype=bool)


def member_flags(cik: int, zone: Zone, days: pd.DatetimeIndex) -> np.ndarray:
    """날짜별 구역 ① 소속 표(길이 len(days))."""
    v = np.zeros(len(days), bool)
    for a, b, _, n in zone.intervals.get(cik, []):
        if n > 1:
            v |= (days >= a) & (days <= b)
    big = zone.big.get(cik)
    if big:
        v &= ~((days >= big[0] - pd.Timedelta(days=92)) & (days <= big[1] + pd.Timedelta(days=92)))
    return v


def manifest_rows(cache: Path = CACHE) -> list[dict]:
    """캐시 meta 전부(하위 폴더 포함)."""
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(cache.rglob("*.meta.json"))]
