"""사이클 실험실 1단계 — 데이터 입출력 (docs/prereg/cycle_lab_discovery.md 0·3절).

모든 데이터는 gate.gate()를 지난 뒤에만 저장한다. French zip은 디스크에 쓰지 않는다.
캐시: data/cache/cycle_lab/ (git 제외). 파일마다 .meta.json에 출처·받은 시각(UTC)·SHA256.
네트워크 오류 메시지에는 URL(키 포함)을 넣지 않는다.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts.cycle_lab.gate import CUTOFF, CUTOFF_STR, assert_cutoff, check_request, gate

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache" / "cycle_lab"
FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = {"ind49": "49_Industry_Portfolios_CSV.zip", "ff3": "F-F_Research_Data_Factors_CSV.zip"}
FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"
ALFRED_IDS = ["A34SNO", "DGORDER", "TOTALSA", "TSIFRGHT", "PERMIT", "WPU102301", "WTISPLC", "BAA", "GS10", "TB3MS", "INDPRO"]
CURRENT_IDS = ALFRED_IDS + ["USREC"]
YF_TICKERS = ["QQQ", "^NDX"]
YF_START = "1999-03-01"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(df, name: str, source: str) -> dict:
    """gate 지난 표를 캐시에 쓰고 meta를 남긴다. 출력: meta dict."""
    assert_cutoff(df)
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.csv"
    df.to_csv(path)
    meta = {"name": name, "source": source, "fetched_utc": _now(), "sha256": _sha(path),
            "first": str(pd.Timestamp(df.index.min()).date()) if len(df) else "", "last": str(pd.Timestamp(df.index.max()).date()) if len(df) else "",
            "rows": int(len(df)), "cutoff": CUTOFF_STR}
    (CACHE / f"{name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


NOT_IN_ALFRED = "does not exist in ALFRED"


class NotInAlfred(RuntimeError):
    """시리즈가 ALFRED에 아예 없음(FRED API 400 + 해당 문구) — 사전 등록 3절 3순위(현재값)로 간다."""


def _get(url: str, params: dict | None = None, tries: int = 4):
    """requests GET + 재시도. 실패 메시지에 URL·키를 넣지 않는다. ALFRED에 없는 시리즈는 NotInAlfred."""
    import requests

    last = ""
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r
            if r.status_code == 400 and NOT_IN_ALFRED in r.text:
                raise NotInAlfred("ALFRED에 없는 시리즈")
            last = f"HTTP {r.status_code}"
        except requests.RequestException as exc:
            last = type(exc).__name__
        time.sleep(3 * (k + 1))
    raise RuntimeError(f"다운로드 실패({last})") from None


# ── French ────────────────────────────────────────────────────────────────────


def parse_french_first_table(text: str) -> pd.DataFrame:
    """French CSV 문자열의 첫 월간 표 → DataFrame(월말 인덱스, 원 단위 %, 결측 표기 그대로).
    첫 번째 열 이름 줄(빈 첫 칸) 다음부터 YYYYMM 행이 끊길 때까지. 날짜 차단은 호출 쪽 gate()가 한다."""
    lines = text.splitlines()
    i = 0
    while i < len(lines) and not re.match(r"^\s*,", lines[i]):
        i += 1
    if i == len(lines):
        raise ValueError("French 표 머리를 찾지 못했습니다")
    cols = [c.strip() for c in lines[i].split(",")[1:]]
    rows, idx = [], []
    for ln in lines[i + 1:]:
        m = re.match(r"^\s*(\d{6})\s*,(.*)$", ln)
        if not m:
            break
        idx.append(pd.Timestamp(year=int(m.group(1)[:4]), month=int(m.group(1)[4:]), day=1) + pd.offsets.MonthEnd(0))
        rows.append([float(x) for x in m.group(2).split(",")])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="month"), columns=cols)


def fetch_french(key: str) -> dict:
    """French zip을 메모리에서 풀어 첫 표를 파싱하고 즉시 2012-12까지 잘라 저장한다. 출력: meta."""
    fname = FRENCH_FILES[key]
    raw = _get(FRENCH_BASE + fname).content
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        inner = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        text = z.read(inner).decode("latin-1")
    del raw
    df = gate(parse_french_first_table(text), key)
    del text
    return _write(df, f"french_{key}", FRENCH_BASE + fname + " 첫 표(월간)")


def load_french(key: str) -> pd.DataFrame:
    """캐시 French 표 → 소수 수익률 DataFrame, −99.99/−999는 NaN."""
    df = pd.read_csv(CACHE / f"french_{key}.csv", index_col=0, parse_dates=True)
    df = gate(df, key)
    df = df.mask(df.isin([-99.99, -999.0]) | (df <= -99.99))
    return df / 100.0


# ── FRED / ALFRED ─────────────────────────────────────────────────────────────


def api_key() -> str:
    """FRED_API_KEY(.env 또는 환경변수). 값은 반환만 하고 출력하지 않는다. 없으면 RuntimeError."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    key = (os.environ.get("FRED_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("FRED_API_KEY가 없습니다")
    return key


def alfred_params(series_id: str, offset: int = 0) -> dict:
    """ALFRED 실시간 구간 요청 인자(키 제외). vintage 끝 = 2012-12-31."""
    return check_request({"series_id": series_id, "file_type": "json", "realtime_start": "1776-07-04",
                          "realtime_end": CUTOFF_STR, "observation_start": "1776-07-04", "observation_end": CUTOFF_STR,
                          "output_type": 1, "limit": 100000, "offset": offset}, alfred=True)


def current_params(series_id: str) -> dict:
    """현재 발표본 요청 인자(키 제외, 3순위·USREC)."""
    return check_request({"series_id": series_id, "file_type": "json", "observation_start": "1776-07-04",
                          "observation_end": CUTOFF_STR}, alfred=False)


def _rt_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.replace("9999-12-31", CUTOFF_STR))


def fetch_alfred(series_id: str) -> dict:
    """ALFRED 실시간 구간 행(vintage ≤ 2012-12-31)을 받아 저장. 출력: meta."""
    key = api_key()
    obs, off = [], 0
    source = f"ALFRED {series_id} realtime_end={CUTOFF_STR} output_type=1"
    while True:
        p = alfred_params(series_id, off)
        try:
            js = _get(FRED_OBS_URL, {**p, "api_key": key}).json()
        except NotInAlfred:
            source += " — ALFRED에 없는 시리즈(빈 표, 모든 달 3순위 현재값)"
            break
        chunk = js.get("observations", [])
        obs.extend(chunk)
        off += len(chunk)
        if not chunk or off >= int(js.get("count", 0)):
            break
    df = pd.DataFrame(obs)
    if df.empty:
        df = pd.DataFrame(columns=["date", "realtime_start", "realtime_end", "value"])
    df["obs"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
    df["realtime_start"] = _rt_date(df["realtime_start"])
    df["realtime_end"] = _rt_date(df["realtime_end"])
    if len(df) and (df["realtime_end"].max() > CUTOFF or df["realtime_start"].max() > CUTOFF):
        raise RuntimeError(f"{series_id}: 2012-12-31 뒤 vintage가 응답에 있습니다")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = gate(df.set_index("obs")[["realtime_start", "realtime_end", "value"]], series_id)
    return _write(df, f"alfred_{series_id}", source)


def load_alfred(series_id: str) -> pd.DataFrame:
    """캐시 ALFRED 행 → 열 obs, realtime_start, realtime_end, value."""
    df = pd.read_csv(CACHE / f"alfred_{series_id}.csv", index_col=0, parse_dates=[0, 1, 2])
    df.index = pd.DatetimeIndex(df.index, name="obs")  # 빈 표(ALFRED에 없는 시리즈)도 날짜 인덱스로
    for c in ("realtime_start", "realtime_end"):
        df[c] = pd.to_datetime(df[c])
    df = gate(df, series_id)
    assert_cutoff(pd.DatetimeIndex(df["realtime_end"]), pd.DatetimeIndex(df["realtime_start"]))
    return df.reset_index().rename(columns={df.index.name or "index": "obs"})


def fetch_current(series_id: str) -> dict:
    """FRED 현재 발표본(관측 ≤ 2012-12-31) 저장. 출력: meta."""
    p = current_params(series_id)
    js = _get(FRED_OBS_URL, {**p, "api_key": api_key()}).json()
    df = pd.DataFrame(js.get("observations", []))
    s = pd.Series(pd.to_numeric(df["value"], errors="coerce").to_numpy(),
                  index=pd.DatetimeIndex(pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0), name="obs"), name="value")
    s = gate(s.dropna(), series_id)
    return _write(s.to_frame(), f"fred_{series_id}", f"FRED 현재 발표본 {series_id} observation_end={CUTOFF_STR}")


def load_current(series_id: str) -> pd.Series:
    """캐시 FRED 현재값(관측 달 월말 인덱스)."""
    s = pd.read_csv(CACHE / f"fred_{series_id}.csv", index_col=0, parse_dates=True)["value"]
    return gate(s, series_id)


# ── QQQ ───────────────────────────────────────────────────────────────────────


def fetch_yf(ticker: str) -> dict:
    """yfinance auto_adjust=False·actions=True 일봉(1999-03-01 ~ 2012-12-31). 출력: meta."""
    import yfinance as yf

    df = yf.download(ticker, start=YF_START, end="2013-01-01", auto_adjust=False, actions=True, progress=False, threads=False)
    if df is None or df.empty:
        raise RuntimeError(f"{ticker}: yfinance 빈 결과")
    out = pd.DataFrame({c: (df[c].iloc[:, 0] if isinstance(df[c], pd.DataFrame) else df[c]) for c in ("Close", "Dividends") if c in df})
    if "Dividends" not in out:
        out["Dividends"] = 0.0
    out.index = pd.DatetimeIndex(out.index).tz_localize(None)
    out = gate(out, ticker)
    return _write(out, f"yf_{ticker.replace('^', '')}", f"yfinance {ticker} auto_adjust=False actions=True start={YF_START} end=2013-01-01")


def load_yf(ticker: str) -> pd.DataFrame:
    """캐시 일봉(Close, Dividends)."""
    df = pd.read_csv(CACHE / f"yf_{ticker.replace('^', '')}.csv", index_col=0, parse_dates=True)
    df["Dividends"] = df["Dividends"].fillna(0.0)
    return gate(df.dropna(subset=["Close"]), ticker)


def qqq_monthly_total(qqq: pd.DataFrame, ndx: pd.DataFrame, max_diff: float = 0.02) -> tuple[pd.Series, int]:
    """QQQ 월간 총수익. 일간 가격 수익률이 ^NDX와 max_diff 넘게 다르면 ^NDX 수익률로 바꾸고, 배당(Div/전일 Close)을 더한다.
    출력: (월말 인덱스 월간 총수익 — 첫 거래일이 있는 달은 빼고 다음 달부터, 바꾼 날 수)."""
    assert_cutoff(qqq, ndx)
    c = qqq["Close"]
    pr = c.pct_change()
    rr = ndx["Close"].reindex(c.index).pct_change()
    bad = ((pr - rr).abs() > max_diff) & rr.notna()
    pr = pr.where(~bad, rr)
    tr = pr + qqq["Dividends"] / c.shift(1)
    tr = tr.iloc[1:]
    m = (1.0 + tr).groupby(tr.index.to_period("M")).prod() - 1.0
    m.index = m.index.to_timestamp(how="end").normalize()
    first_month = c.index[0].to_period("M").to_timestamp(how="end").normalize()
    return m[m.index > first_month], int(bad.sum())


def manifest_rows() -> list[dict]:
    """캐시 meta 전부."""
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(CACHE.glob("*.meta.json"))]
