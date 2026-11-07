"""사이클 실험실 2단계 — 데이터 입출력 (docs/prereg/cycle_lab_confirm.md 0절).

모든 데이터는 confirm_gate.gate()를 지난 뒤에만 저장한다. 키 값은 출력하지 않는다.
cache·end 인자는 태그 전 수신 경로 점검(end=2012-12-31, 스크래치 폴더)용이다. 기본값이 2단계 실행 값이다.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import pandas as pd

from data.fx import fetch_fred_series_range
from scripts.cycle_lab import data as dm
from scripts.cycle_lab.confirm_gate import CUTOFF, CUTOFF_STR, CutoffError, assert_cutoff, check_request, gate
from scripts.cycle_lab.confirm_lab import CYCLE_17

CACHE = dm.ROOT / "data" / "cache" / "cycle_lab" / "v2"
V1_FRENCH = dm.ROOT / "data" / "cache" / "cycle_lab" / "french_ind49.csv"
ALFRED_IDS = ["BAA", "GS10"]
YF_TICKERS = ["QQQ", "^NDX"]
YF_START = "2012-12-01"
FX_START = "2012-12-01"
FX_ID = "DEXKOUS"


def _check_end(end: str) -> pd.Timestamp:
    ts = pd.Timestamp(end)
    if ts > CUTOFF:
        raise CutoffError(f"요청 끝 {end}이 2021-12-31 뒤입니다")
    return ts


def _write(df, name: str, source: str, cache: Path) -> dict:
    """gate 지난 표를 캐시에 쓰고 meta를 남긴다. 출력: meta dict."""
    assert_cutoff(df)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{name}.csv"
    df.to_csv(path)
    meta = {"name": name, "source": source, "fetched_utc": dm._now(), "sha256": dm._sha(path),
            "first": str(pd.Timestamp(df.index.min()).date()) if len(df) else "",
            "last": str(pd.Timestamp(df.index.max()).date()) if len(df) else "", "rows": int(len(df)), "cutoff": CUTOFF_STR}
    (cache / f"{name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def drop_rows_after(text: str, last_yyyymm: int) -> str:
    """French CSV 문자열에서 YYYYMM이 last_yyyymm 뒤인 행을 파싱 전에 버린다."""
    keep = []
    for ln in text.splitlines():
        m = re.match(r"^\s*(\d{6})\s*,", ln)
        if m and int(m.group(1)) > last_yyyymm:
            continue
        keep.append(ln)
    return "\n".join(keep)


def fetch_french(cache: Path = CACHE, end: str = CUTOFF_STR) -> dict:
    """49산업 zip을 메모리에서 풀어 end 뒤 행을 파싱 전에 버리고, 사이클 17개 열만 저장한다. 출력: meta."""
    ts = _check_end(end)
    fname = dm.FRENCH_FILES["ind49"]
    raw = dm._get(dm.FRENCH_BASE + fname).content
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        inner = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        text = z.read(inner).decode("latin-1")
    del raw
    df = dm.parse_french_first_table(drop_rows_after(text, ts.year * 100 + ts.month))
    del text
    df = gate(df, "ind49")
    return _write(df.loc[df.index <= ts, CYCLE_17], "french_ind49", dm.FRENCH_BASE + fname + " 첫 표(월간) 사이클 17개 열", cache)


def load_french(cache: Path = CACHE) -> pd.DataFrame:
    """캐시 표 → 소수 수익률 DataFrame, −99.99/−999는 NaN."""
    df = gate(pd.read_csv(cache / "french_ind49.csv", index_col=0, parse_dates=True), "ind49")
    return df.mask(df <= -99.99) / 100.0


def alfred_params(series_id: str, offset: int = 0, end: str = CUTOFF_STR) -> dict:
    """ALFRED 실시간 구간 요청 인자(키 제외). vintage 끝 = end."""
    return check_request({"series_id": series_id, "file_type": "json", "realtime_start": "1776-07-04", "realtime_end": end,
                          "observation_start": "1776-07-04", "observation_end": end, "output_type": 1,
                          "limit": 100000, "offset": offset}, alfred=True)


def alfred_frame(obs: list[dict], series_id: str, end: str = CUTOFF_STR) -> pd.DataFrame:
    """ALFRED 응답 행 → obs 인덱스 표. vintage가 end 뒤면 CutoffError."""
    df = pd.DataFrame(obs)
    df["obs"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
    for c in ("realtime_start", "realtime_end"):
        df[c] = pd.to_datetime(df[c].replace("9999-12-31", end))
    if df["realtime_end"].max() > pd.Timestamp(end) or df["realtime_start"].max() > pd.Timestamp(end):
        raise CutoffError(f"{series_id}: {end} 뒤 vintage가 응답에 있습니다")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return gate(df.set_index("obs")[["realtime_start", "realtime_end", "value"]], series_id)


def fetch_alfred(series_id: str, cache: Path = CACHE, end: str = CUTOFF_STR) -> dict:
    """ALFRED 실시간 구간 행을 받아 저장. 출력: meta."""
    _check_end(end)
    key = dm.api_key()
    obs, off = [], 0
    while True:
        p = alfred_params(series_id, off, end)
        js = dm._get(dm.FRED_OBS_URL, {**p, "api_key": key}).json()
        chunk = js.get("observations", [])
        obs.extend(chunk)
        off += len(chunk)
        if not chunk or off >= int(js.get("count", 0)):
            break
    return _write(alfred_frame(obs, series_id, end), f"alfred_{series_id}", f"ALFRED {series_id} realtime_end={end} output_type=1", cache)


def load_alfred(series_id: str, cache: Path = CACHE) -> pd.DataFrame:
    """캐시 ALFRED 행 → 열 obs, realtime_start, realtime_end, value."""
    df = pd.read_csv(cache / f"alfred_{series_id}.csv", index_col=0, parse_dates=[0, 1, 2])
    df.index = pd.DatetimeIndex(df.index, name="obs")
    df = gate(df, series_id)
    assert_cutoff(pd.DatetimeIndex(df["realtime_end"]), pd.DatetimeIndex(df["realtime_start"]))
    return df.reset_index()


def fetch_current(series_id: str, cache: Path = CACHE) -> pd.Series:
    """3순위: FRED 현재 발표본(관측 ≤ 2021-12-31). 1·2순위에 빈 달이 있을 때만 부른다. 출력: 관측 달 월말 인덱스 Series."""
    p = check_request({"series_id": series_id, "file_type": "json", "observation_start": "1776-07-04",
                       "observation_end": CUTOFF_STR}, alfred=False)
    js = dm._get(dm.FRED_OBS_URL, {**p, "api_key": dm.api_key()}).json()
    df = pd.DataFrame(js.get("observations", []))
    s = pd.Series(pd.to_numeric(df["value"], errors="coerce").to_numpy(),
                  index=pd.DatetimeIndex(pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0), name="obs"), name="value")
    s = gate(s.dropna(), series_id)
    _write(s.to_frame(), f"fred_{series_id}", f"FRED 현재 발표본 {series_id} observation_end={CUTOFF_STR}", cache)
    return s


def fetch_fx(cache: Path = CACHE, end: str = CUTOFF_STR) -> dict:
    """DEXKOUS 일별 값(FX_START ~ end) 저장. 출력: meta."""
    _check_end(end)
    vals = fetch_fred_series_range(FX_ID, FX_START, end, dm.api_key())
    if not vals:
        raise RuntimeError(f"{FX_ID}: 빈 결과")
    s = pd.Series(vals, name="value", dtype=float)
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index), name="date")
    return _write(gate(s.sort_index(), FX_ID).to_frame(), f"fred_{FX_ID}", f"FRED {FX_ID} {FX_START} ~ {end}", cache)


def load_fx(cache: Path = CACHE) -> pd.Series:
    """캐시 DEXKOUS 일별 Series."""
    return gate(pd.read_csv(cache / f"fred_{FX_ID}.csv", index_col=0, parse_dates=True)["value"], FX_ID)


def fetch_yf(ticker: str, cache: Path = CACHE, end: str = CUTOFF_STR) -> dict:
    """yfinance auto_adjust=False·actions=True 일봉(YF_START ~ end). 출력: meta."""
    import yfinance as yf

    ts = _check_end(end)
    stop = str((ts + pd.Timedelta(days=1)).date())
    df = yf.download(ticker, start=YF_START, end=stop, auto_adjust=False, actions=True, progress=False, threads=False)
    if df is None or df.empty:
        raise RuntimeError(f"{ticker}: yfinance 빈 결과")
    out = pd.DataFrame({c: (df[c].iloc[:, 0] if isinstance(df[c], pd.DataFrame) else df[c]) for c in ("Close", "Dividends") if c in df})
    if "Dividends" not in out:
        out["Dividends"] = 0.0
    out.index = pd.DatetimeIndex(out.index).tz_localize(None)
    out = gate(out, ticker)
    return _write(out.loc[out.index <= ts], f"yf_{ticker.replace('^', '')}",
                  f"yfinance {ticker} auto_adjust=False actions=True start={YF_START} end={stop}", cache)


def load_yf(ticker: str, cache: Path = CACHE) -> pd.DataFrame:
    """캐시 일봉(Close, Dividends)."""
    df = pd.read_csv(cache / f"yf_{ticker.replace('^', '')}.csv", index_col=0, parse_dates=True)
    df["Dividends"] = df["Dividends"].fillna(0.0)
    return gate(df.dropna(subset=["Close"]), ticker)


def compare_v1(ind: pd.DataFrame, path: Path = V1_FRENCH) -> dict:
    """French 2012-12 이전 값을 1단계 캐시와 비교(보고용, 판정에 안 씀). 출력: compared, different, max_abs_pct."""
    if not path.exists():
        return {"compared": 0, "different": 0, "max_abs_pct": float("nan")}
    v1 = pd.read_csv(path, index_col=0, parse_dates=True)
    v1 = v1.mask(v1 <= -99.99) / 100.0
    idx = ind.index.intersection(v1.index)
    diff = (ind.loc[idx, CYCLE_17] - v1.loc[idx, CYCLE_17]).abs().stack()
    return {"compared": int(len(diff)), "different": int((diff > 1e-9).sum()), "max_abs_pct": float(diff.max() * 100.0) if len(diff) else float("nan")}


def manifest_rows(cache: Path = CACHE) -> list[dict]:
    """캐시 meta 전부."""
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(cache.glob("*.meta.json"))]
