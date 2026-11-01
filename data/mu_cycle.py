"""마이크론 사이클 해부 1단계 — 데이터 입출력 (docs/results/mu_cycle/v1).

발견 구간 1990-01-01 ~ 2012-12-31만 쓴다. 2013년 이후 값은 받지도 읽지도 않는다:
- 내려받을 때 끝 날짜를 2012-12-31로 잘라 요청한다(yfinance end=2013-01-01, fredgraph.csv coed=2012-12-31).
- 캐시를 읽을 때 guard()로 한 번 더 확인한다. 2012-12-31 뒤 값이 있으면 FutureDataError로 멈춘다.
- 봉인 ⓑ(2022~)는 그보다 앞선 차단이라 따로 확인할 필요가 없지만, SEAL_START도 같이 막는다.
EDGAR companyfacts는 한 파일에 전체 연도가 들어 있어, 받은 즉시 메모리에서 2012-12-31까지 공시·기간인 사실만 남기고
그것만 캐시에 쓴다(원본 파일은 저장하지 않는다).

캐시: data/cache/mu_cycle/ (CSV + 같은 이름 .meta.json에 출처·받은 날짜).
"""

from __future__ import annotations

import io
import json
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "mu_cycle"
DATA_START = "1985-01-01"  # 1990년 전년비·12개월 창을 계산할 여유
CUTOFF = pd.Timestamp("2012-12-31")  # 발견 구간 끝(이 날 포함)
SEAL_START = pd.Timestamp("2022-01-01")  # 봉인 ⓑ
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_IDS = ["IPG3344S", "A34SNO", "INDPRO", "T10Y2Y", "DTWEXBGS", "TWEXB", "ICSA", "PCOPPUSDM", "FEDFUNDS"]
TICKERS = ["MU", "^IXIC", "QQQ", "AMAT", "LRCX", "KLAC", "TXN", "MCHP"]
MU_CIK = 723125


class FutureDataError(RuntimeError):
    """발견 구간(2012-12-31) 뒤 또는 봉인 ⓑ 값이 들어왔을 때."""


def guard(obj):
    """Series·DataFrame의 인덱스가 모두 CUTOFF 이하인지 확인한다. 넘으면 FutureDataError. 입력 그대로 반환."""
    if obj is not None and len(obj):
        mx = pd.Timestamp(obj.index.max())
        if mx >= SEAL_START:
            raise FutureDataError("봉인 ⓑ(2022~) 값이 들어왔습니다")
        if mx > CUTOFF:
            raise FutureDataError(f"발견 구간 뒤 값이 들어왔습니다(최대 {mx.date()})")
    return obj


def _meta(path: Path, source: str) -> None:
    path.with_suffix(".meta.json").write_text(json.dumps({"source": source, "fetched": date.today().isoformat(),
                                                          "cutoff": CUTOFF.date().isoformat()}, ensure_ascii=False), encoding="utf-8")


def _yf_path(ticker: str) -> Path:
    return CACHE / f"yf_{ticker.replace('^', '')}.csv"


def fetch_yf(ticker: str) -> None:
    """yfinance auto_adjust=False 일봉(Close·Adj Close·Volume)을 2012-12-31까지 받아 캐시에 쓴다. 실패는 예외."""
    import yfinance as yf

    df = yf.download(ticker, start=DATA_START, end="2013-01-01", auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        raise RuntimeError(f"{ticker}: yfinance 빈 결과")
    out = pd.DataFrame({c: (df[c].iloc[:, 0] if isinstance(df[c], pd.DataFrame) else df[c]) for c in ("Close", "Adj Close", "Volume")})
    out.index = pd.DatetimeIndex(out.index).tz_localize(None)
    out = out[out.index <= CUTOFF]
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_csv(_yf_path(ticker))
    _meta(_yf_path(ticker), f"yfinance {ticker} auto_adjust=False start={DATA_START} end=2013-01-01")


def load_yf(ticker: str) -> pd.DataFrame:
    """캐시의 일봉(Close·Adj Close·Volume). 2012-12-31 뒤 값이 있으면 FutureDataError."""
    df = pd.read_csv(_yf_path(ticker), index_col=0, parse_dates=True)
    return guard(df.dropna(subset=["Close"]))


def fetch_fred(series_id: str) -> bool:
    """키 없는 fredgraph.csv로 1985-01-01 ~ 2012-12-31만 받아 캐시에 쓴다. 출력: 값이 하나라도 있으면 True."""
    import subprocess

    url = f"{FRED_URL}?id={series_id}&cosd={DATA_START}&coed={CUTOFF.date().isoformat()}"
    # requests는 FRED가 연결을 끊는 일이 잦아(2026-11-01 확인) 이전 연구 스크립트처럼 curl을 쓴다
    res = subprocess.run(["curl", "-s", "-f", "-m", "120", url], capture_output=True, text=True)
    if res.returncode != 0 or not res.stdout.startswith("observation_date"):
        raise RuntimeError(f"{series_id}: fredgraph.csv 실패(curl 종료 코드 {res.returncode})")
    df = pd.read_csv(io.StringIO(res.stdout))
    df.columns = ["date", "value"]
    s = pd.to_numeric(df["value"], errors="coerce")
    s.index = pd.to_datetime(df["date"])
    s = s.dropna()
    s = s[s.index <= CUTOFF]
    path = CACHE / f"fred_{series_id}.csv"
    CACHE.mkdir(parents=True, exist_ok=True)
    s.rename("value").to_csv(path, index_label="date")
    _meta(path, url)
    return len(s) > 0


def load_fred(series_id: str) -> pd.Series:
    """캐시의 FRED 관측값(관측일 인덱스). 2012-12-31 뒤 값이 있으면 FutureDataError."""
    path = CACHE / f"fred_{series_id}.csv"
    s = pd.read_csv(path, index_col=0, parse_dates=True)["value"]
    return guard(s.dropna())


def fetch_mu_eps() -> int:
    """EDGAR companyfacts에서 MU의 희석 EPS(us-gaap EarningsPerShareDiluted) 중 기간 끝·공시일이 모두 2012-12-31 이하인
    사실만 캐시에 쓴다(원본 JSON은 저장하지 않음). 출력: 남긴 사실 수. SEC_USER_AGENT가 없으면 EdgarConfigError.
    """
    from data import edgar

    ua = edgar.get_user_agent()
    raw = edgar._get(edgar.COMPANY_FACTS_URL_TMPL.format(cik=MU_CIK), ua).json()
    facts = raw.get("facts", {}).get("us-gaap", {}).get("EarningsPerShareDiluted", {}).get("units", {}).get("USD/shares", [])
    rows = [{"start": f.get("start"), "end": f["end"], "filed": f["filed"], "val": f["val"], "form": f.get("form"), "fp": f.get("fp")}
            for f in facts if f.get("end") and f.get("filed")
            and pd.Timestamp(f["end"]) <= CUTOFF and pd.Timestamp(f["filed"]) <= CUTOFF]
    del raw
    path = CACHE / "edgar_MU_eps_diluted.csv"
    CACHE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    _meta(path, f"{edgar.COMPANY_FACTS_URL_TMPL.format(cik=MU_CIK)} us-gaap EarningsPerShareDiluted (end·filed ≤ {CUTOFF.date()})")
    return len(rows)


def load_mu_eps() -> pd.DataFrame:
    """캐시의 MU 희석 EPS 사실(start·end·filed·val). 기간 끝·공시일이 2012-12-31 뒤면 FutureDataError."""
    path = CACHE / "edgar_MU_eps_diluted.csv"
    if not path.exists():
        return pd.DataFrame(columns=["start", "end", "filed", "val"])
    df = pd.read_csv(path, parse_dates=["start", "end", "filed"])
    for c in ("end", "filed"):
        guard(df.set_index(c))
    return df
