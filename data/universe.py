"""나스닥 100 구성 종목 목록을 만든다.

출처 우선순위 (P1.1 2번):
    1) Invesco QQQ 보유 종목 CSV — 나스닥 100을 그대로 추종하는 ETF의 실제 보유 내역
    2) Nasdaq 공식 나스닥 100 구성 종목 API
    3) 위키백과 "List of NASDAQ-100 companies" 문서의 구성 종목 표
    4) data/universe_fallback.csv — 위 세 곳이 모두 실패했을 때만 쓰는 로컬 스냅샷

모델의 기억으로 티커를 추정하지 않는다. 항상 위 네 출처 중 하나에서 실제로 받아온
데이터만 쓰고, 어떤 출처를 썼는지는 반환 DataFrame의 `.attrs["source"]`와 실행 로그에
남긴다. 결과 종목 수가 100 언저리(EXPECTED_COUNT_RANGE)를 벗어나면 경고한다.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent
FALLBACK_CSV = DATA_DIR / "universe_fallback.csv"
NAME_KR_CSV = DATA_DIR / "name_kr.csv"

INVESCO_QQQ_HOLDINGS_URL = (
    "https://www.invesco.com/us/financial-products/etfs/holdings/main/holdings/0"
    "?audienceType=Investor&action=download&ticker=QQQ"
)
NASDAQ_API_URL = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
WIKI_COMPONENTS_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# 나스닥 100은 복수 주식군(알파벳 GOOGL/GOOG 등) 때문에 100~102종목이 정상 범위다.
EXPECTED_COUNT_RANGE = (90, 110)


def _to_yfinance_ticker(ticker: str) -> str:
    """티커의 점(.)을 yfinance 형식(-)으로 바꾼다. 예: BRK.B -> BRK-B"""
    return ticker.strip().upper().replace(".", "-")


def _fetch_invesco_qqq() -> pd.DataFrame:
    """Invesco QQQ ETF의 실제 보유 종목 CSV를 받는다 (나스닥 100을 그대로 추종).

    입력: 없음 (네트워크)
    출력: DataFrame(ticker, name)
    """
    resp = requests.get(INVESCO_QQQ_HOLDINGS_URL, headers=_BROWSER_HEADERS, timeout=20)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [str(c).strip().lower() for c in df.columns]
    ticker_col = next((c for c in df.columns if "ticker" in c), None)
    name_col = next((c for c in df.columns if "name" in c and "fund" not in c), None)
    if ticker_col is None or name_col is None:
        raise ValueError("Invesco 응답에서 ticker/name 열을 찾지 못함 (접근이 막혔을 수 있음)")

    out = df[[ticker_col, name_col]].rename(columns={ticker_col: "ticker", name_col: "name"})
    out = out.dropna(subset=["ticker"])
    out = out[out["ticker"].astype(str).str.match(r"^[A-Z.]{1,6}$")]  # 현금 등 비주식 행 제외
    if len(out) < 50:
        raise ValueError("Invesco 홀딩스 표에서 충분한 종목을 찾지 못함")
    return out.reset_index(drop=True)


def _fetch_nasdaq_official() -> pd.DataFrame:
    """Nasdaq 공식 API에서 나스닥 100 구성 종목을 받는다.

    입력: 없음 (네트워크)
    출력: DataFrame(ticker, name)
    """
    headers = {**_BROWSER_HEADERS, "Accept": "application/json"}
    resp = requests.get(NASDAQ_API_URL, headers=headers, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    rows = payload["data"]["data"]["rows"]

    df = pd.DataFrame(rows)
    out = df.rename(columns={"symbol": "ticker", "companyName": "name"})[["ticker", "name"]]
    if len(out) < 50:
        raise ValueError("Nasdaq API 응답에서 충분한 종목을 찾지 못함")
    return out.reset_index(drop=True)


def _fetch_wikipedia() -> pd.DataFrame:
    """위키백과 "List of NASDAQ-100 companies" 문서의 구성 종목 표를 읽는다.

    (구 "Nasdaq-100" 문서에는 더 이상 티커 표가 없고, 별도 문서로 분리돼 있다.)

    입력: 없음 (네트워크)
    출력: DataFrame(ticker, name)
    """
    resp = requests.get(WIKI_COMPONENTS_URL, headers=_BROWSER_HEADERS, timeout=20)
    resp.raise_for_status()

    tables = pd.read_html(io.BytesIO(resp.content))
    for table in tables:
        cols = {str(c).strip().lower() for c in table.columns}
        if {"ticker", "company"} <= cols:
            df = table.rename(columns={c: str(c).strip().lower() for c in table.columns})
            out = df[["ticker", "company"]].rename(columns={"company": "name"})
            out["ticker"] = out["ticker"].astype(str).str.strip()
            out["name"] = out["name"].astype(str).str.strip()
            return out.reset_index(drop=True)
    raise ValueError("위키백과 표에서 'Ticker'/'Company' 열을 찾지 못했습니다.")


def _load_fallback() -> pd.DataFrame:
    """data/universe_fallback.csv를 읽는다 (위 세 출처가 모두 실패했을 때만 쓴다).

    입력: 없음
    출력: DataFrame(ticker, name)
    """
    df = pd.read_csv(FALLBACK_CSV, dtype=str, comment="#")
    df["ticker"] = df["ticker"].str.strip()
    df["name"] = df["name"].str.strip()
    return df


_SOURCES = [
    ("invesco_qqq", _fetch_invesco_qqq),
    ("nasdaq_official", _fetch_nasdaq_official),
    ("wikipedia", _fetch_wikipedia),
    ("fallback_csv", _load_fallback),
]


def _attach_name_kr(df: pd.DataFrame) -> pd.DataFrame:
    """name_kr.csv를 조인한다. 없는 티커는 빈 문자열로 채운다.

    입력: DataFrame(ticker, name) — ticker는 yfinance 형식
    출력: DataFrame(ticker, name, name_kr)
    """
    if NAME_KR_CSV.exists():
        kr = pd.read_csv(NAME_KR_CSV, dtype=str).fillna("")
        kr["ticker"] = kr["ticker"].str.strip()
        kr["name_kr"] = kr["name_kr"].str.strip()
        merged = df.merge(kr[["ticker", "name_kr"]], on="ticker", how="left")
    else:
        merged = df.copy()
        merged["name_kr"] = ""
    merged["name_kr"] = merged["name_kr"].fillna("")
    return merged


def get_universe() -> pd.DataFrame:
    """나스닥 100 구성 종목 목록을 만든다.

    출처 우선순위: Invesco QQQ 보유 종목 -> Nasdaq 공식 API -> 위키백과 -> 로컬 폴백.
    모델의 기억으로 티커를 추정하지 않는다 — 실제 응답을 파싱한 결과만 쓴다.

    입력: 없음
    출력: DataFrame(ticker, name, name_kr).
         `.attrs["source"]`에 실제 사용한 출처 이름을 담는다.
    """
    raw = None
    used_source = None
    for name, fetch_fn in _SOURCES:
        try:
            raw = fetch_fn()
            used_source = name
            break
        except Exception as exc:
            print(f"[universe] {name} 실패: {exc}")

    if raw is None:
        raise RuntimeError("모든 구성 종목 출처(Invesco/Nasdaq/위키백과/로컬 폴백)가 실패했습니다.")

    raw["ticker"] = raw["ticker"].map(_to_yfinance_ticker)
    raw = raw.drop_duplicates(subset="ticker").reset_index(drop=True)
    result = _attach_name_kr(raw)[["ticker", "name", "name_kr"]]
    result.attrs["source"] = used_source

    print(f"[universe] 구성 종목 출처: {used_source} ({len(result)}종목)")
    lo, hi = EXPECTED_COUNT_RANGE
    if not (lo <= len(result) <= hi):
        print(f"[universe] 경고: 종목 수 {len(result)}개가 예상 범위({lo}~{hi})를 벗어났습니다.")

    return result


if __name__ == "__main__":
    import sys

    for _stream in (sys.stdout, sys.stderr):  # 윈도우 콘솔 cp949 UnicodeEncodeError 방지
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8")

    universe = get_universe()
    print(f"나스닥 100 구성 종목 {len(universe)}개 (출처: {universe.attrs.get('source')})")
    print(universe.to_string(index=False))
