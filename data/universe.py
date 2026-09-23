"""나스닥 100 구성 종목 목록을 만든다.

출처 우선순위 (P2 P1 마무리 3번 — Invesco 출처 제거):
    1) Nasdaq 공식 나스닥 100 구성 종목 API
    2) 위키백과 "List of NASDAQ-100 companies" 문서의 구성 종목 표
    3) data/universe_fallback.csv — 위 두 곳이 모두 실패했을 때만 쓰는 로컬 스냅샷

모델의 기억으로 티커를 추정하지 않는다. 항상 위 세 출처 중 하나에서 실제로 받아온
데이터만 쓰고, 어떤 출처를 썼는지는 반환 DataFrame의 `.attrs["source"]`와 실행 로그에
남긴다. 결과 종목 수가 100 언저리(EXPECTED_COUNT_RANGE)를 벗어나면 경고한다.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent
FALLBACK_CSV = DATA_DIR / "universe_fallback.csv"
NAME_KR_CSV = DATA_DIR / "name_kr.csv"

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

# nasdaq_official의 종목명 뒤에 붙는 증권 종류·주식군 꼬리 (P1.2 3번).
# 예: "Apple Inc. Common Stock" -> "Apple Inc.",
#     "Alphabet Inc. Class A Common Stock" -> "Alphabet Inc.",
#     "PDD Holdings Inc. American Depositary Shares" -> "PDD Holdings Inc."
_NAME_TAIL_RE = re.compile(
    r"[\s,]*(?:"
    r"\([^()]*\)"  # (DE), (Ireland) 같은 꼬리 괄호
    r"|(?:Class|Series)\s+[A-Z]\b"
    # 여러 단어짜리 꼬리는 통째로 먼저 맞춰야 "American"처럼 앞부분이 남지 않는다
    r"|American\s+Depositary\s+Shares?"
    r"|New\s+York\s+Registry\s+Shares?"
    r"|Subordinate\s+Voting\s+Shares?"
    r"|Capital\s+Stock"
    r"|Common\s+Stock"
    r"|Common\s+Shares?"
    r"|Ordinary\s+Shares?"
    r"|Depositary\s+Shares?"
    r"|Registry\s+Shares?"
    r"|Shares?"
    r")\s*$",
    re.IGNORECASE,
)


def _to_yfinance_ticker(ticker: str) -> str:
    """티커의 점(.)을 yfinance 형식(-)으로 바꾼다. 예: BRK.B -> BRK-B"""
    return ticker.strip().upper().replace(".", "-")


def shorten_company_name(name: str) -> str:
    """종목명 뒤의 증권 종류·주식군 꼬리를 떼어 짧게 만든다.

    입력: 원본 종목명 (예: "Alphabet Inc. Class A Common Stock")
    출력: 꼬리를 뗀 이름 (예: "Alphabet Inc."). 전부 떨어져 빈 문자열이 되면
         원본을 그대로 돌려준다.
    """
    short = str(name).strip()
    for _ in range(5):  # "Common Stock Class A"처럼 꼬리가 겹쳐 붙는 경우까지만
        stripped = _NAME_TAIL_RE.sub("", short).strip()
        if stripped == short:
            break
        short = stripped
    return short if short else str(name).strip()


def _fetch_nasdaq_official() -> pd.DataFrame:
    """Nasdaq 공식 API에서 나스닥 100 구성 종목을 받는다.

    종목명 뒤의 " Common Stock", " Class A Common Stock" 같은 꼬리는 떼어
    짧게 표시한다 (P1.2 3번).

    입력: 없음 (네트워크)
    출력: DataFrame(ticker, name)
    """
    headers = {**_BROWSER_HEADERS, "Accept": "application/json"}
    resp = requests.get(NASDAQ_API_URL, headers=headers, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    rows = payload["data"]["data"]["rows"]

    df = pd.DataFrame(rows)
    out = df.rename(columns={"symbol": "ticker", "companyName": "name"})[["ticker", "name"]].copy()
    out["name"] = out["name"].map(shorten_company_name)
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

    출처 우선순위: Nasdaq 공식 API -> 위키백과 -> 로컬 폴백.
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
