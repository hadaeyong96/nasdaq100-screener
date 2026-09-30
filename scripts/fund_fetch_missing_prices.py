"""AI 펀드 F2: 무료 대체 출처(Tiingo)로 missing_tickers.csv의 확인된 종목 가격을 받는다
(사용자 지시 2026-10-01, "무료 데이터로만 진행").

**Stooq는 뺐다**: stooq.com이 지금 JS 작업증명(proof-of-work) 봇 차단을 걸어 두고 있어
(2026-10-01 확인) 일반 HTTP 요청으로 CSV를 받을 수 없다. 이 챌린지를 풀어서 우회하는
코드는 만들지 않았다 — 사이트가 의도적으로 막아 둔 비-브라우저 접근을 뚫는 것이라
"무료 데이터 활용"의 범위를 넘어선다고 판단했다.

**기간은 이 파일에 하드코딩**(사용자 지시): FETCH_START~FETCH_END. 봉인 구간(2022-01-01
이후)은 core.seal.enforce_not_sealed로 한 번 더 확인한다(이중 방어).

**수정주가 방식**: Tiingo의 `close`는 원본(비조정), `adjClose`는 분할+배당을 모두 반영한다.
이 프로젝트 관례(CLAUDE.md — 분할만 반영, 배당 미반영, yfinance auto_adjust=False Close와
동등)와 다르므로 `splitFactor`만 써서 직접 분할-전용 조정을 한다:

    adjusted(t) = raw(t) / Π(splitFactor(d) for 모든 d > t)

AAPL 2020-08-31 4:1 분할 구간으로 대조한 결과 yfinance auto_adjust=False Close와 소수점
단위까지 일치했다(2026-10-01 직접 확인 — 아래 테스트에도 그 값을 그대로 넣어 둠).
MSFT(분할 없음)는 원본 close와 그대로 일치해 대조 기준으로 썼다.

**저장 위치**: data/cache/alt_prices/tiingo/{ticker}.csv — 기존 yfinance 캐시(data/cache/,
data/*.db)와 분리된 폴더. "source" 열에 "tiingo"를 남긴다. 가격 CSV 자체는 커밋 안 한다
(.gitignore의 data/cache/). 이 스크립트와 검증 테스트만 커밋 대상이다.

실행:
    python -u -m scripts.fund_fetch_missing_prices
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

# 봉인 차단(사용자 지시로 이 파일에 고정) — engine/backtest.py의 train_end과 같은 값이지만
# 이 스크립트는 fund_sim 엔진을 안 거치므로 여기서도 독립적으로 고정해 둔다.
FETCH_START = date(2014, 1, 1)
FETCH_END = date(2021, 12, 31)

# 2026-10-01 조사에서 EODHD 상장폐지 목록과 대조해 제외한 5개(오매칭 3개 + 미확인 2개).
# missing_tickers.csv에서 지우지는 않고, 이 스크립트에서만 건너뛴다(사용자 지시).
EXCLUDE_TICKERS = {"ALTR", "CA", "DTV", "SNDK", "SPLS"}

MISSING_CSV_PATH = ROOT / "outputs" / "backtest" / "fund_dataqc" / "missing_tickers.csv"
CACHE_DIR = ROOT / "data" / "cache" / "alt_prices" / "tiingo"
RESULT_PATH = ROOT / "outputs" / "backtest" / "fund_dataqc" / "tiingo_fetch_result.json"

_DELIST_KEYWORDS = [
    "acquired", "merger", "merged", "private", "delisted", "bankrupt", "buyout",
    "taken private", "acquisition", "went private",
]
_RECONSTITUTION_KEYWORDS = [
    "annual index reconstitution", "did not meet", "minimum", "weight requirement",
    "quarterly index reconstitution", "moved its stock listing", "transferred its listing",
]


class TiingoFetchError(RuntimeError):
    """Tiingo API 호출 실패. 메시지에 api_key 값은 절대 넣지 않는다(CLAUDE.md 보안)."""


def fetch_tiingo_daily(ticker: str, start: date, end: date, api_key: str, timeout: float = 30.0) -> list[dict]:
    """Tiingo /tiingo/daily/{ticker}/prices를 호출해 원본 레코드 목록을 받는다.

    입력: ticker, start, end, api_key
    출력: [{"date":..., "open":..., "high":..., "low":..., "close":..., "volume":...,
           "splitFactor":..., "divCash":...}, ...] (날짜 오름차순)
    예외: TiingoFetchError
    """
    import requests

    try:
        resp = requests.get(
            f"https://api.tiingo.com/tiingo/daily/{ticker}/prices",
            params={"startDate": start.isoformat(), "endDate": end.isoformat(), "token": api_key, "format": "json"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise TiingoFetchError(f"{ticker}: Tiingo 요청 실패(네트워크): {exc}") from None
    if resp.status_code == 404:
        raise TiingoFetchError(f"{ticker}: Tiingo에 해당 티커 없음(404)")
    if resp.status_code != 200:
        raise TiingoFetchError(f"{ticker}: Tiingo 요청 실패: HTTP {resp.status_code}")
    try:
        data = resp.json()
    except ValueError:
        raise TiingoFetchError(f"{ticker}: Tiingo 응답이 JSON이 아님") from None
    if not isinstance(data, list):
        raise TiingoFetchError(f"{ticker}: Tiingo 응답 형식이 예상과 다름: {data!r}"[:200])
    return data


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    """Tiingo 원본 레코드를 DataFrame으로 바꾼다 (순수 함수 — 네트워크 없음).

    입력: fetch_tiingo_daily 결과
    출력: 인덱스=날짜(tz 없음, 정규화), 열=raw_open/raw_high/raw_low/raw_close/volume/split_factor
    """
    if not records:
        return pd.DataFrame(columns=["raw_open", "raw_high", "raw_low", "raw_close", "volume", "split_factor"])
    rows = []
    idx = []
    for r in records:
        idx.append(pd.Timestamp(r["date"][:10]))
        rows.append(
            {
                "raw_open": r["open"], "raw_high": r["high"], "raw_low": r["low"], "raw_close": r["close"],
                "volume": r["volume"], "split_factor": r.get("splitFactor", 1.0),
            }
        )
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="date")).sort_index()
    return df


def apply_split_only_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """분할만 반영(배당 미반영)한 open/high/low/close/volume을 만든다 (순수 함수).

    adjusted(t) = raw(t) / Π(split_factor(d) for d > t). AAPL 2020-08-31 4:1 분할로
    yfinance auto_adjust=False Close와 소수점까지 일치 확인(모듈 docstring 참고).

    입력: records_to_frame 결과(오름차순 정렬 필수)
    출력: raw_* 열은 그대로 두고 open/high/low/close/volume/source 열을 추가한 DataFrame
    """
    if df.empty:
        out = df.copy()
        for col in ("open", "high", "low", "close"):
            out[col] = pd.Series(dtype=float)
        out["source"] = pd.Series(dtype=str)
        return out

    # cum_factor_after[t] = t보다 뒤(엄격히 나중)의 모든 split_factor를 곱한 값.
    # 뒤에서부터 누적곱을 앞으로 밀어 계산한다(오늘 이후 값은 1.0으로 시작).
    reversed_factors = df["split_factor"].iloc[::-1]
    cum_from_today = reversed_factors.cumprod().iloc[::-1]
    cum_factor_after = (cum_from_today / df["split_factor"]).astype(float)

    out = df.copy()
    for raw_col, adj_col in (("raw_open", "open"), ("raw_high", "high"), ("raw_low", "low"), ("raw_close", "close")):
        out[adj_col] = out[raw_col] / cum_factor_after
    out["volume"] = out["volume"] * cum_factor_after
    out["source"] = "tiingo"
    return out


def last_real_trading_day(df: pd.DataFrame) -> date | None:
    """실제 거래가 있던(volume > 0) 마지막 날짜를 찾는다 (순수 함수).

    상장폐지 직전 며칠은 거래정지로 거래량 0인 채 마지막 가격이 반복 기록되는 경우가
    있다(YHOO 2017-06-19~06-26 확인) — 그런 "멈춰 있는" 날짜를 실제 마지막 거래일로
    잘못 보지 않기 위한 함수.

    입력: open/high/low/close/volume 있는 DataFrame(날짜 인덱스, 오름차순)
    출력: 마지막으로 volume > 0인 날짜, 전부 0/NaN이면 마지막 인덱스 날짜, 비어 있으면 None
    """
    if df.empty:
        return None
    nonzero = df.index[df["volume"].fillna(0) > 0]
    if len(nonzero):
        return nonzero[-1].date()
    return df.index[-1].date()


@dataclass(frozen=True)
class MissingTickerRow:
    ticker: str
    company_name: str
    removal_reason: str
    needed_price_range: str


def load_target_tickers(csv_path: Path, exclude: set[str]) -> list[MissingTickerRow]:
    """missing_tickers.csv에서 연구 구간과 겹치고 exclude에 없는 종목만 뽑는다.

    입력: csv_path(scripts/fund_dataqc_report.py 결과), exclude(제외 티커 집합)
    출력: MissingTickerRow 목록
    """
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if r["ticker"] in exclude:
            continue
        if r["needed_price_range"] == "겹치는 구간 없음(연구 구간 밖)":
            continue
        out.append(
            MissingTickerRow(
                ticker=r["ticker"], company_name=r["company_name"],
                removal_reason=r["removal_reason"], needed_price_range=r["needed_price_range"],
            )
        )
    return out


def _needed_end_date(needed_price_range: str) -> date | None:
    """"2014-01-01~2015-10-07; 2016-...~2018-..." 형식에서 가장 늦은 종료일을 뽑는다."""
    ends = []
    for part in needed_price_range.split("; "):
        if "~" not in part:
            continue
        _, e = part.split("~")
        try:
            ends.append(date.fromisoformat(e))
        except ValueError:
            continue
    return max(ends) if ends else None


def classify_reason(reason: str) -> str:
    """removal_reason 텍스트로 "delist"(실제 상폐/합병) vs "trading"(지수 재조정,
    회사는 계속 거래) vs "unclear"를 가른다 (순수 함수, scripts/fund_dataqc_report.py의
    분류와 같은 방식)."""
    r = (reason or "").lower()
    if any(k in r for k in _DELIST_KEYWORDS):
        return "delist"
    if any(k in r for k in _RECONSTITUTION_KEYWORDS):
        return "trading"
    return "unclear"


def check_alignment(category: str, last_trading_day: date | None, needed_end: date | None, tolerance_days: int = 10) -> tuple[str, str]:
    """마지막 거래일이 기대와 맞는지 판정한다 (순수 함수).

    - "delist"(실제 상폐): 마지막 거래일이 needed_end ± tolerance_days 안이어야 정상.
      더 일찍 끝났거나 뒤에도 데이터가 있으면(다른 회사 가격 혼입 의심) REVIEW.
    - "trading"(지수 재조정, 회사는 계속 거래): 마지막 거래일이 needed_end보다
      한참 이전이면(데이터가 예상보다 일찍 끊김) REVIEW, 아니면 OK.
    - "unclear": 항상 "확인 필요".

    출력: (판정, 설명) — 판정은 "OK" | "REVIEW" | "확인 필요"
    """
    if last_trading_day is None or needed_end is None:
        return "확인 필요", "데이터 또는 기대 종료일 없음"
    diff = (last_trading_day - needed_end).days
    if category == "delist":
        if abs(diff) <= tolerance_days:
            return "OK", f"실제 상폐 사유, 마지막 거래일 오차 {diff:+d}일 (허용 {tolerance_days}일 이내)"
        return "REVIEW", f"실제 상폐 사유인데 마지막 거래일이 기대와 {diff:+d}일 차이 — 다른 회사 데이터 혼입 의심"
    if category == "trading":
        if diff >= -tolerance_days:
            return "OK", f"지수 재조정 사유(회사는 계속 거래), 마지막 거래일이 기대 구간 끝까지 도달(diff {diff:+d}일)"
        return "REVIEW", f"지수 재조정 사유인데 데이터가 기대보다 {abs(diff)}일 일찍 끊김 — 확인 필요"
    return "확인 필요", "사유 텍스트로 분류 불가"


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    import os

    api_key = (os.environ.get("TIINGO_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit(".env에 TIINGO_API_KEY가 없습니다.")

    import yaml

    from core.seal import SealedDataError, enforce_not_sealed

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    seal_date_str = cfg["backtest"].get("seal_date")
    seal_date = date.fromisoformat(seal_date_str) if seal_date_str else None
    try:
        enforce_not_sealed(FETCH_END, seal_date, unseal=False)
    except SealedDataError as exc:
        raise SystemExit(f"FETCH_END가 봉인 구간을 넘습니다 — 코드를 고치지 마세요: {exc}") from None

    targets = load_target_tickers(MISSING_CSV_PATH, EXCLUDE_TICKERS)
    print(f"대상 {len(targets)}개 종목 (제외: {sorted(EXCLUDE_TICKERS)})", flush=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for row in targets:
        print(f"  {row.ticker} ({row.company_name}) 요청 중...", flush=True)
        try:
            records = fetch_tiingo_daily(row.ticker, FETCH_START, FETCH_END, api_key)
        except TiingoFetchError as exc:
            print(f"    실패: {exc}", flush=True)
            results.append({"ticker": row.ticker, "status": "FETCH_FAILED", "error": str(exc)})
            continue

        frame = records_to_frame(records)
        adjusted = apply_split_only_adjustment(frame)
        adjusted.to_csv(CACHE_DIR / f"{row.ticker}.csv")

        last_day = last_real_trading_day(adjusted)
        first_day = adjusted.index[0].date() if len(adjusted) else None
        needed_end = _needed_end_date(row.needed_price_range)
        category = classify_reason(row.removal_reason)
        verdict, note = check_alignment(category, last_day, needed_end)

        print(f"    {len(adjusted)}행, {first_day}~{last_day} (기대 종료 {needed_end}) -> {verdict}: {note}", flush=True)
        results.append(
            {
                "ticker": row.ticker, "status": "OK", "rows": len(adjusted),
                "first_day": first_day.isoformat() if first_day else None,
                "last_real_trading_day": last_day.isoformat() if last_day else None,
                "needed_end": needed_end.isoformat() if needed_end else None,
                "reason_category": category, "verdict": verdict, "note": note,
            }
        )

    ok = sum(1 for r in results if r["status"] == "OK")
    failed = [r for r in results if r["status"] != "OK"]
    review = [r for r in results if r.get("verdict") == "REVIEW"]
    print(f"\n총 {len(targets)}개 중 성공 {ok}개 / 실패 {len(failed)}개 / 정렬 재검토 필요(REVIEW) {len(review)}개", flush=True)

    RESULT_PATH.write_text(
        __import__("json").dumps(
            {"generated_at": datetime.now().isoformat(timespec="seconds"), "fetch_start": FETCH_START.isoformat(),
             "fetch_end": FETCH_END.isoformat(), "excluded": sorted(EXCLUDE_TICKERS), "results": results},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"결과 저장: {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
