"""AI 펀드 F2: missing_tickers.csv의 종목이 EODHD 상장폐지 목록에 있는지 확인한다
(사용자 지시 2026-09-30 — 코드만 준비, 실행은 사용자가 API 키를 넣은 뒤 직접 한다).

방법(조사 결과):
- EODHD(eodhistoricaldata.com)는 거래소별 "Exchange Symbol List" API에 delisted=1을
  붙이면 그 거래소의 상장폐지 종목만 돌려준다:
      GET https://eodhd.com/api/exchange-symbol-list/{EXCHANGE}?delisted=1&api_token={KEY}&fmt=json
  응답 필드: Code(티커), Name(회사명), Exchange, Country, Currency, Type, Isin.
- 무료 요금제(신용카드 불필요)로 가입하면 일일 20회 호출 한도로 이 API를 쓸 수 있다.
  "All plans"에 나열돼 있어 상장폐지 목록 자체는 무료 요금제에서도 조회 가능한 것으로
  보이지만(2026-09-30 공식 문서·가격 페이지 조사 기준), 요금제별 세부 제한이 100%
  명문화돼 있지는 않았다 — 이 스크립트를 처음 돌렸을 때 402/403이 오면 유료 전환이
  필요하다는 뜻이니 그 결과를 그대로 보고할 것.
- 나스닥100은 US 거래소 소속이 대부분이라 exchange_code="US" 하나로 충분하다(다른
  거래소 상장이었던 종목은 이 스크립트로 못 잡는다 — 결과 보고에 "확인 안 됨"으로 남는다).
- 티커 표기 차이 주의: 이 프로젝트는 data/universe_history.py에서 BRK.B 같은 티커를
  yfinance 형식(BRK-B, "."→"-")으로 바꿔 쓴다. EODHD의 Code 필드가 어떤 구분자를 쓰는지는
  실제 응답을 받아 봐야 확실하다 — 이 스크립트는 원본과 "."↔"-" 두 표기를 모두 시도해
  매칭한다.

실행(사용자가 .env에 EODHD_API_KEY를 채운 뒤 직접):
    python -u -m scripts.fund_check_eodhd_delisted

출력:
    outputs/backtest/fund_dataqc/eodhd_delisted_match.csv — missing_tickers.csv 40개
    각각에 대해 EODHD 상장폐지 목록에서 찾았는지, 찾았다면 EODHD Name·Isin·Type.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

MISSING_CSV_PATH = ROOT / "outputs" / "backtest" / "fund_dataqc" / "missing_tickers.csv"
OUT_PATH = ROOT / "outputs" / "backtest" / "fund_dataqc" / "eodhd_delisted_match.csv"


class EodhdFetchError(RuntimeError):
    """EODHD API 호출 실패(키 없음·요금제 제한·네트워크 오류 등)."""


def fetch_delisted_list(exchange_code: str, api_key: str, timeout: float = 30.0) -> list[dict]:
    """EODHD exchange-symbol-list?delisted=1을 받아 원본 레코드 목록을 돌려준다.

    입력: exchange_code(예: "US"), api_key
    출력: [{"Code":..., "Name":..., "Exchange":..., "Type":..., "Isin":...}, ...]
    예외: EodhdFetchError — 메시지에 api_key 값은 절대 넣지 않는다(CLAUDE.md 보안).
    """
    import requests

    url = f"https://eodhd.com/api/exchange-symbol-list/{exchange_code}"
    try:
        resp = requests.get(url, params={"delisted": 1, "api_token": api_key, "fmt": "json"}, timeout=timeout)
    except requests.RequestException as exc:
        raise EodhdFetchError(f"EODHD 요청 실패(네트워크): {exc}") from None
    if resp.status_code != 200:
        raise EodhdFetchError(f"EODHD 요청 실패: HTTP {resp.status_code} (요금제 제한이면 402/403)")
    try:
        data = resp.json()
    except ValueError:
        raise EodhdFetchError("EODHD 응답이 JSON이 아닙니다 — 응답 형식이 바뀌었을 수 있음") from None
    if not isinstance(data, list):
        raise EodhdFetchError(f"EODHD 응답 형식이 예상과 다릅니다: {type(data)}")
    return data


def _ticker_variants(ticker: str) -> set[str]:
    """"."<->"-" 표기 차이를 흡수해 매칭 후보를 만든다 (순수 함수)."""
    return {ticker, ticker.replace("-", "."), ticker.replace(".", "-")}


def match_against_delisted(our_tickers: list[str], eodhd_records: list[dict]) -> dict[str, dict | None]:
    """our_tickers 각각을 eodhd_records(Code 필드)와 매칭한다 (순수 함수 — 네트워크 없음).

    출력: {ticker: EODHD 레코드(dict) 또는 None(못 찾음)}
    """
    by_code = {rec.get("Code", ""): rec for rec in eodhd_records}
    out: dict[str, dict | None] = {}
    for t in our_tickers:
        found = None
        for variant in _ticker_variants(t):
            if variant in by_code:
                found = by_code[variant]
                break
        out[t] = found
    return out


def load_our_missing_tickers() -> list[str]:
    if not MISSING_CSV_PATH.exists():
        raise FileNotFoundError(f"{MISSING_CSV_PATH} 없음 — 먼저 scripts/fund_dataqc_report.py를 돌리세요.")
    with open(MISSING_CSV_PATH, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [r["ticker"] for r in rows if r.get("needed_price_range") != "겹치는 구간 없음(연구 구간 밖)"]


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    api_key = (os.environ.get("EODHD_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit(
            ".env에 EODHD_API_KEY가 없습니다. .env.example을 보고 무료 가입한 키를 .env에 넣은 뒤 다시 실행하세요."
        )

    import yaml

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    exchange_code = cfg["eodhd_delisted_check"]["exchange_code"]
    cache_path = ROOT / cfg["eodhd_delisted_check"]["cache_path"]

    our_tickers = load_our_missing_tickers()
    print(f"확인 대상 {len(our_tickers)}개 종목(연구 구간과 겹치는 것만) — missing_tickers.csv 기준", flush=True)

    if cache_path.exists():
        print(f"캐시 사용: {cache_path}", flush=True)
        records = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        print(f"EODHD API 호출 중 (거래소={exchange_code}, 무료 요금제 일일 20회 한도 — 이 호출로 1회 소모)...", flush=True)
        records = fetch_delisted_list(exchange_code, api_key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        print(f"상장폐지 레코드 {len(records)}건 수신, 캐시 저장: {cache_path}", flush=True)

    matches = match_against_delisted(our_tickers, records)
    found = sum(1 for v in matches.values() if v is not None)
    print(f"매칭 결과: {found}/{len(our_tickers)}개 EODHD 상장폐지 목록에서 확인됨", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "found_on_eodhd", "eodhd_name", "eodhd_type", "eodhd_isin"])
        writer.writeheader()
        for t in our_tickers:
            rec = matches[t]
            writer.writerow(
                {
                    "ticker": t,
                    "found_on_eodhd": rec is not None,
                    "eodhd_name": rec.get("Name") if rec else "",
                    "eodhd_type": rec.get("Type") if rec else "",
                    "eodhd_isin": rec.get("Isin") if rec else "",
                }
            )
    print(f"결과 저장: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
