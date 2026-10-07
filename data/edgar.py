"""SEC EDGAR 재무 수집기 (해자 분석 H1, docs/design/moat_plan.md).

무료 공식 API만 쓴다:
- 티커 -> CIK: https://www.sec.gov/files/company_tickers.json
- 재무(XBRL 전체): https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json

User-Agent는 .env의 SEC_USER_AGENT에서 읽는다 — SEC는 모든 요청에 연락처가 포함된
User-Agent를 요구하고, 없으면 차단한다. 값이 없으면 EdgarConfigError로 멈춘다(CLAUDE.md
보안 원칙과 같은 이유로 이 값 자체를 로그·예외 메시지에 그대로 넣지는 않는다 — 다만
User-Agent는 비밀이 아니라 "누가 요청했는지 밝히는" 값이라 에러 메시지엔 존재 여부만 적는다).

초당 10회 이하로 요청한다(SEC fair access 권고). 결과는 캐시(data/cache/edgar/)에 저장해
재실행 시 네트워크를 다시 안 탄다 — force_refresh=True로 강제 갱신 가능.

이 모듈은 "원본 XBRL 사실을 그대로 꺼내는 것"까지만 한다. 해자 지표 계산(M1~M5), 태그
대체 목록, 세율 가정 같은 도메인 로직은 core/moat.py에 있다 — 이 모듈은 그 재료를 공급한다.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache" / "edgar"
FACTS_CACHE_DIR = CACHE_DIR / "facts"
TICKER_MAP_CACHE_PATH = CACHE_DIR / "company_tickers.json"

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL_TMPL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

_MIN_REQUEST_INTERVAL_SEC = 0.11  # 초당 10회 이하(SEC 권고) — 여유를 두고 0.11초


class EdgarConfigError(RuntimeError):
    """SEC_USER_AGENT가 없을 때 낸다. 멈추고 사용자에게 알린다."""


class EdgarFetchError(RuntimeError):
    """EDGAR 요청 실패(네트워크·HTTP 오류·응답 형식 오류)."""


class _RateLimiter:
    """초당 _MIN_REQUEST_INTERVAL_SEC 이하 간격만 허용하는 모듈 전역 상태.

    테스트에서 실제로 sleep하지 않도록 sleep_fn을 주입할 수 있게 한다.
    """

    def __init__(self, min_interval: float = _MIN_REQUEST_INTERVAL_SEC, sleep_fn=time.sleep, time_fn=time.monotonic):
        self.min_interval = min_interval
        self._sleep = sleep_fn
        self._now = time_fn
        self._last_call: float | None = None

    def wait(self) -> None:
        now = self._now()
        if self._last_call is not None:
            elapsed = now - self._last_call
            if elapsed < self.min_interval:
                self._sleep(self.min_interval - elapsed)
        self._last_call = now


_rate_limiter = _RateLimiter()


def get_user_agent() -> str:
    """.env의 SEC_USER_AGENT를 읽는다. 없으면 EdgarConfigError로 멈춘다.

    출력: User-Agent 문자열(예: "nasdaq100-screener you@example.com")
    예외: EdgarConfigError — 값이 없거나 공백뿐일 때
    """
    import os

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    ua = (os.environ.get("SEC_USER_AGENT") or "").strip()
    if not ua:
        raise EdgarConfigError(
            ".env에 SEC_USER_AGENT가 없습니다. SEC는 연락처가 포함된 User-Agent를 요구합니다 — "
            "예: SEC_USER_AGENT=\"nasdaq100-screener you@example.com\" (.env.example 참고)."
        )
    return ua


def _get(url: str, user_agent: str, timeout: float = 30.0):
    """속도 제한을 지키며 GET 요청을 보낸다 (네트워크). 실패 시 EdgarFetchError."""
    import requests

    _rate_limiter.wait()
    try:
        resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=timeout)
    except requests.RequestException as exc:
        raise EdgarFetchError(f"EDGAR 요청 실패(네트워크): {url} — {exc}") from None
    if resp.status_code != 200:
        raise EdgarFetchError(f"EDGAR 요청 실패: HTTP {resp.status_code} — {url}")
    return resp


def fetch_ticker_to_cik(user_agent: str | None = None, force_refresh: bool = False) -> dict[str, int]:
    """SEC의 티커->CIK 전체 목록을 받는다(캐시 우선).

    입력: user_agent(생략하면 get_user_agent()), force_refresh(True면 캐시 무시하고 새로 받음)
    출력: {티커(대문자): CIK(정수)}
    예외: EdgarConfigError(User-Agent 없음), EdgarFetchError(요청 실패)
    """
    if not force_refresh and TICKER_MAP_CACHE_PATH.exists():
        raw = json.loads(TICKER_MAP_CACHE_PATH.read_text(encoding="utf-8"))
        return {k: int(v) for k, v in raw.items()}

    ua = user_agent if user_agent is not None else get_user_agent()
    resp = _get(TICKER_MAP_URL, ua)
    try:
        data = resp.json()
    except ValueError:
        raise EdgarFetchError(f"{TICKER_MAP_URL} 응답이 JSON이 아님") from None

    out: dict[str, int] = {}
    # company_tickers.json은 {"0": {"cik_str":..., "ticker":..., "title":...}, "1": {...}, ...}
    # 형태다("fields"/"data" 배열 구조가 아니다 — 2026-10-01 직접 확인).
    for row in data.values():
        out[str(row["ticker"]).upper()] = int(row["cik_str"])

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TICKER_MAP_CACHE_PATH.write_text(json.dumps(out), encoding="utf-8")
    return out


def fetch_company_facts(cik: int, user_agent: str | None = None, force_refresh: bool = False) -> dict:
    """한 회사의 전체 XBRL 사실(companyfacts)을 받는다(캐시 우선).

    입력: cik(정수), user_agent(생략하면 get_user_agent()), force_refresh
    출력: SEC companyfacts 원본 JSON(dict) — facts["facts"]["us-gaap" 또는 "ifrs-full"][태그명]["units"][단위]
         가 사실 목록이다.
    예외: EdgarConfigError, EdgarFetchError
    """
    cache_path = FACTS_CACHE_DIR / f"CIK{cik:010d}.json"
    if not force_refresh and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    ua = user_agent if user_agent is not None else get_user_agent()
    resp = _get(COMPANY_FACTS_URL_TMPL.format(cik=cik), ua)
    try:
        data = resp.json()
    except ValueError:
        raise EdgarFetchError(f"CIK{cik:010d} companyfacts 응답이 JSON이 아님") from None

    FACTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    return data


def extract_fact_entries(company_facts: dict, taxonomy: str, tag: str) -> list[dict]:
    """companyfacts에서 한 taxonomy·태그의 사실 목록을 그대로 꺼낸다 (순수 함수).

    입력: company_facts(fetch_company_facts 결과), taxonomy("us-gaap"|"ifrs-full"),
         tag(예: "OperatingIncomeLoss")
    출력: [{"val","start"(있으면),"end","filed","fy","fp","form","accn","unit"}, ...] —
         단위(units)가 여러 개면(드묾) 전부 합쳐서 돌려준다. taxonomy·태그가 없으면 [].
    """
    node = company_facts.get("facts", {}).get(taxonomy, {}).get(tag)
    if node is None:
        return []
    out = []
    for unit, entries in node.get("units", {}).items():
        for e in entries:
            out.append({**e, "unit": unit})
    return out


def extract_duration_fact_entries(
    company_facts: dict, taxonomy: str, tag: str, as_of: date,
    min_days: int, max_days: int, forms: tuple[str, ...] = ("10-Q",),
) -> list[dict]:
    """extract_fact_entries 결과 중 기간(duration) 길이가 [min_days, max_days] 안인 것만 고른다
    (순수 함수, 해자 약화 경보 B3용 분기 추출 — core/moat_alert.py·core/moat_paper.py에서 쓴다).

    XBRL 10-Q는 같은 개념·같은 분기말(end)에 대해 "그 분기 3개월"과 "회계연도 시작부터
    그 분기까지 누적"을 둘 다 start만 다르게 내는 경우가 많다 — (end-start) 일수로 둘을
    구분한다(3개월≈80~100일, 9개월 누적≈260~290일, 호출부가 min_days·max_days로 지정).

    입력: company_facts, taxonomy, tag, as_of(filed가 이 날짜보다 미래인 사실은 제외 —
         미래 데이터 금지), min_days·max_days(기간 일수 범위, 양끝 포함), forms(허용할
         form 목록, 기본 10-Q만)
    출력: start·filed·end가 모두 있고 조건을 만족하는 extract_fact_entries 항목 그대로
         (정렬 안 함 — 호출부가 필요에 따라 정렬·병합한다)
    """
    as_of_str = as_of.isoformat()
    out = []
    for e in extract_fact_entries(company_facts, taxonomy, tag):
        if e.get("form") not in forms:
            continue
        start, filed, end = e.get("start"), e.get("filed"), e.get("end")
        if not start or not filed or not end:
            continue
        if filed > as_of_str:
            continue
        days = (date.fromisoformat(end) - date.fromisoformat(start)).days
        if min_days <= days <= max_days:
            out.append(e)
    return out
