"""나스닥 100 시점별(point-in-time) 구성 종목 재구성 (P5-1 2번).

위키백과 "Historical components of the Nasdaq-100" 문서의 변경 이력 표(위키텍스트)를
파싱해, 현재 구성 종목(data.universe.get_universe)에서 과거로 거슬러 올라가며 특정
날짜의 구성 종목을 재구성한다. 순수 계산 부분(parse_changes, reconstruct_membership,
membership_checkpoints, universe_on)은 네트워크 없이 테스트할 수 있다.

생존자 편향(survivorship bias) 회피: 백테스트가 "오늘 남아 있는 종목"만 대상으로
과거 신호를 판정하면 상장폐지·편입 제외된 종목의 손실이 결과에서 빠져 성과가
부풀려진다. 이 모듈로 그날 실제 구성 종목만 신호 대상으로 쓸 수 있으면
`SURVIVORSHIP_BIAS = FALSE`, 못 구하면(이 모듈이 실패하면) 현재 구성 종목을 쓰고
`SURVIVORSHIP_BIAS = TRUE`로 표시한다 (engine/backtest.py가 최종 판단).
"""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from data.universe import _to_yfinance_ticker

DATA_DIR = Path(__file__).resolve().parent
CACHE_PATH = DATA_DIR / "cache" / "nasdaq100_changes.json"
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
WIKI_PAGE_TITLE = "Historical_components_of_the_Nasdaq-100"

_MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}


@dataclass(frozen=True)
class IndexChange:
    """나스닥 100 구성 종목 변경 한 건. added/removed는 yfinance 형식 티커 또는 None."""

    date: date
    added: str | None
    removed: str | None


def fetch_changes_wikitext(timeout: float = 30.0) -> str:
    """위키백과 "Historical components of the Nasdaq-100" 문서의 원본 위키텍스트를 받는다.

    입력: timeout(초)
    출력: 위키텍스트 문자열
    예외: 네트워크·응답 형식 오류는 그대로 올린다 (호출부가 경고로 모은다)
    """
    import requests

    resp = requests.get(
        WIKI_API_URL,
        params={"action": "parse", "page": WIKI_PAGE_TITLE, "prop": "wikitext", "format": "json"},
        headers={"User-Agent": "nasdaq100-screener/1.0 (backtest research)"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["parse"]["wikitext"]["*"]


def _parse_date_cell(text: str) -> date | None:
    m = re.match(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", text.strip())
    if not m:
        return None
    month = _MONTHS.get(m.group(1))
    if month is None:
        return None
    return date(int(m.group(3)), month, int(m.group(2)))


def _clean_ticker_cell(text: str) -> str | None:
    """표의 Added/Removed 티커 칸을 정리한다. 각주·위키링크를 떼고 yfinance 형식으로 바꾼다.

    칸이 비어 있거나(변경이 편입만/제외만인 경우), 공백이 섞여 있어 티커로 보기
    어려우면(표 형식이 어긋난 행 방어) None.
    """
    text = text.strip()
    if not text:
        return None
    text = re.sub(r"<ref[^>]*/?>.*?(</ref>|$)", "", text, flags=re.S)
    text = re.sub(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", r"\1", text)
    text = text.strip()
    if not text or " " in text:
        return None
    return _to_yfinance_ticker(text)


def parse_changes(wikitext: str) -> list[IndexChange]:
    """"changes" 표(위키텍스트)를 파싱한다 (순수 함수, 네트워크 없음 — 테스트 가능).

    입력: 문서 전체 위키텍스트
    출력: IndexChange 목록 (표에 나온 순서 — 보통 최신이 먼저)
    """
    m = re.search(r'\{\|\s*class="wikitable[^\n]*id="changes"(.*?)\n\|\}', wikitext, re.S)
    if not m:
        return []
    body = m.group(1)
    changes: list[IndexChange] = []
    for chunk in body.split("\n|-"):
        chunk = chunk.strip("\n")
        if not chunk or chunk.lstrip().startswith("!"):
            continue
        cells = chunk.split("\n|")
        cells[0] = cells[0].lstrip("|")
        cells = [c.strip() for c in cells]
        if len(cells) < 5:
            continue
        d = _parse_date_cell(cells[0])
        if d is None:
            continue
        added = _clean_ticker_cell(cells[1])
        removed = _clean_ticker_cell(cells[3])
        if added is None and removed is None:
            continue
        changes.append(IndexChange(date=d, added=added, removed=removed))
    return changes


def reconstruct_membership(current_tickers: set[str], changes: list[IndexChange], as_of: date) -> set[str]:
    """현재 구성 종목에서 changes를 거슬러 적용해 as_of 날짜(포함, 그날 발효분까지)의
    구성 종목을 재구성한다 (순수 함수).

    날짜가 as_of보다 뒤인 변경을 최신 것부터 거꾸로 되돌린다: 그 변경으로 편입된
    종목은 빼고, 제외된 종목은 되돌려 넣는다.

    입력: current_tickers(오늘 구성 종목 집합), changes(parse_changes 결과, 순서 무관 —
         이 함수가 날짜 내림차순으로 다시 정렬한다), as_of
    출력: as_of 날짜의 구성 종목 집합
    """
    members = set(current_tickers)
    for change in sorted(changes, key=lambda c: c.date, reverse=True):
        if change.date <= as_of:
            break
        if change.added and change.added in members:
            members.discard(change.added)
        if change.removed:
            members.add(change.removed)
    return members


def membership_checkpoints(
    current_tickers: set[str], changes: list[IndexChange], start: date
) -> list[tuple[date, frozenset[str]]]:
    """start부터 오늘까지, 구성 종목이 바뀌는 시점마다 스냅샷을 만든다 (순수 함수).

    백테스트가 거래일마다 매번 reconstruct_membership을 다시 계산하지 않고
    universe_on()으로 이분 탐색만 하도록 미리 만들어 두는 표다.

    입력: current_tickers, changes, start(이 날짜부터)
    출력: [(그 구성이 적용되기 시작하는 날짜, 그 날짜의 구성 종목 집합), ...] 날짜
         오름차순, start 포함(맨 앞).
    """
    members = reconstruct_membership(current_tickers, changes, start)
    checkpoints: list[tuple[date, frozenset[str]]] = [(start, frozenset(members))]
    forward = sorted((c for c in changes if c.date > start), key=lambda c: c.date)
    i = 0
    while i < len(forward):
        d = forward[i].date
        while i < len(forward) and forward[i].date == d:
            c = forward[i]
            if c.added:
                members.add(c.added)
            if c.removed:
                members.discard(c.removed)
            i += 1
        checkpoints.append((d, frozenset(members)))
    return checkpoints


def universe_on(checkpoints: list[tuple[date, frozenset[str]]], as_of: date) -> frozenset[str]:
    """membership_checkpoints 결과에서 as_of 날짜의 구성 종목을 이분 탐색으로 찾는다 (순수 함수).

    as_of가 checkpoints의 첫 날짜보다 이르면 빈 집합(범위 밖).
    """
    dates = [c[0] for c in checkpoints]
    idx = bisect_right(dates, as_of) - 1
    if idx < 0:
        return frozenset()
    return checkpoints[idx][1]


def _load_cache() -> list[IndexChange] | None:
    if not CACHE_PATH.exists():
        return None
    try:
        raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return [IndexChange(date=date.fromisoformat(r["date"]), added=r["added"], removed=r["removed"]) for r in raw]
    except Exception:
        return None


def _save_cache(changes: list[IndexChange]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw = [{"date": c.date.isoformat(), "added": c.added, "removed": c.removed} for c in changes]
    CACHE_PATH.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")


def get_changes(force_refresh: bool = False) -> list[IndexChange]:
    """변경 이력을 구한다 (캐시 우선, 없거나 force_refresh면 위키백과에서 새로 받는다).

    예외: 캐시도 없고 네트워크도 실패하면 그대로 올린다 (호출부가 CURRENT_CONSTITUENTS
         폴백 여부를 결정한다).
    """
    if not force_refresh:
        cached = _load_cache()
        if cached:
            return cached
    wikitext = fetch_changes_wikitext()
    changes = parse_changes(wikitext)
    if not changes:
        raise ValueError("나스닥 100 변경 이력 표를 찾지 못했습니다 (위키백과 문서 구조가 바뀌었을 수 있음)")
    _save_cache(changes)
    return changes


def point_in_time_universe(as_of: date, current_tickers: set[str] | None = None) -> set[str]:
    """as_of 날짜의 나스닥 100 구성 종목을 구한다 (오늘 구성 종목 + 변경 이력 역산).

    입력: as_of, current_tickers(생략하면 data.universe.get_universe()로 오늘 구성 종목을 받는다)
    출력: 그날 구성 종목 티커 집합(yfinance 형식)
    """
    if current_tickers is None:
        from data.universe import get_universe

        current_tickers = set(get_universe()["ticker"])
    changes = get_changes()
    return reconstruct_membership(current_tickers, changes, as_of)
