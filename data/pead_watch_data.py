"""실적급등 추천 장부(pead_watch) 입출력 — 월말 명단, EDGAR 8-K 접수 목록·색인 Accepted (docs/prereg/pead_watch.md 2·3장).

네트워크 실패는 예외로 올리고, 모아 보고하는 일은 engine/pead_watch가 한다. SEC 요청은 data/edgar._get(초당 10회 이하).
"""

from __future__ import annotations

import re
import time

import pandas as pd

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_UA = "nasdaq100-screener/1.0 (research)"
INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{nodash}/{accn}-index.htm"


def parse_sp100_wikitext(txt: str) -> set[str]:
    """위키 S&P 100 문서 원문 → 명단 표의 티커 집합 (yfinance 형식, BRK.B → BRK-B). 표가 없으면 빈 집합.
    scripts/sw1_data_check.parse_sp100_wikitext와 같은 규칙(봉인 ⓐ 명단과 같게)."""
    i = txt.find("! Symbol")
    if i < 0:
        return set()
    j = txt.find("|}", i)
    body = txt[i: j if j > 0 else None]
    found = re.findall(r"\|-\s*\n\|\s*(?:\[\[[^|\]]*\|)?\s*([A-Z]{1,5}(?:[.\-/][A-Z]{1,2})?)\s*(?:\]\])?\s*(?:\n|\|\|)", body)
    return {t.replace(".", "-").replace("/", "-") for t in found}


def parse_index_accepted(html: str) -> str | None:
    """EDGAR 공시 색인 페이지 → Accepted 값('YYYY-MM-DD HH:MM:SS', 동부시간). SW1과 같은 규칙."""
    m = re.search(r'Accepted</div>\s*<div class="info">([^<]+)', html)
    return m.group(1).strip() if m else None


def _wiki_get(params: dict) -> dict:
    import requests

    for attempt in range(4):
        r = requests.get(WIKI_API, params={**params, "format": "json", "formatversion": 2},
                         headers={"User-Agent": WIKI_UA}, timeout=30)
        if r.status_code == 200:
            return r.json()
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"위키 API HTTP {r.status_code}")


def sp100_at(month_end) -> tuple[set[str], str]:
    """그 달 말(그날 끝 전) 마지막 판의 위키 「S&P 100」 명단. 출력: (티커 집합, 출처 문구 'wiki rev …')."""
    end = (pd.Timestamp(month_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    j = _wiki_get({"action": "query", "prop": "revisions", "titles": "S&P 100", "rvlimit": 1, "rvstart": end,
                   "rvdir": "older", "rvprop": "ids|timestamp|content", "rvslots": "main"})
    rev = j["query"]["pages"][0]["revisions"][0]
    tick = parse_sp100_wikitext(rev["slots"]["main"]["content"])
    if len(tick) < 90:
        raise RuntimeError(f"S&P 100 명단 {len(tick)}개 — 표 형식 바뀜 의심")
    return tick, f"wiki rev {rev['revid']} ({rev['timestamp'][:10]})"


def ndx_at(month_end, current: set[str]) -> tuple[set[str], str]:
    """그 달 말 나스닥100 명단 = 현재 공식 구성 + 위키 변경 이력 역산 (data/universe_history). 출력: (집합, 출처 문구)."""
    from data import universe_history as uh

    s = set(uh.point_in_time_universe(pd.Timestamp(month_end).date(), current))
    if len(s) < 95:
        raise RuntimeError(f"나스닥100 명단 {len(s)}개 — 이상")
    return s, "현재 구성 + 위키 변경 이력"


def recent_8k(cik: int, user_agent: str) -> list[dict]:
    """EDGAR submissions 최근 접수 목록 → 8-K·8-K/A 행 [{form, items, filing_date, accn}]."""
    from data import edgar

    r = edgar._get(edgar.SUBMISSIONS_URL_TMPL.format(cik=int(cik)), user_agent).json()["filings"]["recent"]
    items = r.get("items") or [""] * len(r["form"])
    return [{"form": f, "items": it, "filing_date": d, "accn": a}
            for f, it, d, a in zip(r["form"], items, r["filingDate"], r["accessionNumber"]) if f in ("8-K", "8-K/A")]


def accepted_et(cik: int, accn: str, user_agent: str) -> pd.Timestamp:
    """공시 색인 페이지의 Accepted(동부시간, tz 없음). 못 찾으면 RuntimeError."""
    from data import edgar

    html = edgar._get(INDEX_URL.format(cik=int(cik), nodash=accn.replace("-", ""), accn=accn), user_agent).text
    v = parse_index_accepted(html)
    if not v:
        raise RuntimeError(f"{accn} 색인에 Accepted 없음")
    return pd.Timestamp(v)
