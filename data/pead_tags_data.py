"""실적급등 장부 기록 꼬리표 입출력 — EDGAR 전문 검색(하루 8-K), 접수 목록(10-Q/10-K), 공시 색인·보도자료, LM 사전.

정의: docs/prereg/pead_watch_tags.md. 네트워크 실패는 예외로 올리고, 빈칸 처리·모아 보고는 engine/pead_watch가 한다.
SEC 요청은 data/edgar._get(초당 10회 이하).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LM_CSV = ROOT / "data" / "lm" / "lm_sentiment_words.csv"
EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{nodash}/{accn}-index.htm"
EFTS_PAGE = 100
EFTS_MAX_PAGES = 60  # 하루 6,000문서 넘으면 오류(실제는 수백)


def efts_8k_day(day: str, user_agent: str) -> list[dict]:
    """접수일 day('YYYY-MM-DD')인 8-K 계열 문서(_source) 전부. 다 못 받으면 RuntimeError."""
    import requests

    from data import edgar

    out, total = [], None
    for page in range(EFTS_MAX_PAGES):
        edgar._rate_limiter.wait()
        try:
            r = requests.get(EFTS_URL, params={"forms": "8-K", "dateRange": "custom", "startdt": day, "enddt": day,
                                               "from": page * EFTS_PAGE}, headers={"User-Agent": user_agent}, timeout=30)
        except requests.RequestException as exc:
            raise RuntimeError(f"EDGAR 전문 검색 연결 실패({type(exc).__name__})") from None
        if r.status_code != 200:
            raise RuntimeError(f"EDGAR 전문 검색 HTTP {r.status_code}")
        j = r.json()["hits"]
        total = int(j["total"]["value"])
        out += [h["_source"] for h in j["hits"]]
        if not j["hits"] or len(out) >= total:
            return out
    raise RuntimeError(f"EDGAR 전문 검색 {day}: {total}건 중 {len(out)}건만 받음")


def recent_filings(cik: int, user_agent: str, forms=("10-Q", "10-K")) -> list[dict]:
    """submissions 최근 접수 목록 → forms 행 [{form, filing_date, accn}]."""
    from data import edgar

    r = edgar._get(edgar.SUBMISSIONS_URL_TMPL.format(cik=int(cik)), user_agent).json()["filings"]["recent"]
    return [{"form": f, "filing_date": d, "accn": a}
            for f, d, a in zip(r["form"], r["filingDate"], r["accessionNumber"]) if f in forms]


def index_html(cik: int, accn: str, user_agent: str) -> str:
    """공시 색인 페이지 HTML."""
    from data import edgar

    return edgar._get(INDEX_URL.format(cik=int(cik), nodash=accn.replace("-", ""), accn=accn), user_agent).text


def parse_index_docs(html: str) -> list[dict]:
    """색인 페이지 문서 표 → [{type, url}] (sec.gov 절대 주소, 인라인 XBRL 보기 주소는 원 문서로)."""
    out = []
    for row in re.findall(r"(?is)<tr[^>]*>(.*?)</tr>", html):
        cells = re.findall(r"(?is)<td[^>]*>(.*?)</td>", row)
        m = re.search(r'href="([^"]+)"', row)
        if len(cells) < 4 or not m:
            continue
        href = m.group(1).replace("/ix?doc=", "")
        typ = re.sub(r"<[^>]+>", "", cells[3]).replace("&nbsp;", "").strip()
        out.append({"type": typ, "url": href if href.startswith("http") else "https://www.sec.gov" + href})
    return out


def fetch_text(url: str, user_agent: str) -> str:
    """SEC 문서 본문 글."""
    from data import edgar

    r = edgar._get(url, user_agent)
    r.encoding = r.encoding or "utf-8"
    return r.text


def load_lm_lists(path: Path = LM_CSV) -> dict[str, set[str]]:
    """LM 단어 목록 → {'positive','negative','uncertainty': 대문자 단어 집합}."""
    d = pd.read_csv(path, keep_default_na=False)
    return {k: set(d.loc[d[k] == 1, "word"]) for k in ("positive", "negative", "uncertainty")}
