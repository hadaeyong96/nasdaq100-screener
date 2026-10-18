"""신기술 레이더 G0 — 등록 단어 ↔ 10-K·10-Q 회사 연결 (순수 함수). 네트워크·파일은 data/radar/edgar_fetch.py.

- 검색어: 단수·복수를 OR로 묶은 구 검색(EDGAR 전문 검색은 둘을 따로 센다). 기간 2001-01-01 ~ 2021-12-31.
- 티커: 그 공시 접수일에 그 CIK가 쓰던 티커 — 시점별 명단 구간(CIK, 티커, 시작, 끝)에 접수일이 들어갈 때만. 못 찾으면 매칭 실패.
- 언급 수·발췌: 문서 본문(HTML 태그 제거)에서 표현을 대소문자 무시, 단어 사이 공백·하이픈 허용, 끝 단어 단수·복수로 센다.
  발췌는 첫 언급 앞뒤 200자.
"""

from __future__ import annotations

import html as _html
import re

import pandas as pd

from core.radar import term_monthly as tm

START, END = "2001-01-01", "2021-12-31"
FORMS = "10-K,10-Q"
EXCERPT = 200


def plural(w: str) -> str:
    """끝 단어 복수형 (검색용, 간단 규칙)."""
    if w.endswith("y") and len(w) > 2 and w[-2] not in "aeiou":
        return w[:-1] + "ies"
    if w.endswith(("s", "x", "z", "ch", "sh")):
        return w + "es"
    return w + "s"


def query(term: str) -> str:
    """EDGAR 전문 검색어: "단수" OR "복수"."""
    ws = term.split(" ")
    pl = " ".join(ws[:-1] + [plural(ws[-1])])
    return f'"{term}" OR "{pl}"'


def efts_params(term: str, frm: int = 0) -> dict:
    return {"q": query(term), "forms": FORMS, "dateRange": "custom", "startdt": START, "enddt": END, "from": frm}


def parse_hits(j: dict) -> tuple[list[dict], int]:
    """efts 응답 → (공시 목록, 전체 수). 공시: adsh, filename, cik(첫 CIK, 정수), file_date, form, name."""
    h = j.get("hits", {})
    out = []
    for x in h.get("hits", []):
        s = x.get("_source", {})
        adsh, _, fn = x.get("_id", "").partition(":")
        ciks = s.get("ciks") or []
        out.append({"adsh": adsh, "filename": fn, "cik": int(ciks[0]) if ciks else None, "file_date": s.get("file_date"),
                    "form": s.get("form") or s.get("file_type"), "name": (s.get("display_names") or [""])[0]})
    return out, int(h.get("total", {}).get("value", 0))


def build_intervals(rows: list[tuple]) -> pd.DataFrame:
    """[(cik, ticker, start, end)] → 구간 표 (빈 CIK·티커는 버림, 날짜 문자열 YYYY-MM-DD)."""
    d = pd.DataFrame(rows, columns=["cik", "ticker", "start", "end"]).dropna(subset=["cik", "ticker", "start", "end"])
    d["cik"] = d["cik"].astype(int)
    d["start"] = pd.to_datetime(d["start"]).dt.strftime("%Y-%m-%d")
    d["end"] = pd.to_datetime(d["end"]).dt.strftime("%Y-%m-%d")
    return d.drop_duplicates().reset_index(drop=True)


def match_ticker(iv: pd.DataFrame, cik: int | None, file_date: str) -> str | None:
    """접수일에 그 CIK가 쓰던 티커. 여러 개면 시작이 가장 늦은 구간, 같으면 티커 순 첫 번째."""
    if cik is None or not file_date:
        return None
    c = iv[(iv["cik"] == cik) & (iv["start"] <= file_date) & (iv["end"] >= file_date)]
    if c.empty:
        return None
    return str(c.sort_values(["start", "ticker"], ascending=[False, True]).iloc[0]["ticker"])


_TAGS = re.compile(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", re.S | re.I)
_SP = re.compile(r"\s+")


def html_to_text(raw: str) -> str:
    """공시 HTML(또는 텍스트) → 공백 정리한 본문."""
    return _SP.sub(" ", _html.unescape(_TAGS.sub(" ", raw))).strip()


def term_regex(term: str) -> re.Pattern:
    """표현 찾기: 단어 사이 공백·하이픈, 끝 단어는 단수 또는 복수, 대소문자 무시, 단어 경계."""
    ws = term.split(" ")
    last = f"(?:{re.escape(ws[-1])}|{re.escape(plural(ws[-1]))})"
    body = r"[\s\-]+".join([re.escape(w) for w in ws[:-1]] + [last])
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", re.I)


def mentions(text: str, term: str) -> tuple[int, str]:
    """(언급 수, 첫 언급 앞뒤 200자 발췌)."""
    rx = term_regex(term)
    hits = list(rx.finditer(text))
    if not hits:
        return 0, ""
    a, b = hits[0].span()
    return len(hits), text[max(0, a - EXCERPT):b + EXCERPT]


def normalize(term: str) -> str:
    return tm.normalize_term(term)
