"""실적급등 추천 장부 기록 꼬리표 — 순수 함수 (docs/prereg/pead_watch_tags.md).

판정·매매에 쓰지 않는 기록용. 모든 값은 반응일 d0 장 마감(16:00 동부)까지 알 수 있는 정보만 쓴다.
네트워크·파일 접근 없음. 입력은 일봉·검색 결과·접수 목록·본문 글, 출력은 숫자 또는 None(빈칸).
"""

from __future__ import annotations

import html as _html
import re

import numpy as np
import pandas as pd

SMA_WINDOW = 200
HIGH_WINDOW = 252
CLOSE_CUTOFF_HOUR = 16  # d0 16:00 동부
WEEKDAYS = ("월", "화", "수", "목", "금", "토", "일")
TAG_COLS = ("tag_sma200_above", "tag_sma200_dist", "tag_hi52", "tag_close_pos", "tag_n202_d0", "tag_n202_prev",
            "tag_weekday", "tag_q_before_close", "tag_lm_pos", "tag_lm_neg", "tag_lm_unc", "tag_lm_words", "tag_errors")


def _upto(close: pd.Series, d0) -> pd.Series | None:
    """d0까지 종가(결측 뺌). d0 행이 없으면 None."""
    d0 = pd.Timestamp(d0)
    h = close.loc[:d0].dropna()
    return h if len(h) and h.index[-1] == d0 else None


def sma200(close: pd.Series, d0, window: int = SMA_WINDOW) -> tuple[int | None, float | None]:
    """입력: 종가(날짜 오름차순), d0. 출력: (d0 종가 > d0 포함 window개 평균이면 1 아니면 0, 거리 %). 자료 부족이면 (None, None)."""
    h = _upto(close, d0)
    if h is None or len(h) < window:
        return None, None
    m = float(h.iloc[-window:].mean())
    c = float(h.iloc[-1])
    if not m > 0:
        return None, None
    return int(c > m), (c / m - 1) * 100


def high52(close: pd.Series, d0, window: int = HIGH_WINDOW) -> float | None:
    """입력: 종가, d0. 출력: (d0 종가 ÷ d0 포함 window개 종가 최고 − 1) × 100 (≤ 0). 자료 부족이면 None."""
    h = _upto(close, d0)
    if h is None or len(h) < window:
        return None
    hi = float(h.iloc[-window:].max())
    return (float(h.iloc[-1]) / hi - 1) * 100 if hi > 0 else None


def close_position(high: float, low: float, close: float) -> float | None:
    """d0 (종가 − 저가) ÷ (고가 − 저가). 고가 ≤ 저가·값 없음·종가가 범위 밖이면 None."""
    v = np.array([high, low, close], dtype=float)
    if not np.isfinite(v).all() or not high > low or not (low <= close <= high):
        return None
    return float((close - low) / (high - low))


def weekday_kr(d0) -> str:
    """날짜 → 한국어 요일 한 글자."""
    return WEEKDAYS[pd.Timestamp(d0).dayofweek]


def count_8k_202(hits: list[dict]) -> int:
    """EDGAR 전문 검색 결과 문서(_source) 목록 → form '8-K'(8-K/A 제외)이고 items에 2.02가 있는 서로 다른 CIK 수."""
    ciks: set[str] = set()
    for h in hits:
        if h.get("form") != "8-K" or "2.02" not in [str(x).strip() for x in (h.get("items") or [])]:
            continue
        ciks.update(str(c).lstrip("0") for c in (h.get("ciks") or []))
    return len(ciks)


def q_report_candidates(filings: list[dict], k8_filing_date: str, d0) -> list[dict]:
    """접수 목록 [{form, filing_date, accn}] → 같은 분기 10-Q/10-K 후보(정정 제외, 접수일이 8-K 접수일 ~ d0). 접수일 순."""
    d = pd.Timestamp(d0).date().isoformat()
    out = [f for f in filings if f["form"] in ("10-Q", "10-K") and k8_filing_date <= f["filing_date"] <= d]
    return sorted(out, key=lambda f: (f["filing_date"], f["accn"]))


def q_before_close(cands: list[dict], d0, accepted: dict[str, pd.Timestamp | None]) -> int | None:
    """후보 중 d0 장 마감 전에 나온 것이 있으면 1, 없으면 0.

    입력: cands(q_report_candidates 출력), accepted(접수일 = d0인 accn → 색인 Accepted 동부, 받기 실패면 None).
    접수일 < d0 → 1. 접수일 = d0 → Accepted < d0 16:00이면 1. 같은 날 건 Accepted가 없어 정할 수 없으면 None.
    """
    d0 = pd.Timestamp(d0).normalize()
    unknown = False
    for f in cands:
        fd = pd.Timestamp(f["filing_date"])
        if fd < d0:
            return 1
        a = accepted.get(f["accn"])
        if a is None:
            unknown = True
        elif pd.Timestamp(a) < d0 + pd.Timedelta(hours=CLOSE_CUTOFF_HOUR):
            return 1
    return None if unknown else 0


def is_ex991(doc_type: str) -> bool:
    """색인 문서 Type이 EX-99.1(번호 표기만 다른 EX-99.01 포함)이면 True. 번호 없는 EX-99·EX-99.2는 False."""
    m = re.fullmatch(r"EX-99\.0*1", str(doc_type).strip().upper())
    return m is not None


def html_words(doc: str) -> list[str]:
    """HTML(또는 글) → 대문자 단어 목록. 스크립트·스타일·태그를 지우고 엔티티를 푼 뒤 [A-Za-z]+."""
    t = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", doc)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    t = _html.unescape(t)
    return [w.upper() for w in re.findall(r"[A-Za-z]+", t)]


def lm_tone(words: list[str], lists: dict[str, set[str]]) -> dict | None:
    """단어 목록 + {'positive','negative','uncertainty': 대문자 단어 집합} → 비율(%)과 단어 수. 단어 0개면 None."""
    n = len(words)
    if n == 0:
        return None
    cnt = {k: sum(1 for w in words if w in s) for k, s in lists.items()}
    return {"pos": cnt["positive"] / n * 100, "neg": cnt["negative"] / n * 100, "unc": cnt["uncertainty"] / n * 100,
            "words": n}


def price_tags(px: pd.DataFrame, d0) -> tuple[dict, list[str]]:
    """일봉(open·high·low·close) → 가격 꼬리표 칸과 빈칸 사유. d0 뒤 행은 보지 않는다."""
    d0 = pd.Timestamp(d0)
    h = px.loc[:d0]
    out, why = {}, []
    above, dist = sma200(h["close"], d0)
    out.update(tag_sma200_above=above, tag_sma200_dist=dist)
    if above is None:
        why.append("200일선: 종가 200개 미만")
    out["tag_hi52"] = high52(h["close"], d0)
    if out["tag_hi52"] is None:
        why.append("52주 최고: 종가 252개 미만")
    cp = None
    if len(h) and h.index[-1] == d0 and {"high", "low"} <= set(h.columns):
        r = h.iloc[-1]
        cp = close_position(float(r["high"]), float(r["low"]), float(r["close"]))
    out["tag_close_pos"] = cp
    if cp is None:
        why.append("종가 위치: 고가·저가 없음")
    out["tag_weekday"] = weekday_kr(d0)
    return out, why
