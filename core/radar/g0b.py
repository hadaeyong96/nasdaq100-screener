"""신기술 레이더 G0-보완 — 분야 필터·조각 정리·사건 단어·꾸준한 성장 규칙·사업 설명 부분 (순수 함수).

규칙: docs/research/radar_g0b_rules.md (결과 보기 전 고정). 네트워크·파일 없음.
"""

from __future__ import annotations

import re

import numpy as np

# ① 산업 기술 분야 (논문의 대표 분야 primary 기준)
INDUSTRIAL_PREFIX = ("cs.", "eess.")
INDUSTRIAL_EXACT = frozenset({"cond-mat.mtrl-sci", "physics.app-ph", "physics.optics", "quant-ph", "q-bio.QM", "cmp-lg"})
MIN_INDUSTRIAL_SHARE = 0.60

# ② 조각·일반어
FRAGMENT_SHARE = 0.70
GENERIC = frozenset("based achieve novel trained end input model method proposed propose approach framework task performance".split())

# ③ 사건 단어 (표현 안에 단어 경계로 들어 있으면 제외)
EVENT = ("coronavirus", "covid", "pandemic", "sars", "mers", "respiratory syndrome", "social distancing", "lockdown",
         "confirmed case", "second wave", "stay at home", "outbreak", "epidemic", "vaccine rollout", "quarantine",
         "contact tracing", "wuhan", "ebola", "zika", "h1n1", "disease spread")

# ④ 꾸준한 성장
GROWTH_CAGR = 0.30
GROWTH_MIN_MENTIONS = 200


def is_industrial(primary: str) -> bool:
    """논문 대표 분야가 산업 기술 분야인지."""
    p = primary or ""
    return p.startswith(INDUSTRIAL_PREFIX) or p in INDUSTRIAL_EXACT


def generic_edge(term: str) -> bool:
    """첫 단어나 끝 단어가 일반어 목록에 있으면 True (제외)."""
    w = term.split(" ")
    return w[0] in GENERIC or w[-1] in GENERIC


def event_term(term: str) -> bool:
    """사건 단어 목록의 말이 표현 안에(단어 경계) 있으면 True (제외)."""
    return any(re.search(rf"(?<![a-z0-9]){re.escape(e)}(?![a-z0-9])", term) for e in EVENT)


def contains(trigram: str, bigram: str) -> bool:
    """3단어 표현이 2단어 표현을 이어진 단어로 품는지."""
    t, b = trigram.split(" "), bigram.split(" ")
    return len(t) == 3 and len(b) == 2 and (t[:2] == b or t[1:] == b)


def register_growth(n_docs: np.ndarray, total: np.ndarray, window: int = 12, years: int = 3,
                    min_cagr: float = GROWTH_CAGR, min_mentions: int = GROWTH_MIN_MENTIONS) -> np.ndarray:
    """꾸준한 성장 등록 달 위치(없으면 −1). 달 m: 최근 12개월 월별 비율 평균 R1, 3년 전 같은 12개월(m−47 ~ m−36) 평균 R0,
    R0 > 0 이고 (R1/R0)^(1/3) − 1 ≥ 30%, 그리고 최근 12개월 언급 논문 합 ≥ 200. 처음 만족한 달. 음수 shift 없이 누적합만.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(total > 0, n_docs / np.where(total > 0, total, 1), 0.0)
    cr = np.concatenate([np.zeros((r.shape[0], 1)), np.cumsum(r, axis=1)], axis=1)
    cn = np.concatenate([np.zeros((n_docs.shape[0], 1)), np.cumsum(n_docs, axis=1)], axis=1)
    lag = 12 * years
    out = np.full(r.shape[0], -1, dtype=int)
    need = (1 + min_cagr) ** years
    for m in range(lag + window - 1, r.shape[1]):
        r1 = (cr[:, m + 1] - cr[:, m + 1 - window]) / window
        r0 = (cr[:, m + 1 - lag] - cr[:, m + 1 - lag - window]) / window
        men = cn[:, m + 1] - cn[:, m + 1 - window]
        with np.errstate(divide="ignore", invalid="ignore"):
            ok = (r0 > 0) & (r1 >= need * r0) & (men >= min_mentions)
        new = ok & (out < 0)
        out[new] = m
    return out


# ── ⑤ 사업 설명 부분 ─────────────────────────────────────────────────────

_APOS = "['’`]?"
K_START = re.compile(r"item\s*1\s*[\.:\-–—]?\s*business(?!\s*combination)", re.I)
K_END = re.compile(r"item\s*1a\s*[\.:\-–—]?\s*risk\s*factors|item\s*1b\s*[\.:\-–—]?\s*unresolved|item\s*2\s*[\.:\-–—]?\s*(?:description\s*of\s*)?propert", re.I)
Q_START = re.compile(rf"item\s*2\s*[\.:\-–—]?\s*management{_APOS}s?\s*discussion", re.I)
Q_END = re.compile(r"item\s*3\s*[\.:\-–—]?\s*quantitative|item\s*4\s*[\.:\-–—]?\s*controls", re.I)
MIN_SECTION = 2000  # 목차 줄(짧음)을 본문으로 잘못 고르지 않게


def section(text: str, form: str) -> str | None:
    """10-K → Item 1(Business) ~ Item 1A(Risk Factors)/1B/2 앞, 10-Q → Item 2(MD&A) ~ Item 3/4 앞.
    시작 후보마다 그 뒤 첫 끝 표지까지를 재고, 가장 긴 구간(MIN_SECTION자 이상)을 고른다(목차 건너뜀). 못 찾으면 None."""
    f = (form or "").upper()
    if f.startswith("10-K"):
        st, en = K_START, K_END
    elif f.startswith("10-Q"):
        st, en = Q_START, Q_END
    else:
        return None
    best = None
    for m in st.finditer(text):
        e = en.search(text, m.end())
        if e is None:
            continue
        span = (m.start(), e.start())
        if span[1] - span[0] >= MIN_SECTION and (best is None or span[1] - span[0] > best[1] - best[0]):
            best = span
    return text[best[0]:best[1]] if best else None
