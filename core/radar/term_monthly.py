"""신기술 레이더 G0 — 달별 2~3단어 표현 표와 급증 등록 규칙 (순수 함수).

규칙 (docs/research/radar_g0_rules.md, 결과 보기 전 고정):
- 글: 제목 + 요약, 소문자. $…$ 수식과 \\명령은 지운다. 구두점(. , ; : ! ? 괄호 따옴표)에서 문장 조각을 끊고, 조각 안에서
  [a-z0-9]가 아닌 글자(하이픈 포함)로 단어를 나눈다. 한 글자 단어와 숫자만인 단어는 끊는 자리로 본다(표현이 그 너머로 이어지지 않음).
- 표현: 이어진 2~3단어. 첫 단어나 끝 단어가 불용어면 버린다(가운데 불용어는 허용 — "internet of things").
  [사용자 결정 2026-10-18] 끝 단어만 간단히 단수로 합친다(networks→network, batteries→battery; -ss·-us·-is·-ics·-ous와 예외 목록은 그대로).
- 논문 하나에 같은 표현이 여러 번 나와도 1편으로 센다(언급 논문 수).
- 급증 등록: 그달 기준 최근 12개월 월별 비율 평균 ≥ 직전 36개월 월별 비율 평균 × 5, 그리고 최근 12개월 언급 논문 합 ≥ 50.
  직전 36개월이 모두 있어야 판정한다(자료 시작 뒤 48번째 달부터). 처음 조건을 만족한 달이 등록 달.
"""

from __future__ import annotations

import re
import zlib

import numpy as np

STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being below between both but by
can could did do does doing down during each either et etc few for from further had has have having he her here hers herself
him himself his how however i ie if in into is it its itself just let may me might more most must my myself neither no nor
not of off on once only or other our ours ourselves out over own per same shall she should since so some such than that the
their theirs them themselves then there therefore these they this those thus through to too under until up upon us use used
using very via was we were what when where whether which while who whom whose why will with within without would yet you
your yours yourself yourselves one two three new show shown shows paper study studies
""".split())
SINGULAR_KEEP = frozenset("series species means news lens bias chaos axis basis thesis analysis gas plus mass class".split())

_MATH = re.compile(r"\$[^$]*\$")
_CMD = re.compile(r"\\[a-zA-Z]+")
_SEG = re.compile(r"[.,;:!?()\[\]{}\"'`]+")
_TOK = re.compile(r"[^a-z0-9]+")


def singular(w: str) -> str:
    """끝 단어 간단 단수화 (사용자 결정 2026-10-18)."""
    if len(w) <= 3 or w in SINGULAR_KEEP or not w.endswith("s"):
        return w
    if w.endswith(("ss", "us", "is", "ics", "ous")):
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    return w[:-1]


def segments(text: str) -> list[list[str]]:
    """글 → 끊기지 않은 단어 줄들."""
    t = _CMD.sub(" ", _MATH.sub(" ", text.lower()))
    out = []
    for seg in _SEG.split(t):
        run: list[str] = []
        for w in _TOK.split(seg):
            if len(w) < 2 or w.isdigit():
                if len(run) >= 2:
                    out.append(run)
                run = []
            else:
                run.append(w)
        if len(run) >= 2:
            out.append(run)
    return out


def doc_terms(title: str, abstract: str) -> set[str]:
    """논문 하나의 2~3단어 표현 집합 (언급 논문 수를 세려고 집합)."""
    out: set[str] = set()
    for run in segments(f"{title} . {abstract}"):
        n = len(run)
        for k in (2, 3):
            for i in range(n - k + 1):
                g = run[i:i + k]
                if g[0] in STOPWORDS or g[-1] in STOPWORDS:
                    continue
                out.add(" ".join(g[:-1] + [singular(g[-1])]))
    return out


def doc_words(title: str, abstract: str, words: frozenset[str]) -> set[str]:
    """참고용: 정해진 한 단어(words)가 논문에 있는지 (감지율 참고 칸 전용, 등록에는 쓰지 않음)."""
    out = set()
    for run in segments(f"{title} . {abstract}"):
        for w in run:
            s = singular(w)
            if s in words:
                out.add(s)
    return out


def normalize_term(term: str) -> str:
    """사람이 쓴 표현을 표의 표현 모양으로 (소문자, 하이픈 → 띄어쓰기, 끝 단어 단수)."""
    ws = [w for w in _TOK.split(term.lower()) if w]
    return " ".join(ws[:-1] + [singular(ws[-1])]) if ws else ""


# ── 근사 세기 (1차: 전체 합 20회 미만 걸러내기) ─────────────────────────────


def hashes(term: str, width_bits: int) -> tuple[int, int]:
    """표현의 두 해시 칸 (crc32, adler32 — 실행마다 같은 값). 카운트-민 스케치 두 줄용."""
    b = term.encode("utf-8")
    m = (1 << width_bits) - 1
    return zlib.crc32(b) & m, zlib.adler32(b) * 2654435761 & m


def sketch_add(sk: np.ndarray, terms, width_bits: int) -> None:
    """sk(2 × 2^width_bits, uint32)에 표현들을 1씩 더한다."""
    if not terms:
        return
    h = np.array([hashes(t, width_bits) for t in terms], dtype=np.int64)
    np.add.at(sk[0], h[:, 0], 1)
    np.add.at(sk[1], h[:, 1], 1)


def sketch_min(sk: np.ndarray, term: str, width_bits: int) -> int:
    """두 줄 중 작은 값 — 실제 횟수 이상(과대 추정만 함)."""
    a, b = hashes(term, width_bits)
    return int(min(sk[0][a], sk[1][b]))


# ── 학술 상투어·급증 등록 ───────────────────────────────────────────────


def boilerplate(ratio: np.ndarray, threshold: float = 0.01, share: float = 0.5) -> np.ndarray:
    """ratio(표현 × 달, 그달 비율) → 상투어 여부. 대부분의 달(= 달의 50% 넘게)에서 비율 > 1%인 표현."""
    return (ratio > threshold).mean(axis=1) > share


def register(n_docs: np.ndarray, total: np.ndarray, recent: int = 12, prior: int = 36, mult: float = 5.0,
             min_mentions: int = 50) -> np.ndarray:
    """급증 등록 달 위치. n_docs(표현 × 달, 언급 논문 수), total(달, 전체 논문 수) → 표현마다 처음 등록된 달 위치(없으면 −1).

    달 m 판정(m ≥ prior + recent − 1): 최근 12개월(m−11 ~ m) 월별 비율 평균 ≥ 직전 36개월(m−47 ~ m−12) 월별 비율 평균 × 5
    그리고 최근 12개월 언급 논문 합 ≥ 50. 직전 평균이 0이면 첫 조건은 참(새로 생긴 표현). 음수 shift 없이 누적합으로만 계산.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(total > 0, n_docs / np.where(total > 0, total, 1), 0.0)
    cr = np.concatenate([np.zeros((r.shape[0], 1)), np.cumsum(r, axis=1)], axis=1)
    cn = np.concatenate([np.zeros((n_docs.shape[0], 1)), np.cumsum(n_docs, axis=1)], axis=1)
    T = r.shape[1]
    out = np.full(r.shape[0], -1, dtype=int)
    for m in range(prior + recent - 1, T):
        rec = (cr[:, m + 1] - cr[:, m + 1 - recent]) / recent
        pri = (cr[:, m + 1 - recent] - cr[:, m + 1 - recent - prior]) / prior
        men = cn[:, m + 1] - cn[:, m + 1 - recent]
        ok = (rec >= mult * pri) & (men >= min_mentions) & (rec > 0)
        new = ok & (out < 0)
        out[new] = m
    return out
