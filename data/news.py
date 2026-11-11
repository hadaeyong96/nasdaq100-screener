"""아침 편지 뉴스 받기 (2026-10-24, 아침 편지 v3) — 구글 뉴스 한국어 RSS(무료, 키 없음).

검색어마다 https://news.google.com/rss/search?q=<검색어> -<제외어>… when:1d&hl=ko&gl=KR&ceid=KR:ko 를 받는다.
주제·검색어·가중치는 config/news_topics.yaml. 실패한 검색어는 조용히 넘기지 않고 목록으로 돌려준다.
"""

from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import quote

import yaml

from core import news as cnews

ROOT = Path(__file__).resolve().parents[1]
TOPICS_PATH = ROOT / "config" / "news_topics.yaml"
RSS_URL = "https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"
TIMEOUT = 10
PAUSE = 0.3  # 검색어 사이 쉬는 시간(초)


def load_topics(path: Path | None = None) -> dict:
    """config/news_topics.yaml → dict."""
    with open(path or TOPICS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def queries(topics_cfg: dict) -> list[dict]:
    """설정 → [{topic, q, weight}] (설정 순서). 검색어 가중치가 없으면 주제 가중치."""
    out = []
    for t in topics_cfg.get("topics") or []:
        for q in t.get("queries") or []:
            if isinstance(q, dict):
                out.append({"topic": t["name"], "q": str(q["q"]), "weight": float(q.get("weight", t.get("weight", 1.0)))})
            else:
                out.append({"topic": t["name"], "q": str(q), "weight": float(t.get("weight", 1.0))})
    return out


def search_query(q: str, exclude=(), when: str = "1d") -> str:
    """RSS 검색어: 검색어 + 공통 제외어(-단어) + 기간(기본 최근 24시간 when:1d)."""
    neg = " ".join(f"-{w}" for w in exclude or [] if str(w).strip())
    return " ".join(x for x in (q, neg, f"when:{when}") if x)


def _get(url: str) -> str:
    import requests

    r = requests.get(url, timeout=TIMEOUT, headers={"Accept": "application/rss+xml, application/xml"})
    r.raise_for_status()
    return r.text


def fetch(topics_cfg: dict, get=None, pause: float | None = None) -> dict:
    """검색어마다 RSS를 받아 읽는다.

    출력: {"fetched": [{topic, weight, q, items}], "failed": [검색어]} — 하나도 못 받으면 fetched가 빈 목록."""
    get = get or _get
    pause = PAUSE if pause is None else pause
    fetched, failed = [], []
    exclude = topics_cfg.get("exclude") or []
    when = str(topics_cfg.get("when", "1d"))
    for i, q in enumerate(queries(topics_cfg)):
        if i and pause:
            time.sleep(pause)
        try:
            items = cnews.parse_rss(get(RSS_URL.format(q=quote(search_query(q["q"], exclude, when)))))
        except Exception as exc:  # noqa: BLE001 — 검색어 하나 실패가 편지를 막으면 안 된다(목록으로 보고)
            print(f"[news] '{q['q']}' 뉴스를 받지 못했습니다: {type(exc).__name__}")
            failed.append(q["q"])
            continue
        fetched.append({**q, "items": items})
    return {"fetched": fetched, "failed": failed}


def letter_news(now_utc, topics_cfg: dict | None = None, get=None, pause: float | None = None) -> dict:
    """편지용 뉴스: {"status": ok|partial|fail, "top": [...], "topics": {...}, "failed": [...]}."""
    topics_cfg = topics_cfg if topics_cfg is not None else load_topics()
    res = fetch(topics_cfg, get=get, pause=pause)
    if not res["fetched"]:
        return {"status": "fail", "top": [], "topics": {}, "failed": res["failed"]}
    sel = cnews.select(res["fetched"], topics_cfg, now_utc)
    status = "partial" if res["failed"] else "ok"
    if not sel["top"]:
        status = "fail" if res["failed"] else "ok"
    return {"status": status, **sel, "failed": res["failed"]}


EARNINGS_TOP = 6  # 실적 발표 뉴스 최대 줄 수 (2026-11-11)
EARNINGS_MAX_QUERIES = 15  # 검색어(회사) 최대 수 — RSS 요청 수 상한
EARNINGS_PER_COMPANY = 3  # 종목 카드 아래 회사별 기사 수 (2026-11-12)
EARNINGS_WORDS = ("실적", "매출", "이익", "분기", "가이던스", "어닝", "전망")  # 제목에 하나는 있어야 실적 기사로 본다


def _plain(s: str) -> str:
    """제목·이름 비교용: 띄어쓰기·하이픈을 지운다 ("테이크-투" == "테이크투")."""
    return s.replace(" ", "").replace("-", "")


def earnings_news(now_utc, companies: list, topics_cfg: dict | None = None, get=None, pause: float | None = None,
                  top: int = EARNINGS_TOP, days: int = 1, per_company: int = EARNINGS_PER_COMPANY) -> dict:
    """실적급등 편지의 실적 발표 뉴스(2026-11-11): 회사 이름·별칭마다 "{이름} 실적" 검색(최근 24시간, 같은 언론사 허용 목록·제외어),
    제목에 그 회사 이름이나 별칭(띄어쓰기·하이픈 무시)과 실적 낱말(EARNINGS_WORDS)이 함께 있는 기사만 남기고 회사마다 같은 사건을 묶은 뒤,
    점수(언론사 수 × 최신성) 순 최대 top개. 별칭(name_kr.csv alias 칸)은 검색과 제목 필터에만 쓴다.
    구글 뉴스 RSS는 "(A OR B)"가 합집합이 되지 않아(2026-11-11 실측: 별칭에만 걸린 기사 10건 중 0건) 별칭은 따로 검색한다.

    입력: companies — 회사 이름 문자열 또는 {"name", "aliases"}. 검색어는 앞 회사부터 모두 합쳐 최대 EARNINGS_MAX_QUERIES개.
    days — 검색 기간(일, when:{days}d). 24시간이 넘은 기사는 최신성 0.4.
    출력: letter_news와 같은 모양(top만 채움) + by_company {회사 이름: 기사 최대 per_company개(점수 순)} — 종목 카드 아래 뉴스.
    """
    base = topics_cfg if topics_cfg is not None else load_topics()
    by_name: dict[str, list[str]] = {}
    for c in companies:
        name, aliases = (c, []) if isinstance(c, str) else (c.get("name") or "", c.get("aliases") or [])
        name = str(name).strip()
        if name and name not in by_name:
            by_name[name] = list(dict.fromkeys([name, *(a.strip() for a in aliases if a.strip() and a.strip() != name)]))
    company_of: dict[str, str] = {}  # 검색어 → 회사
    for name, terms in by_name.items():
        for t in terms:
            company_of.setdefault(f"{t} 실적", name)
    company_of = dict(list(company_of.items())[:EARNINGS_MAX_QUERIES])
    if not company_of:
        return {"status": "ok", "top": [], "topics": {}, "failed": [], "by_company": {}}
    cfg = {**base, "topics": [{"name": "실적 발표", "weight": 1.0, "queries": list(company_of)}],
           "core_min_score": 0.0, "core_max": max(top, per_company), "core_min": 0,
           "when": f"{int(days)}d", "max_age_hours": 24.0 * int(days)}
    res = fetch(cfg, get=get, pause=pause)
    groups: dict[str, list[dict]] = {}
    for f in res["fetched"]:
        name = company_of[f["q"]]
        keys = [_plain(t) for t in by_name[name]]
        f["items"] = [it for it in f["items"] if any(k in _plain(it["title"]) for k in keys)
                      and any(w in it["title"] for w in EARNINGS_WORDS)]
        groups.setdefault(name, []).append(f)
    if not res["fetched"]:
        return {"status": "fail", "top": [], "topics": {}, "failed": res["failed"], "by_company": {}}
    # 같은 사건 묶기는 회사마다 따로 — 다른 회사 기사가 "실적·주가" 같은 낱말로 한 사건으로 묶이지 않게.
    # 한 회사의 이름·별칭 검색 결과는 함께 넣는다(select가 같은 기사를 하나로 합친다).
    by_company = {name: cnews.select(fs, cfg, now_utc)["top"] for name, fs in groups.items()}
    arts = [a for v in by_company.values() for a in v]
    arts.sort(key=lambda a: (-a["score"], -a["published"].value, a["title"]))
    # 별칭이 같은 회사(알파벳 A주·C주)는 같은 기사가 두 번 잡힌다 — 제목으로 한 번만
    seen: set[str] = set()
    arts = [a for a in arts if not (a["title"] in seen or seen.add(a["title"]))]
    return {"status": "partial" if res["failed"] else "ok", "top": arts[:top], "topics": {}, "failed": res["failed"],
            "by_company": {k: v[:per_company] for k, v in by_company.items() if v}}
