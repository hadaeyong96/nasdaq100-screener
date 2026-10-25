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


def search_query(q: str, exclude=()) -> str:
    """RSS 검색어: 검색어 + 공통 제외어(-단어) + 최근 24시간(when:1d)."""
    neg = " ".join(f"-{w}" for w in exclude or [] if str(w).strip())
    return " ".join(x for x in (q, neg, "when:1d") if x)


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
    for i, q in enumerate(queries(topics_cfg)):
        if i and pause:
            time.sleep(pause)
        try:
            items = cnews.parse_rss(get(RSS_URL.format(q=quote(search_query(q["q"], exclude)))))
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
