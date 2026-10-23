"""아침 편지 뉴스 고르기 (2026-10-24, 아침 편지 v3) — 순수 함수(네트워크·파일·현재 시각 접근 없음).

구글 뉴스 RSS 글(문자열)을 읽어 기사로 바꾸고, 허용 언론사만 남기고, 같은 사건을 묶고, 점수를 매겨
■ 오늘의 핵심(점수 ≥ 3.0 최대 10개, 모자라면 상위 3)과 주제 칸(나머지에서 최대 3씩)을 고른다. 제목은 그대로 쓴다(요약·해석 없음).

점수 = 묶음 안 서로 다른 언론사 수 × 주제 가중치 × 최신성(12시간 안 1.0, 24시간 안 0.7, 그보다 오래되면 뺀다).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import pandas as pd

TOP_N, TOPIC_N, TITLE_MAX = 3, 3, 60
_JOSA = sorted(["으로부터", "에서부터", "에게서", "으로서", "으로써", "이라며", "이라고", "라며", "라고", "에서는", "에서도",
                "에서", "에게", "으로", "까지", "부터", "보다", "처럼", "마저", "조차", "이나", "이며", "하고", "한테",
                "은", "는", "이", "가", "을", "를", "의", "에", "로", "와", "과", "도", "만", "께", "나", "며"],
               key=len, reverse=True)
_TAG = re.compile(r"\[[^\]]*\]|【[^】]*】|\((?:종합|속보|단독|사진|영상|포토)[^)]*\)")


class RssError(ValueError):
    """RSS 글을 읽을 수 없을 때(형식 오류·DOCTYPE 포함)."""


def parse_rss(xml_text: str) -> list[dict]:
    """구글 뉴스 RSS 글 → [{title, source, url, published(UTC Timestamp)}]. 제목 끝 " - 언론사"는 뗀다.
    DOCTYPE(외부 개체)가 들어 있으면 읽지 않는다."""
    if "<!DOCTYPE" in xml_text[:2000].upper() or "<!ENTITY" in xml_text.upper():
        raise RssError("DOCTYPE/ENTITY가 있는 RSS는 읽지 않습니다")
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise RssError(f"RSS 형식 오류: {exc}") from exc
    out = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        src_el = it.find("source")
        source = (src_el.text or "").strip() if src_el is not None else ""
        while source and title.endswith(" - " + source):  # 구글이 붙인 것 + 언론사가 제목에 직접 붙인 것
            title = title[: -len(" - " + source)].strip()
        try:
            pub = pd.Timestamp(parsedate_to_datetime(it.findtext("pubDate") or "")).tz_convert("UTC")
        except (TypeError, ValueError):
            continue
        if title:
            out.append({"title": title, "source": source, "url": (it.findtext("link") or "").strip(), "published": pub})
    return out


def canonical_source(name: str, sources: dict) -> str | None:
    """구글 뉴스 언론사 이름 → 허용 목록 이름(띄어쓰기 무시 정확 일치). 목록에 없으면 None."""
    key = re.sub(r"\s+", "", name or "")
    for canon, aliases in (sources or {}).items():
        if key in {re.sub(r"\s+", "", str(a)) for a in [canon, *(aliases or [])]}:
            return canon
    return None


def keywords(title: str, stopwords=()) -> set[str]:
    """제목 → 핵심어 집합: 꺾쇠 머리표([속보] 등) 빼고, 낱말 끝 조사 떼고, 불용어·한 글자 낱말 뺀다."""
    t = _TAG.sub(" ", title)
    stop = set(stopwords)
    out = set()
    for w in re.findall(r"[0-9A-Za-z가-힣]+", t):
        if re.fullmatch(r"[가-힣]+", w):
            for j in _JOSA:
                if w.endswith(j) and len(w) - len(j) >= 2:
                    w = w[: -len(j)]
                    break
        w = w.lower()
        if len(w) >= 2 and w not in stop:
            out.add(w)
    return out


def same_event(a: set, b: set, overlap: float = 0.5, min_shared: int = 2) -> bool:
    """두 제목 핵심어가 같은 사건인지: 겹친 낱말 수 ≥ min_shared 이고 짧은 쪽 핵심어의 overlap 비율 이상."""
    if not a or not b:
        return False
    shared = len(a & b)
    return shared >= min_shared and shared / min(len(a), len(b)) >= overlap


def recency(age_hours: float) -> float:
    """최신성 가중치: 12시간 안 1.0, 24시간 안 0.7, 그 밖 0."""
    if age_hours <= 12:
        return 1.0
    if age_hours <= 24:
        return 0.7
    return 0.0


def shorten(title: str, n: int = TITLE_MAX) -> str:
    """제목이 n자를 넘으면 n-1자 + "…"."""
    return title if len(title) <= n else title[: n - 1].rstrip() + "…"


def pick_top(scored: list[dict], min_score: float = 3.0, most: int = 10, least: int = TOP_N) -> list[dict]:
    """■ 오늘의 핵심: 점수 순 목록에서 「핵심 뉴스」(점수 ≥ min_score)를 최대 most개.
    핵심 뉴스가 least개 미만이면 점수 상위 least개(빈 날 방지)."""
    core = [s for s in scored if s["score"] >= min_score][:most]
    return core if len(core) >= least else scored[:least]


def select(fetched: list[dict], topics_cfg: dict, now_utc) -> dict:
    """검색어별로 받은 기사 → 편지 뉴스.

    fetched: [{topic, weight, items: parse_rss 결과}] (검색어 하나당 하나, 설정 순서대로)
    now_utc: 기준 시각(UTC). 24시간보다 오래된 기사는 뺀다.
    출력: {"top": [기사], "topics": {주제: [기사]}} — 기사 = {title, source, url, published, score, n_sources, topic}
    """
    now = pd.Timestamp(now_utc)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    sources = topics_cfg.get("sources") or {}
    stop = topics_cfg.get("stopwords") or []
    overlap = float(topics_cfg.get("group_overlap", 0.5))
    min_shared = int(topics_cfg.get("group_min_shared", 2))
    order = [t["name"] for t in topics_cfg.get("topics") or []]

    # 1) 허용 언론사·24시간 안만, 같은 기사(주소 또는 제목+언론사)는 가중치 큰 주제 하나로
    arts: dict[tuple, dict] = {}
    for f in fetched:
        for it in f.get("items") or []:
            canon = canonical_source(it.get("source", ""), sources)
            if canon is None:
                continue
            age = (now - it["published"]).total_seconds() / 3600
            if age < -1 or recency(max(age, 0)) == 0:
                continue
            key = (it.get("url") or "", it["title"], canon)
            cur = arts.get(key)
            if cur is None or float(f["weight"]) > cur["weight"]:
                arts[key] = {**it, "source": canon, "topic": f["topic"], "weight": float(f["weight"]), "age": max(age, 0)}
    items = sorted(arts.values(), key=lambda a: (a["published"], a["title"]))

    # 2) 같은 사건 묶기 — 이른 기사부터, 묶음 대표(가장 이른 기사)의 핵심어와 비교
    groups: list[dict] = []
    for a in items:
        kw = keywords(a["title"], stop)
        for g in groups:
            if same_event(g["kw"], kw, overlap, min_shared):
                g["members"].append(a)
                break
        else:
            groups.append({"kw": kw, "members": [a]})

    # 3) 점수
    scored = []
    for g in groups:
        rep = g["members"][0]
        best = max(g["members"], key=lambda m: (m["weight"], -order.index(m["topic"]) if m["topic"] in order else 0))
        n_src = len({m["source"] for m in g["members"]})
        fresh = recency(min(m["age"] for m in g["members"]))
        scored.append({"title": rep["title"], "source": rep["source"], "url": rep.get("url", ""), "published": rep["published"],
                       "topic": best["topic"], "n_sources": n_src, "score": round(n_src * best["weight"] * fresh, 6)})
    scored.sort(key=lambda s: (-s["score"], -s["n_sources"], -s["published"].value, s["title"]))

    top = pick_top(scored, float(topics_cfg.get("core_min_score", 3.0)), int(topics_cfg.get("core_max", 10)),
                   int(topics_cfg.get("core_min", TOP_N)))
    rest = scored[len(top):]
    by_topic = {name: [s for s in rest if s["topic"] == name][:TOPIC_N] for name in order}
    return {"top": top, "topics": {k: v for k, v in by_topic.items() if v}}
