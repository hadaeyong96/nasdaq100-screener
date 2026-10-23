"""아침 편지 v3 뉴스(core/news.py·data/news.py) — 네트워크 없이 가짜 RSS로."""

from __future__ import annotations

import pandas as pd
import pytest

from core import news as cn
from data import news as dn

NOW = pd.Timestamp("2026-10-23 22:00", tz="UTC")
TOPICS = dn.load_topics()


def _rss(items):
    body = "".join(
        f"<item><title>{t} - {src}</title><link>https://n.example/{i}</link>"
        f"<pubDate>{(NOW - pd.Timedelta(hours=h)).strftime('%a, %d %b %Y %H:%M:%S GMT')}</pubDate>"
        f'<source url="https://s.example">{src}</source></item>'
        for i, (t, src, h) in enumerate(items))
    return f'<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>{body}</channel></rss>'


def test_parse_rss_strips_source_suffix_and_rejects_doctype():
    items = cn.parse_rss(_rss([("연준, 기준금리 동결", "연합뉴스", 1), ("국채 금리 5% 하회 - 머니투데이", "머니투데이", 1)]))
    assert items[0]["title"] == "연준, 기준금리 동결" and items[0]["source"] == "연합뉴스"
    assert items[1]["title"] == "국채 금리 5% 하회"
    with pytest.raises(cn.RssError):
        cn.parse_rss('<!DOCTYPE x [<!ENTITY a "b">]><rss/>')


def test_allowed_sources_only():
    assert cn.canonical_source("KBS 뉴스", TOPICS["sources"]) == "KBS"
    assert cn.canonical_source("연합뉴스TV", TOPICS["sources"]) is None
    assert cn.canonical_source("아무개일보", TOPICS["sources"]) is None


def test_grouping_score_and_representative():
    items = cn.parse_rss(_rss([
        ("연준 파월 의장 기준금리 동결 시사", "KBS", 3),
        ("파월 의장, 기준금리 동결 시사…시장 안도", "연합뉴스", 5),
        ("[속보] 파월 \"기준금리 동결\" 시사", "뉴스1", 2),
        ("파월 기준금리 동결 시사", "아무개일보", 6),       # 허용 목록 밖 — 뺀다
        ("유럽 증시 혼조 마감", "MBC", 13),                  # 12시간 넘음 — 최신성 0.7
        ("오래된 기사 하나", "YTN", 30),                     # 24시간 넘음 — 뺀다
    ]))
    sel = cn.select([{"topic": "미국 경제·금리", "weight": 1.5, "items": items}], TOPICS, NOW)
    top = sel["top"]
    assert top[0]["title"] == "파월 의장, 기준금리 동결 시사…시장 안도" and top[0]["source"] == "연합뉴스"  # 대표 = 가장 이른 기사
    assert top[0]["n_sources"] == 3 and top[0]["score"] == pytest.approx(3 * 1.5 * 1.0)
    assert top[1]["title"] == "유럽 증시 혼조 마감" and top[1]["score"] == pytest.approx(1 * 1.5 * 0.7)
    assert all("오래된" not in a["title"] for a in top)


def _scored(scores):
    return [{"title": f"t{i}", "score": s} for i, s in enumerate(scores)]


@pytest.mark.parametrize("n_core,expect", [(0, 3), (3, 3), (12, 10)])
def test_core_count_rule(n_core, expect):
    scored = _scored([6.0] * n_core + [1.0] * 5)
    assert len(cn.pick_top(scored, TOPICS["core_min_score"], TOPICS["core_max"], TOPICS["core_min"])) == expect


def test_topic_sections_exclude_top():
    items = cn.parse_rss(_rss([(t, "연합뉴스", 1) for t in ["관세 협상 타결", "유럽 의회 선거", "일본 엔화 급락", "중국 수출 둔화", "대만 총통 연설", "독일 경기 침체"]]))
    sel = cn.select([{"topic": "국제정세", "weight": 1.0, "items": items}], TOPICS, NOW)
    top_titles = {a["title"] for a in sel["top"]}
    assert len(sel["top"]) == 3 and len(sel["topics"]["국제정세"]) == 3
    assert not top_titles & {a["title"] for a in sel["topics"]["국제정세"]}


def test_fetch_failures_reported():
    def get(url):
        raise ConnectionError("no net")

    res = dn.letter_news(NOW, TOPICS, get=get, pause=0)
    assert res["status"] == "fail" and len(res["failed"]) == len(dn.queries(TOPICS))
    calls = []

    def half(url):
        calls.append(url)
        if len(calls) % 2:
            raise ConnectionError("no net")
        return _rss([("연준 기준금리 동결 시사", "연합뉴스", 1)])

    res = dn.letter_news(NOW, TOPICS, get=half, pause=0)
    assert res["status"] == "partial" and res["top"] and res["failed"]
    assert "hl=ko&gl=KR&ceid=KR:ko" in calls[0]


def test_shorten_title():
    assert cn.shorten("가" * 61) == "가" * 59 + "…" and cn.shorten("가" * 60) == "가" * 60
