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


# ── 실적 발표 뉴스 (실적급등 편지, 2026-11-11) ───────────────────────────────


def test_earnings_news_keeps_titles_with_company_name_only():
    feeds = {
        "앱러빈 실적": _rss([("앱러빈, 3분기 실적 예상 상회…주가 30% 급등", "연합뉴스", 2),
                           ("美 증시 혼조, 기술주 약세", "연합뉴스", 2)]),
        "맥도날드 실적": _rss([("맥도날드 실적 호조에 주가 6%↑", "한국경제", 5), ("맥도날드 실적", "모르는언론", 1),
                             ("맥도날드, 새 메뉴 출시", "뉴스1", 1)]),
    }

    def get(url):
        for q, body in feeds.items():
            if dn.quote(q) in url:
                return body
        raise RuntimeError("없음")

    res = dn.earnings_news(NOW, ["앱러빈", "맥도날드", "앱러빈"], topics_cfg=TOPICS, get=get, pause=0)
    titles = [a["title"] for a in res["top"]]
    assert res["status"] == "ok"
    assert titles == ["앱러빈, 3분기 실적 예상 상회…주가 30% 급등", "맥도날드 실적 호조에 주가 6%↑"]  # 이름 없는 기사·허용 밖 언론사·실적 낱말 없는 기사 제외


def test_earnings_news_empty_and_failed():
    assert dn.earnings_news(NOW, [], topics_cfg=TOPICS, get=lambda u: "", pause=0)["top"] == []

    def boom(url):
        raise RuntimeError("x")

    res = dn.earnings_news(NOW, ["앱러빈"], topics_cfg=TOPICS, get=boom, pause=0)
    assert res["status"] == "fail" and res["failed"] == ["앱러빈 실적"]


def test_earnings_news_alias_searched_separately_and_merged_once():
    # 구글 뉴스 RSS는 OR가 합집합이 아니라서 별칭은 따로 검색한다. 제목의 하이픈("테이크-투")은 무시한다.
    feeds = {
        "테이크투인터랙티브 실적": _rss([("테이크투인터랙티브, 2분기 실적 예상 하회", "연합뉴스", 3)]),
        "테이크투 실적": _rss([("[美특징주]테이크-투, 실적 부진에도 주가↑", "이데일리", 2), ("테이크투, 신작 출시일 공개", "연합뉴스", 2),
                            ("테이크투인터랙티브, 2분기 실적 예상 하회", "연합뉴스", 3)]),
        "알파벳(구글) A주 실적": _rss([]),
        "알파벳 실적": _rss([("구글 모회사 알파벳 3분기 실적 호조", "한국경제", 4)]),
        "구글 실적": _rss([("구글 모회사 알파벳 3분기 실적 호조", "한국경제", 4)]),
        "알파벳(구글) C주 실적": _rss([]),
    }
    asked = []

    def get(url):
        for k, body in feeds.items():
            if url.startswith(dn.RSS_URL.split("{q}")[0] + dn.quote(k + " ")):
                asked.append(k)
                return body
        raise RuntimeError("없음")

    comps = [{"name": "테이크투인터랙티브", "aliases": ["테이크투"]},
             {"name": "알파벳(구글) A주", "aliases": ["알파벳", "구글"]},
             {"name": "알파벳(구글) C주", "aliases": ["알파벳", "구글"]}]
    res = dn.earnings_news(NOW, comps, topics_cfg=TOPICS, get=get, pause=0)
    assert asked == ["테이크투인터랙티브 실적", "테이크투 실적", "알파벳(구글) A주 실적", "알파벳 실적", "구글 실적",
                     "알파벳(구글) C주 실적"]  # 같은 검색어는 한 번
    titles = [a["title"] for a in res["top"]]
    assert sorted(titles) == sorted(["[美특징주]테이크-투, 실적 부진에도 주가↑", "테이크투인터랙티브, 2분기 실적 예상 하회",
                                     "구글 모회사 알파벳 3분기 실적 호조"])  # 같은 기사는 한 번, 실적 낱말 없는 기사 제외


def test_earnings_news_query_cap_counts_alias_searches():
    asked = []

    def get(url):
        asked.append(url)
        return _rss([])

    comps = [{"name": f"회사{i}", "aliases": [f"별칭{i}"]} for i in range(10)]
    dn.earnings_news(NOW, comps, topics_cfg=TOPICS, get=get, pause=0)
    assert len(asked) == dn.EARNINGS_MAX_QUERIES


def test_earnings_news_without_aliases_unchanged():
    feeds = {"앱러빈 실적": _rss([("앱러빈, 3분기 실적 예상 상회", "연합뉴스", 2)])}

    def get(url):
        for k, body in feeds.items():
            if dn.quote(k) in url:
                return body
        raise RuntimeError("없음")

    a = dn.earnings_news(NOW, ["앱러빈"], topics_cfg=TOPICS, get=get, pause=0)
    b = dn.earnings_news(NOW, [{"name": "앱러빈", "aliases": []}], topics_cfg=TOPICS, get=get, pause=0)
    assert [x["title"] for x in a["top"]] == [x["title"] for x in b["top"]] == ["앱러빈, 3분기 실적 예상 상회"]


def test_load_name_aliases_rules(tmp_path):
    from data import universe

    p = tmp_path / "kr.csv"
    p.write_text("ticker,name_kr,alias\nTTWO,테이크투인터랙티브,테이크투\nX,엑스,엑|엑스|엑스코\nY,와이,\n", encoding="utf-8")
    assert universe.load_name_aliases(p) == {"TTWO": ["테이크투"], "X": ["엑스코"]}  # 한 글자·정식 이름과 같은 별칭 버림
    old = tmp_path / "old.csv"
    old.write_text("ticker,name_kr\nTTWO,테이크투인터랙티브\n", encoding="utf-8")
    assert universe.load_name_aliases(old) == {} and universe.load_name_aliases(tmp_path / "없음.csv") == {}


def test_repo_aliases_are_few_and_at_least_two_chars():
    import pandas as pd

    from data import universe

    kr = pd.read_csv(universe.NAME_KR_CSV, dtype=str).fillna("")
    raw = [x for a in kr["alias"] for x in a.split("|") if x]
    assert raw and all(len(x.strip()) >= universe.ALIAS_MIN_LEN for x in raw)
    al = universe.load_name_aliases()
    assert al["TTWO"] == ["테이크투"] and al["GOOGL"] == ["알파벳", "구글"] and len(al) <= 10


def test_earnings_news_days_window_and_by_company():
    # 종목 카드 아래 뉴스(2026-11-12): when:7d로 검색, 7일 안 기사만, 회사별 최대 per_company개
    feeds = {"앱러빈 실적": _rss([("앱러빈 3분기 실적 호조", "연합뉴스", 2), ("앱러빈 매출 전망 상향", "한국경제", 100),
                               ("앱러빈 분기 이익 급증", "이데일리", 150), ("앱러빈 가이던스 발표", "뉴스1", 160),
                               ("앱러빈 실적 예고", "연합뉴스", 200)])}
    urls = []

    def get(url):
        urls.append(url)
        return feeds["앱러빈 실적"]

    res = dn.earnings_news(NOW, ["앱러빈"], topics_cfg=TOPICS, get=get, pause=0, days=7, per_company=3)
    assert dn.quote("when:7d") in urls[0]
    got = [a["title"] for a in res["by_company"]["앱러빈"]]
    assert got[0] == "앱러빈 3분기 실적 호조" and len(got) == 3 and "앱러빈 실적 예고" not in got  # 200시간 = 7일 넘음
    one_day = dn.earnings_news(NOW, ["앱러빈"], topics_cfg=TOPICS, get=get, pause=0)
    assert dn.quote("when:1d") in urls[-1] and [a["title"] for a in one_day["top"]] == ["앱러빈 3분기 실적 호조"]  # 기본은 그대로 24시간


def test_recency_extra_window_default_unchanged():
    from core import news as cn

    assert cn.recency(30) == 0.0 and cn.recency(30, 168) == 0.4 and cn.recency(10, 168) == 1.0 and cn.recency(200, 168) == 0.0
