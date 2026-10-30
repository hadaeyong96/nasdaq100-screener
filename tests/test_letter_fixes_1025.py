"""2026-10-25 점검 묶음 — 반등 확인 장부의 v3.1 비교값, 뉴스 제외어, 주의 한 줄, 가상 실험 장부별 한 줄, 폭락장 첫 줄.
네트워크 없음(conftest가 막음), 합성 데이터만."""

import pandas as pd
import pytest

from core import news as cnews
from data import news as dnews
from engine import puddle_rebound as eng
from engine import puddle_v31 as e31
from notify import letter_plain as lp
from tests.test_puddle_rebound import _history_world, _row as _rb_row, _run

PH = lp.load_phrases()


# ── 3. v3.1 비교값 ───────────────────────────────────────────────────────────


def test_v31_compare_same_start_reads_value_as_is():
    """10/23 사고: v3.1 탭 99,900,000인데 반등 확인 탭 v3.1 열이 100,000,000 — 같은 시작일이면 그대로 읽는다."""
    v31 = {"2026-10-23": 99_900_000.0, "2026-10-26": 100_500_000.0}
    assert eng._v31_rebased(v31, "2026-10-23", "2026-10-23") == 99_900_000.0
    assert eng._v31_rebased(v31, "2026-10-23", "2026-10-26") == 100_500_000.0
    assert eng._v31_rebased(v31, "2026-10-23", "2026-10-27") is None


def test_v31_compare_different_start_rebases_to_1e8():
    """시작일이 다르면 사전 등록대로 이 장부 시작일 v3.1 값을 1억으로 맞춘다."""
    v31 = {"2026-10-23": 99_900_000.0, "2026-10-26": 101_000_000.0, "2026-10-27": 102_010_000.0}
    assert eng._v31_rebased(v31, "2026-10-26", "2026-10-27") == pytest.approx(1.01e8)
    assert eng._v31_rebased(v31, "2026-10-28", "2026-10-27") is None  # 시작일 기록 없음


def test_refresh_fixes_stored_row_and_sheet_pp():
    stored = [{**_rb_row(date="2026-10-23", value=99_900_000.0, qqq=99_900_000.0, v31=100_000_000.0)}]
    fixed = eng.refresh_v31(stored, {"2026-10-23": 99_900_000.0}, "2026-10-23")
    assert len(fixed) == 1 and fixed[0]["v31"] == 99_900_000.0
    row = eng.sheet_row(fixed[0])
    assert row[3] == 99_900_000 and row[5] == 0  # v3.1(원), v3.1 대비(%p)
    assert eng.refresh_v31(fixed, {"2026-10-23": 99_900_000.0}, "2026-10-23") == []  # 이미 맞으면 고칠 줄 없음


def test_already_run_rewrites_wrong_v31_row(tmp_path, monkeypatch):
    """다음 실행(새 거래일이 없어 already여도)에서 저장된 10/23 줄의 v3.1 값을 고쳐 DB·시트에 다시 쓴다."""
    cal, data = _history_world()
    start = str(cal[-17].date())
    r1 = _run(tmp_path, start, monkeypatch)  # v3.1 기록 없이 먼저 → v31 None으로 저장
    assert r1["rows"][0]["v31"] is None
    kw = dict(db_path=tmp_path / "vb.db", history_fn=lambda t, c: data[t], universe_fn=lambda: ["AAA", "BBB"],
              fx_fn=lambda ds: {str(pd.Timestamp(d).date()): 1400.0 for d in ds},
              year_days_fn=lambda y: pd.bdate_range(f"{y}-01-01", f"{y}-12-31"), filings_fn=lambda tk: [], sheet_write=False)
    from core import puddle_v31 as pv

    e31.run({"puddle_v31": {"enabled": True}}, start, p=pv.Params(n_random=5), **kw)
    written = []
    monkeypatch.setattr(eng, "_write_sheet", lambda result, pc, client, sw, rows, book=None: written.extend(rows))
    r2 = _run(tmp_path, start, monkeypatch)
    assert r2["status"] == "already"
    from store import virtual_books as vbstore

    conn = vbstore.connect(tmp_path / "vb.db")
    v31 = eng.v31_values(conn)
    hist = eng.daily_history(conn)
    conn.close()
    assert hist[0]["v31"] == pytest.approx(v31[start]) and [r["date"] for r in written] == [start]


# ── 4. 뉴스 제외어 ───────────────────────────────────────────────────────────


def test_search_query_adds_negative_words():
    q = dnews.search_query("중동 전쟁", ["범죄", "경찰"])
    assert q == "중동 전쟁 -범죄 -경찰 when:1d"
    assert dnews.search_query("연준") == "연준 when:1d"


def test_fetch_puts_exclusions_in_url():
    urls = []
    dnews.fetch({"topics": [{"name": "전쟁·지정학", "queries": ["북한 미사일"]}], "exclude": ["마약"]},
                get=lambda u: urls.append(u) or "<rss><channel></channel></rss>", pause=0)
    from urllib.parse import unquote

    assert "-마약" in unquote(urls[0]) and "북한 미사일" in unquote(urls[0])


def test_select_drops_excluded_titles_and_core_only_from_topics():
    now = pd.Timestamp("2026-10-25 00:00", tz="UTC")
    pub = now - pd.Timedelta(hours=1)
    items = [{"title": "강력범죄와의 전쟁 선포", "source": src, "url": f"u{i}", "published": pub}
             for i, src in enumerate(["연합뉴스", "뉴스1", "KBS", "MBC"])]
    items += [{"title": "이스라엘 이란 휴전 협상 재개", "source": src, "url": f"v{i}", "published": pub}
              for i, src in enumerate(["연합뉴스", "뉴스1"])]
    cfg = {"topics": [{"name": "전쟁·지정학", "weight": 1.5}], "sources": {s: [s] for s in ["연합뉴스", "뉴스1", "KBS", "MBC"]},
           "exclude": ["범죄", "사건사고"]}
    sel = cnews.select([{"topic": "전쟁·지정학", "weight": 1.5, "items": items}], cfg, now)
    titles = [a["title"] for a in sel["top"]] + [a["title"] for v in sel["topics"].values() for a in v]
    assert titles and all("범죄" not in t for t in titles)
    assert all(a["topic"] == "전쟁·지정학" for a in sel["top"])  # 핵심 = 주제 검색 결과에서만
    assert cnews.excluded("보이스 피싱 조직 검거", ["보이스피싱"])  # 띄어쓰기 무시


def test_news_topics_config():
    t = dnews.load_topics()
    war = next(x for x in t["topics"] if x["name"] == "전쟁·지정학")
    assert war["queries"] == ["우크라이나 전쟁", "러시아 우크라이나 휴전", "중동 전쟁", "이스라엘 이란", "대만 해협", "북한 미사일"]
    assert t["exclude"] == ["범죄", "경찰", "마약", "보이스피싱", "살인", "사건사고"]


# ── 5. 문구 ──────────────────────────────────────────────────────────────────


def _risky(*slugs):
    return {"macro_rows": [{"slug": s, "value": 4.5, "badge": {"label": "위험"}} for s in slugs] + [{"slug": "vix", "badge": {"label": "보통"}}]}


def test_warn_is_one_line():
    assert lp.warn_line(_risky("fed", "dgs10"), PH) == "높은 10년 금리와 기준금리 인상은 조심해서 봅니다."
    assert lp.warn_line(_risky("fed"), PH) == "기준금리 인상은 조심해서 봅니다."
    assert lp.warn_line(_risky("t10y2y", "hy", "dgs10"), PH) == \
        "높은 10년 금리, 회사채 위험 신호(하이일드 스프레드)와 장단기 금리차는 조심해서 봅니다."
    assert lp.warn_line(_risky(), PH) is None
    lines = lp.market_lines(_risky("fed", "dgs10", "vix", "hy"), {}, PH, lp.PERSONAL)
    assert sum("조심해서 봅니다" in x for x in lines) == 1


def test_lab_one_line_per_ledger_no_duplicates():
    v31 = e31.letter_lines([{"date": "2026-10-23", "v31": 99.9e6, "qqq": 99.9e6, "v3": 99.9e6, "weights": {},
                             "fills": [], "sells": [], "skipped": [], "failed": []}])
    rb = eng.letter_lines([_rb_row(date="2026-10-23", value=99.9e6, qqq=99.9e6, v31=99.9e6, weights={}, fills=[], skipped=[],
                                   buys=[], waiting=[], opened=[], unqualified=[])])
    s = {"letter_today_kst": pd.Timestamp("2026-10-27"), "puddle_v31": v31, "puddle_rebound": rb}
    for ed in (lp.PERSONAL, lp.GROUP):
        lab = lp.lab_lines(s, {"research_news": {}}, PH, ed)
        assert lab == ["웅덩이 v3.1: QQQ와 같습니다. QQQ만 들고 있습니다.",
                       "웅덩이 반등 확인: QQQ와 같습니다. QQQ만 들고 있습니다."]
        assert len(lab) == len(set(lab))
