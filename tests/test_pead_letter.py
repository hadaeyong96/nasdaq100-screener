"""notify/pead_letter.py·letter_plain 실적급등 모드·pead_chart·daily 연결 (네트워크 없음)."""

from __future__ import annotations

import pandas as pd

from notify import letter_plain, pead_chart, pead_letter
from scripts import report_preview


def _L(**over):
    L = {"next_day": "2026-11-12", "actions": {"buy": [], "sell": []}, "new": [], "watching": [], "closed": [],
         "scorecards": {}, "earnings": {"rows": [], "failed": 0}, "counts": {"new": 0, "watching": 0}, "chart": None}
    L.update(over)
    return L


def test_todo_lines_tonight_vs_later_and_none():
    buy = [{"ticker": "TTWO", "name": "테이크투인터랙티브"}]
    sell = [{"ticker": "AMAT", "name": "AMAT"}]
    L = _L(actions={"buy": buy, "sell": sell})
    assert pead_letter.todo_lines(L, "2026-11-12") == ["오늘 밤 미국장 시작 때 매수: 테이크투인터랙티브(TTWO)",
                                                        "오늘 밤 장 마감 때 매도: AMAT"]
    later = pead_letter.todo_lines(_L(next_day="2026-11-16", actions={"buy": buy, "sell": []}), "2026-11-14")
    assert later == ["11/16(월) 밤 미국장 시작 때 매수: 테이크투인터랙티브(TTWO)"]
    assert pead_letter.todo_lines(_L(), "2026-11-12") == ["오늘 할 일 없습니다."]


def test_new_lines_have_metrics_dates_and_late_and_holding_notes():
    r = {"ticker": "MCD", "name": "맥도날드", "label": "S&P100", "d0": "2026-11-10", "ret": 0.065, "excess": 0.056,
         "vol_ratio": 2.14, "buy_day": "2026-11-12", "check_day": "2027-02-10", "sell_day": "2027-03-25", "late": True, "skip": None}
    lines = pead_letter.new_lines(_L(new=[r]))
    assert lines == ["맥도날드(MCD) · S&P100 · 늦게 확인했습니다",
                     "반응일 11/10(화) +6.5% · QQQ 대비 +5.6%p · 거래량 2.1배",
                     "매수 11/12(목) 시가 · 중간 점검 2/10(수) · 매도 예정 3/25(목)"]
    held = pead_letter.new_lines(_L(new=[{**r, "late": False, "skip": "보유 중"}]))
    assert held[-1] == "이미 관찰 중인 종목입니다. 새로 담지 않습니다."
    assert pead_letter.new_lines(_L()) == ["새 추천 없습니다."]


def test_watching_earnings_closed_and_scorecards():
    w = [{"ticker": "APP", "name": "앱러빈", "status": "open", "held": 4, "hold": 90, "ret": 0.071, "vs_qqq": 0.064,
          "plan_day": "2027-03-19"}]
    assert pead_letter.watching_lines(_L(watching=w)) == ["앱러빈(APP) · 보유 4/90일 · 매수가 대비 +7.1% · QQQ 대비 +6.4%p · 매도 예정 3/19(금)"]
    e = {"rows": [{"ticker": "NVDA", "name": "엔비디아", "date": "2026-11-18"}, {"ticker": "WMT", "name": "월마트", "date": "2026-11-18"}],
         "failed": 2}
    assert pead_letter.earnings_lines(_L(earnings=e)) == ["11/18(수): 엔비디아(NVDA), 월마트(WMT)", "예정일 확인 실패 2개"]
    sc = {"main": {"n": 3, "mean": 1.234, "win": 66.7, "basis": "cost"}, "compare": {"n": 0}}
    out = pead_letter.closed_lines(_L(scorecards=sc))
    assert out == ["끝난 종목이 아직 없습니다.", "성적표 90일: 3건 · 평균 초과 +1.23% · 승률 67%",
                   "성적표 60일: 아직 끝난 매매가 없습니다."]


def test_group_edition_has_counts_only_and_error_line():
    L = _L(counts={"new": 2, "watching": 3}, new=[{"ticker": "TTWO"}])
    secs = pead_letter.sections({"status": "ok", "letter": L}, "group")
    assert [s["key"] for s in secs] == ["pead_group"]
    text = " ".join(secs[0]["lines"])
    assert "새 추천 2개 · 관찰 중 3개" in text and "TTWO" not in text
    err = pead_letter.sections({"status": "error", "letter": L}, "personal")
    assert err[0]["key"] == "pead_error"
    assert pead_letter.sections(None, "personal") == []


def _preview_summary():
    import yaml

    cfg = yaml.safe_load(open(report_preview.ROOT / "config.yaml", encoding="utf-8"))
    s = report_preview.sample_summary(cfg)
    return report_preview.sample_pead(s), cfg


def test_report_view_pead_mode_order_head_and_fold():
    s, cfg = _preview_summary()
    rv = letter_plain.report_view(s, cfg, "personal")
    assert rv["head"] == "검증 중 · 승률 약 50% · 실제 돈 아님"
    assert [x["key"] for x in rv["sections"]] == ["pead_todo", "pead_new", "pead_watching", "pead_earnings", "pead_news",
                                                   "pead_closed"]
    assert rv["extras"] == [] and rv["monday_chart"] is False and rv["pead"] is True
    assert [x["key"] for x in rv["fold_top"]] == ["glance"]  # 시장·내 돈·일반 뉴스·가상 실험 소식은 뺀다
    glance = rv["fold_top"][0]
    assert glance["lines"][0].startswith("시장 온도 58도")
    assert any("웅덩이 v3.2" in x for x in glance["lines"]) and any("반등 확인 v2" in x for x in glance["lines"])
    news = next(x for x in rv["sections"] if x["key"] == "pead_news")["lines"]
    assert news[0]["source"] == "연합뉴스" and "앱러빈" in news[0]["title"]
    g = letter_plain.report_view(s, cfg, "group")
    assert [x["key"] for x in g["sections"]] == ["pead_group"]
    assert letter_plain.has_todo(s, cfg, "personal") is True


def test_glance_drops_crash_warning(monkeypatch):
    s, cfg = _preview_summary()
    from notify import letter

    monkeypatch.setattr(letter, "crash_warning", lambda q: "급락장입니다")
    ph = letter_plain.load_phrases()
    lines = letter_plain.glance_lines(s, cfg, ph, "personal")
    assert letter_plain.phrase(ph, "todo.crash", letter_plain._day(s)) not in lines


def test_news_lines_states():
    assert pead_letter.news_lines(None) == ["실적 발표 뉴스를 받지 못했습니다."]
    assert pead_letter.news_lines({"status": "ok", "top": []}) == ["최근 24시간 실적 발표 뉴스가 없습니다."]


def test_report_view_without_pead_keeps_old_layout():
    import yaml

    cfg = yaml.safe_load(open(report_preview.ROOT / "config.yaml", encoding="utf-8"))
    s = report_preview.sample_summary(cfg)
    rv = letter_plain.report_view(s, cfg, "personal")
    assert rv["head"] is None and rv["fold_top"] == [] and rv["sections"][0]["key"] == "todo"


def test_rendered_html_personal_has_chart_group_has_no_names(tmp_path, monkeypatch):
    from notify import report_html

    monkeypatch.setattr(report_preview, "OUT", tmp_path)
    s, cfg = _preview_summary()
    p = report_html.render_report(s, cfg, tmp_path, edition="personal").read_text(encoding="utf-8")
    g = report_html.render_report(s, cfg, tmp_path, edition="group").read_text(encoding="utf-8")
    assert "실적급등 추천 종목 비교 그래프" in p and "테이크투인터랙티브(TTWO)" in p and "실적 발표 뉴스" in p
    assert "공포 매수판" not in p and "지수·금리·환율" not in p  # 접기 상세에서도 시장 칸 뺌
    assert "실적급등 추천 종목 비교 그래프" not in g and "TTWO" not in g and "새 추천 2개" in g and "앱러빈" not in g


def test_chart_render_none_without_lines(tmp_path):
    assert pead_chart.render(None, tmp_path / "x.png") is None
    assert pead_chart.render({"lines": [], "qqq": {}}, tmp_path / "x.png") is None


def test_attach_pead_watch_sets_summary_and_survives_chart_error():
    from engine import daily

    summary = {"as_of": pd.Timestamp("2026-11-11"), "letter_today_kst": pd.Timestamp("2026-11-12"), "letter_check": {}}
    fake = {"status": "ok", "letter": _L(), "timing": {"total": 1.0}}

    def bad_chart(*a, **k):
        raise RuntimeError("x")

    daily._attach_pead_watch(summary, {}, run_fn=lambda cfg, as_of, **kw: fake, chart_fn=bad_chart)
    assert summary["pead_watch"]["status"] == "ok" and "pead_chart" not in summary
    s2 = {"as_of": pd.Timestamp("2026-11-11"), "letter_check": {}}
    daily._attach_pead_watch(s2, {}, run_fn=lambda cfg, as_of, **kw: {"status": "error", "letter": None}, chart_fn=lambda *a: None)
    assert s2["pead_watch"]["status"] == "error" and "실적급등 장부 오류" in s2["letter_check"]["notes"]


def test_earnings_news_companies_order_and_attach():
    from engine import daily

    L = _L(new=[{"ticker": "TTWO", "name": "테이크투인터랙티브"}], watching=[{"ticker": "APP", "name": "앱러빈"}],
           earnings={"rows": [{"ticker": "NVDA", "name": "엔비디아", "date": "2026-11-12"},
                              {"ticker": "WMT", "name": "월마트", "date": "2026-11-19"}], "failed": 0})
    assert daily.earnings_news_companies(L, "2026-11-12") == ["테이크투인터랙티브", "앱러빈", "엔비디아"]
    ttwo = {"name": "테이크투인터랙티브", "aliases": ["테이크투"]}
    assert daily.earnings_news_companies(L, "2026-11-12", {"TTWO": ["테이크투"]}) == [ttwo, "앱러빈", "엔비디아"]
    seen = []
    s = {"pead_watch": {"status": "ok", "letter": L}, "letter_today_kst": pd.Timestamp("2026-11-12")}
    daily._attach_earnings_news(s, news_fn=lambda n: seen.append(n) or {"status": "ok", "top": [], "failed": []})
    assert seen == [[ttwo, "앱러빈", "엔비디아"]] and s["earnings_news"]["status"] == "ok"  # 별칭은 name_kr.csv alias 칸
    s2 = {"pead_watch": {"status": "ok", "letter": L}}
    daily._attach_earnings_news(s2, news_fn=lambda n: (_ for _ in ()).throw(RuntimeError("x")))
    assert s2["earnings_news"]["status"] == "fail" and s2["earnings_news"]["failed"][0] == "테이크투인터랙티브"
    s3 = {"pead_watch": {"status": "error", "letter": L}}
    daily._attach_earnings_news(s3)
    assert s3["earnings_news"] is None
