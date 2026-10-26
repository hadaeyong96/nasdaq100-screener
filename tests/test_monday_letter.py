"""2026-10-26 월요일 편지 사고 — 월요일(한국) 기준일이 토요일 편지와 같은 미국 금요일이라 중복 방지에 막혀 한 번도 안 나갔다.
주간 편지 요일(chart.weekday)에는 한 번 더 보내고, 편지에 붙는 단계가 실패해도 본문은 나가는지. 네트워크 없음(가짜 requests.post)."""

import pandas as pd

from tests.test_curve_chart import _send_env, _send_summary


def _seed(telegram, tmp_path, *keys):
    conn = telegram.db.connect(tmp_path / "s.db")
    for k in keys:
        telegram.db.record_notified(conn, k, "2026-10-24T01:32:24")
    conn.close()


def _sent(telegram, tmp_path, key):
    conn = telegram.db.connect(tmp_path / "s.db")
    try:
        return telegram.db.has_notified(conn, key)
    finally:
        conn.close()


def test_notify_key_weekly_only_on_chart_day(cfg):
    from notify import telegram

    mon = {"letter_today_kst": pd.Timestamp("2026-10-26")}
    tue = {"letter_today_kst": pd.Timestamp("2026-10-27")}
    assert telegram.notify_key("2026-10-23", mon, cfg) == "2026-10-23:weekly:2026-10-26"
    assert telegram.notify_key("2026-10-23", mon, cfg, group=True) == "2026-10-23:group:weekly:2026-10-26"
    assert telegram.notify_key("2026-10-26", tue, cfg) == "2026-10-26"
    assert telegram.notify_key("2026-10-23", {}, cfg) == "2026-10-23"


def test_monday_letter_sent_although_saturday_sent_same_as_of(tmp_path, monkeypatch, cfg):
    """10/26 재현: 10/23 기준 편지를 토요일에 보냈어도 월요일 편지는 나가고, 같은 월요일 두 번째 실행은 막힌다."""
    telegram, calls = _send_env(monkeypatch, tmp_path, cfg)
    s, rep = _send_summary(tmp_path, "2026-10-26")
    s["as_of"] = pd.Timestamp("2026-10-23")
    _seed(telegram, tmp_path, "2026-10-23", "2026-10-23:group")
    telegram.send_briefing("본문", s, cfg)
    telegram.send_group_briefing("본문", rep, s, cfg)
    assert [m for m, _ in calls] == ["sendMessage", "sendPhoto", "sendDocument"] * 2
    assert _sent(telegram, tmp_path, "2026-10-23:weekly:2026-10-26")
    assert _sent(telegram, tmp_path, "2026-10-23:group:weekly:2026-10-26")
    calls.clear()
    telegram.send_briefing("본문", s, cfg)
    telegram.send_group_briefing("본문", rep, s, cfg)
    assert calls == []


def test_sunday_and_weekday_still_once_per_as_of(tmp_path, monkeypatch, cfg):
    telegram, calls = _send_env(monkeypatch, tmp_path, cfg)
    s, rep = _send_summary(tmp_path, "2026-10-25")  # 일요일
    s["as_of"] = pd.Timestamp("2026-10-23")
    _seed(telegram, tmp_path, "2026-10-23", "2026-10-23:group")
    telegram.send_briefing("본문", s, cfg)
    telegram.send_group_briefing("본문", rep, s, cfg)
    assert calls == []


def test_letter_body_failure_still_sends_fallback_and_file(tmp_path, monkeypatch, cfg):
    telegram, calls = _send_env(monkeypatch, tmp_path, cfg)

    def boom(*a, **k):
        raise KeyError("x")

    monkeypatch.setattr(telegram, "letter_message", boom)
    texts = []
    real = telegram.requests.post

    def post(url, data=None, files=None, timeout=None):
        if url.endswith("sendMessage"):
            texts.append(data["text"])
        return real(url, data=data, files=files, timeout=timeout)

    monkeypatch.setattr(telegram.requests, "post", post)
    s, _ = _send_summary(tmp_path, "2026-10-27")
    telegram.send_briefing("본문", s, cfg)
    assert [m for m, _ in calls] == ["sendMessage", "sendDocument"]
    assert "편지 본문을 만들지 못했습니다" in texts[0]


def test_chart_photo_error_is_none(cfg):
    from notify import telegram

    assert telegram.chart_photo({"curve_chart": {"personal": object()}, "letter_today_kst": pd.Timestamp("2026-10-26")}, cfg) is None


def test_safe_step_marks_problem_line_and_continues():
    from engine import daily
    from notify import letter_plain as lp

    s = {"as_of": pd.Timestamp("2026-10-23"), "letter_check": {"lag": 0, "notes": []}}

    def boom():
        raise RuntimeError("x")

    assert daily._safe_step(s, "뉴스", boom) is None
    assert daily._safe_step(s, "그래프", boom) is None
    assert daily._safe_step(s, "내 돈", lambda: {"qty": 1}) == {"qty": 1}
    probs = lp.problems(s, lp.load_phrases(), lp.PERSONAL)
    assert "뉴스를 만들지 못했습니다." in probs and "그래프를 만들지 못했습니다." in probs
    s2 = {"as_of": pd.Timestamp("2026-10-23")}  # letter_check가 아직 없을 때(편지 기본 칸 단계 전)
    daily._safe_step(s2, "시장 지수", boom)
    assert s2["letter_check"]["notes"] == ["시장 지수를 만들지 못했습니다"]


def test_main_wraps_every_letter_step():
    """engine/daily.py main의 편지 단계가 모두 _safe_step 안에서 불리는지(맨몸 호출이 남지 않게)."""
    import inspect

    from engine import daily

    src = inspect.getsource(daily.main)
    for name in ("_attach_market_glance", "_read_holdings", "_attach_dec_tax", "_attach_letter_v3", "_attach_puddle_v31",
                 "_attach_puddle_rebound", "_attach_curve_chart", "_attach_candidate_charts", "_attach_news",
                 "vbooks.run_safely", "_attach_regime_line"):
        for line in src.splitlines():
            if name + "(" in line:
                assert "_safe_step(" in line, line.strip()
