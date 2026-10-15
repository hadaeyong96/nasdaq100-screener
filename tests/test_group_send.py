"""단체방도 개인 채팅과 같은 보고서·첨부 설명 (작업 7, 2026-10-15) — 네트워크 없음.

같은 파일·같은 설명, 발송 기록은 각각(개인 "YYYY-MM-DD" / 단체방 "YYYY-MM-DD:group"), 한쪽이 실패해도 다른 쪽은 나감,
오류 알림은 개인 채팅에만, 공개용 보고서 경로는 없음.
"""

from __future__ import annotations

import pandas as pd
import pytest

from engine import daily
from notify import briefing, report_html, telegram
from store import db


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "personal")
    monkeypatch.setenv("TELEGRAM_GROUP_CHAT_ID", "group")
    monkeypatch.setattr(telegram, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "data" / "state.db")
    report = tmp_path / "report_live_2026-10-14.html"
    report.write_text("<html>개인 보고서</html>", encoding="utf-8")
    summary = {"mode": "live", "mode_label": "실전", "as_of": pd.Timestamp("2026-10-14"), "report_path": report,
               "regime_line": "오늘 국면: QQQ 200일선 위", "buy_groups": {"b1": [{"ticker": "NVDA"}]}, "buy_count": 1,
               "sell_rows": [], "rebuilt_from": None}
    return tmp_path, report, summary


def _capture(monkeypatch, fail_for=None):
    sent = []

    def doc(token, chat_id, path, filename=None, caption=None):
        sent.append({"chat": chat_id, "path": path, "filename": filename, "caption": caption})
        return chat_id != fail_for

    monkeypatch.setattr(telegram, "_send_document", doc)
    monkeypatch.setattr(telegram, "_send_document_checked",
                        lambda token, chat_id, path, filename=None, caption=None:
                        (doc(token, chat_id, path, filename, caption), None, None if chat_id != fail_for else "boom"))
    monkeypatch.setattr(telegram, "_send_text", lambda *a, **k: pytest.fail("본문 글을 따로 보내면 안 된다"))
    return sent


def test_group_gets_same_file_and_same_caption(setup, monkeypatch, cfg):
    _, report, summary = setup
    sent = _capture(monkeypatch)
    telegram.send_briefing("본문", summary, cfg)
    res = telegram.send_group_briefing("본문", report, summary, cfg)
    assert res["ok"] and [s["chat"] for s in sent] == ["personal", "group"]
    p, g = sent
    assert p["path"] == g["path"] == report and p["filename"] == g["filename"] == "철수의 아침 편지_2026-10-14.html"
    assert p["caption"] == g["caption"] == "철수의 아침 편지 · 10/14(수)\n오늘 국면: QQQ 200일선 위\n매수 1(NVDA) · 매도 0"
    conn = db.connect(db.db_path_for_mode("live"))
    assert db.has_notified(conn, "2026-10-14") and db.has_notified(conn, "2026-10-14:group")
    conn.close()


def test_each_side_has_own_dedup_record(setup, monkeypatch, cfg):
    _, report, summary = setup
    sent = _capture(monkeypatch)
    telegram.send_group_briefing("본문", report, summary, cfg)
    telegram.send_group_briefing("본문", report, summary, cfg)  # 같은 날 다시 — 단체방은 건너뜀
    telegram.send_briefing("본문", summary, cfg)  # 개인은 아직 안 보냈으니 나간다
    assert [s["chat"] for s in sent] == ["group", "personal"]


def test_group_failure_does_not_block_personal_and_alerts_personal_only(setup, monkeypatch, cfg):
    _, report, summary = setup
    sent = _capture(monkeypatch, fail_for="group")
    telegram.send_briefing("본문", summary, cfg)
    res = telegram.send_group_briefing("본문", report, summary, cfg)
    assert not res["ok"] and telegram.group_send_warning_line(res).startswith("단체방 발송 실패")
    conn = db.connect(db.db_path_for_mode("live"))
    assert db.has_notified(conn, "2026-10-14") and not db.has_notified(conn, "2026-10-14:group")
    conn.close()
    assert "⚠" not in telegram.group_send_warning_line(res)


def test_personal_failure_does_not_block_group(setup, monkeypatch, cfg):
    _, report, summary = setup
    sent = _capture(monkeypatch, fail_for="personal")
    telegram.send_briefing("본문", summary, cfg)
    res = telegram.send_group_briefing("본문", report, summary, cfg)
    assert res["ok"] and [s["chat"] for s in sent] == ["personal", "group"]


def test_daily_main_sends_both_even_if_personal_raises(monkeypatch, cfg, tmp_path):
    """engine.daily main: 개인 발송이 예외를 내도 단체방은 나가고, 오류 알림은 개인 채팅(ops)에만."""
    import sys

    summary = daily._empty_run_summary("live", pd.Timestamp("2026-10-14").date(), [])
    summary.update(as_of=pd.Timestamp("2026-10-14"), report_path=tmp_path / "r.html", stale=False, skipped=False)
    calls = []
    monkeypatch.setattr(daily, "load_config", lambda: cfg)
    monkeypatch.setattr(daily, "run", lambda *a, **k: summary)
    monkeypatch.setattr(daily.vbooks, "run_safely", lambda *a, **k: None)
    monkeypatch.setattr(daily.vbooks, "send_alerts_safely", lambda *a, **k: [])
    monkeypatch.setattr(daily.briefing, "build_briefing_text", lambda s, c: "본문")

    def boom(*a, **k):
        raise RuntimeError("personal down")

    monkeypatch.setattr(daily.telegram, "send_briefing", boom)
    monkeypatch.setattr(daily.telegram, "send_group_briefing",
                        lambda text, path, s, c, force_no_send=False: calls.append(("group", path)) or {"ok": True})
    monkeypatch.setattr(daily.telegram, "notify_ops_error", lambda m: calls.append(("ops", m)) or True)
    monkeypatch.setattr(sys, "argv", ["daily", "--mode", "live"])
    daily.main()
    assert ("group", tmp_path / "r.html") in calls
    assert any(k == "ops" and "personal down" in m for k, m in calls)


def test_no_public_report_path_left():
    assert not hasattr(report_html, "render_public_report") and not hasattr(briefing, "build_public_briefing_text")
