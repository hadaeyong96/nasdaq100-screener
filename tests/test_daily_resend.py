"""engine.daily._resend_last 테스트. 네트워크 없음.

paper 모드는 텔레그램을 아예 보내지 않으므로(notify.telegram의 하드 가드),
--resend는 --mode로 무엇을 받든 항상 실전(live) 기록·파일만 재발송해야 한다.
"""

from __future__ import annotations

from engine import daily
from notify import telegram
from store import db


def test_resend_last_with_paper_mode_arg_still_uses_live_db_and_files(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "state.db")
    monkeypatch.setattr(db, "PAPER_DB_PATH", tmp_path / "paper_state.db")
    monkeypatch.setattr(daily, "OUTPUT_DIR", tmp_path)

    live_conn = db.connect(db.db_path_for_mode("live"))
    db.set_meta(live_conn, "last_processed_date", "2026-09-25")
    live_conn.close()

    paper_conn = db.connect(db.db_path_for_mode("paper"))
    db.set_meta(paper_conn, "last_processed_date", "2026-09-24")  # 일부러 다른 날짜 — 구분 확인용
    paper_conn.close()

    (tmp_path / "report_live_2026-09-25.html").write_text("<html></html>", encoding="utf-8")
    (tmp_path / "telegram_live_2026-09-25.txt").write_text("본문", encoding="utf-8")

    calls: dict = {}

    def _fake_resend_last(report_path, text_path, as_of_str, force_no_send=False):
        calls["report_path"] = report_path
        calls["text_path"] = text_path
        calls["as_of_str"] = as_of_str
        return True

    monkeypatch.setattr(telegram, "resend_last", _fake_resend_last)

    daily._resend_last({}, "paper", force_no_send=False)

    assert calls["as_of_str"] == "2026-09-25"  # paper의 09-24가 아니라 live의 09-25
    assert calls["report_path"] == tmp_path / "report_live_2026-09-25.html"
    assert calls["text_path"] == tmp_path / "telegram_live_2026-09-25.txt"


def test_resend_last_with_live_mode_arg_is_unaffected(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "state.db")
    monkeypatch.setattr(db, "PAPER_DB_PATH", tmp_path / "paper_state.db")
    monkeypatch.setattr(daily, "OUTPUT_DIR", tmp_path)

    live_conn = db.connect(db.db_path_for_mode("live"))
    db.set_meta(live_conn, "last_processed_date", "2026-09-25")
    live_conn.close()

    calls: dict = {}
    monkeypatch.setattr(
        telegram,
        "resend_last",
        lambda report_path, text_path, as_of_str, force_no_send=False: calls.update(as_of_str=as_of_str) or True,
    )

    daily._resend_last({}, "live", force_no_send=False)

    assert calls["as_of_str"] == "2026-09-25"


def test_resend_last_no_live_history_prints_notice_and_does_not_call_telegram(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "state.db")
    monkeypatch.setattr(db, "PAPER_DB_PATH", tmp_path / "paper_state.db")
    monkeypatch.setattr(daily, "OUTPUT_DIR", tmp_path)

    def _boom(*args, **kwargs):
        raise AssertionError("재발송할 실전 기록이 없으면 telegram.resend_last를 부르면 안 된다")

    monkeypatch.setattr(telegram, "resend_last", _boom)

    daily._resend_last({}, "paper", force_no_send=False)

    out = capsys.readouterr().out
    assert "처리된 기준일 기록이 없습니다" in out
