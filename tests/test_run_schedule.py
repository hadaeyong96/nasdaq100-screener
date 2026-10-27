"""2026-10-27 예약 4개(KST 07:37·08:07·08:37·09:07) + 10/22 편지 누락 회귀.

- 같은 기준일에 여러 번 돌아도 편지는 기준일당 한 번(월요일 주간 편지는 하루 한 번 더) — run()의 「이미 처리됨」 단계.
- 10/26 월요일: 발송 키는 고쳤지만 run()이 「이미 처리됨」으로 먼저 끝나 발송 단계까지 못 갔다 → 월요일은 상태 그대로 편지만.
- 데이터 지연 알림은 마지막 예약·수동 실행에서만.
- 10/22: 사용자가 지운 「체결」 탭을 run()이 읽다 실패(exit 1) — sheet_plan_fills: false면 시트 계획·체결을 읽지 않는다.
네트워크 없음.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from data.prices import US_EASTERN
from engine import daily as engine_daily
from store import db
from tests.test_state_persistence import _patch_io, _seeded_holding

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "daily.yml"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_workflow_four_schedules_and_single_concurrency_group(cfg):
    wf = _workflow()
    on = wf.get("on", wf.get(True))  # pyyaml은 on:을 True로 읽는다
    crons = [c["cron"] for c in on["schedule"]]
    assert crons == ["37 22 * * *", "7 23 * * *", "37 23 * * *", "7 0 * * *"]
    assert crons[-1] == cfg["run_schedule"]["final_cron"]
    assert "workflow_dispatch" in on
    assert wf["concurrency"] == {"group": "daily-screener", "cancel-in-progress": False}
    run_step = next(s for s in wf["jobs"]["daily"]["steps"] if s.get("name", "").startswith("Run engine.daily"))
    assert run_step["env"]["SCHEDULE_CRON"] == "${{ github.event.schedule }}"
    assert "job" not in wf["jobs"]["daily"].get("concurrency", {})  # 잡 단위 그룹을 따로 두지 않는다(그룹 하나)


def test_no_schedule_on_the_hour():
    """정각(:00)은 GitHub 예약이 몰려 늦는다(2026-09-30) — 피한다."""
    for c in re.findall(r'cron:\s*"([^"]+)"', WORKFLOW.read_text(encoding="utf-8")):
        assert c.split()[0] != "0", c


def _seed_db(last_processed: str, notified: tuple[str, ...] = ()):
    conn = db.connect(db.db_path_for_mode("live"))
    db.set_meta(conn, "last_processed_date", last_processed)
    db.save_position(conn, _seeded_holding())
    for k in notified:
        db.record_notified(conn, k, "2026-10-24T01:32:24")
    conn.close()


def _db_snapshot():
    conn = db.connect(db.db_path_for_mode("live"))
    try:
        return (db.get_meta(conn, "last_processed_date"),
                conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
                conn.execute("SELECT COUNT(*) FROM events").fetchone()[0],
                db.load_all_positions(conn)["TEST"])
    finally:
        conn.close()


def _run(monkeypatch, tmp_path, cfg, as_of: str, now_et: str):
    _patch_io(monkeypatch, tmp_path, as_of, datetime.fromisoformat(now_et).replace(tzinfo=US_EASTERN))
    return lambda: engine_daily.run(cfg, "live", do_replay=False, dry_run=False)


def test_weekday_second_slot_skips_same_as_of(monkeypatch, tmp_path, cfg):
    """화요일(KST 10/27) 07:37 실행이 10/26을 처리한 뒤 08:07·08:37·09:07 실행은 아무것도 안 한다."""
    go = _run(monkeypatch, tmp_path, cfg, "2026-10-26", "2026-10-26T18:37:00")  # = KST 10/27 07:37
    _seed_db("2026-10-25")
    first = go()
    assert not first.get("skipped") and str(first["as_of"].date()) == "2026-10-26"
    for _ in range(3):
        assert go().get("skipped")


def test_sunday_slots_skip(monkeypatch, tmp_path, cfg):
    """일요일(KST 10/25): 기준일 10/23은 토요일에 처리·발송 — 네 예약 모두 건너뜀."""
    go = _run(monkeypatch, tmp_path, cfg, "2026-10-23", "2026-10-24T20:00:00")  # = KST 10/25 09:00
    _seed_db("2026-10-23", ("2026-10-23", "2026-10-23:group"))
    assert go().get("skipped")


def test_monday_letter_not_skipped_and_writes_nothing(monkeypatch, tmp_path, cfg):
    """10/26 재현: 월요일(KST) 기준일 10/23은 이미 처리됐어도 주간 편지는 만든다 — DB(상태·실행 기록·이벤트)는 그대로."""
    go = _run(monkeypatch, tmp_path, cfg, "2026-10-23", "2026-10-25T18:37:00")  # = KST 10/26 07:37
    _seed_db("2026-10-23", ("2026-10-23", "2026-10-23:group"))
    before = _db_snapshot()
    s = go()
    assert not s.get("skipped") and str(s["as_of"].date()) == "2026-10-23"
    assert Path(s["report_path"]).exists()
    assert s.get("rebuilt_from") is None  # 발송 단계는 주간 키로만 막힌다(정정본 우회 없음)
    assert _db_snapshot() == before


def test_monday_after_weekly_sent_skips(monkeypatch, tmp_path, cfg):
    """월요일 첫 예약이 주간 편지를 보냈으면(발송 기록) 나머지 예약은 건너뜀."""
    go = _run(monkeypatch, tmp_path, cfg, "2026-10-23", "2026-10-25T19:07:00")  # = KST 10/26 08:07
    _seed_db("2026-10-23", ("2026-10-23", "2026-10-23:group", "2026-10-23:weekly:2026-10-26"))
    assert go().get("skipped")


@pytest.mark.parametrize("cron,held", [
    ("37 22 * * *", True), ("7 23 * * *", True), ("37 23 * * *", True),
    ("7 0 * * *", False),  # 마지막 예약
    ("", False),  # 수동 실행·로컬
])
def test_delay_notice_only_on_final_slot(monkeypatch, cfg, cron, held):
    monkeypatch.setenv("SCHEDULE_CRON", cron)
    assert engine_daily._delay_notice_held(cfg) is held


def test_main_holds_delay_notice_before_final_slot():
    """main이 지연 알림 앞에서 _delay_notice_held를 먼저 본다."""
    import inspect

    src = inspect.getsource(engine_daily.main)
    assert src.index("_delay_notice_held(cfg)") < src.index("telegram.send_delay_notice(")


def test_fresh_start_does_not_read_plan_fills_tabs(monkeypatch, tmp_path, cfg):
    """10/22 누락 회귀: 시트 인증이 있어도 sheet_plan_fills: false면 계획·체결 탭(지운 탭)을 읽지 않는다."""
    go = _run(monkeypatch, tmp_path, cfg, "2026-10-26", "2026-10-26T18:37:00")
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", "{}")
    monkeypatch.setenv("GOOGLE_SHEETS_ID", "x")

    def boom(*a, **k):
        raise AssertionError("계획·체결 탭을 읽으면 안 된다")

    monkeypatch.setattr(engine_daily.sheets, "read_sheets", boom)
    _seed_db("2026-10-25")
    assert cfg["live"]["sheet_plan_fills"] is False
    assert not go().get("skipped")
