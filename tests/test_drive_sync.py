"""scripts/drive_sync.py 테스트. 네트워크 없음 — store.drive.download_from_drive/
upload_to_drive를 가짜 함수로 몽키패치한다 (docs/design/live_advisor.md 1단계 1g).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import drive_sync
from store import db


def test_sync_download_skips_without_file_id_env(monkeypatch, capsys):
    monkeypatch.delenv("GOOGLE_DRIVE_STATE_FILE_ID_LIVE", raising=False)
    calls = []
    monkeypatch.setattr(drive_sync.drive, "download_from_drive", lambda *a, **k: calls.append((a, k)))

    did_sync = drive_sync.sync("download", "live")

    assert did_sync is False
    assert calls == []
    assert "건너뜁니다" in capsys.readouterr().out


def test_sync_download_calls_drive_with_correct_file_id_and_path(monkeypatch):
    monkeypatch.setenv("GOOGLE_DRIVE_STATE_FILE_ID_LIVE", "file-abc")
    calls = []
    monkeypatch.setattr(drive_sync.drive, "download_from_drive", lambda file_id, path: calls.append((file_id, path)))

    did_sync = drive_sync.sync("download", "live")

    assert did_sync is True
    assert len(calls) == 1
    file_id, path = calls[0]
    assert file_id == "file-abc"
    assert path == db.db_path_for_mode("live")


def test_sync_upload_calls_drive_with_correct_file_id_and_path(monkeypatch):
    monkeypatch.setenv("GOOGLE_DRIVE_STATE_FILE_ID_PAPER", "file-xyz")
    calls = []
    monkeypatch.setattr(drive_sync.drive, "upload_to_drive", lambda file_id, path: calls.append((file_id, path)))

    did_sync = drive_sync.sync("upload", "paper")

    assert did_sync is True
    assert calls == [("file-xyz", db.db_path_for_mode("paper"))]


def test_sync_live_and_paper_use_different_env_vars(monkeypatch):
    monkeypatch.setenv("GOOGLE_DRIVE_STATE_FILE_ID_LIVE", "file-live")
    monkeypatch.delenv("GOOGLE_DRIVE_STATE_FILE_ID_PAPER", raising=False)
    live_calls = []
    monkeypatch.setattr(drive_sync.drive, "download_from_drive", lambda *a, **k: live_calls.append(a))

    assert drive_sync.sync("download", "live") is True
    assert drive_sync.sync("download", "paper") is False
    assert live_calls == [("file-live", db.db_path_for_mode("live"))]


def test_main_parses_args_and_invokes_sync(monkeypatch):
    monkeypatch.setenv("GOOGLE_DRIVE_STATE_FILE_ID_LIVE", "file-abc")
    calls = []
    monkeypatch.setattr(drive_sync.drive, "download_from_drive", lambda *a, **k: calls.append(a))

    drive_sync.main(["download", "--mode", "live"])

    assert calls == [("file-abc", db.db_path_for_mode("live"))]
