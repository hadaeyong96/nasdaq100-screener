"""store/drive.py 테스트. 네트워크 없음 — 구글 드라이브 API v3 서비스는
가짜 객체로 주입한다 (docs/design/live_advisor.md 1단계 1d).
"""

from __future__ import annotations

import json

import pytest

from store import drive


class _FakeRequest:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class _FakeFilesResource:
    def __init__(self, media_by_file_id: dict[str, bytes] | None = None):
        self._media_by_file_id = media_by_file_id or {}
        self.update_calls: list[tuple[str, object]] = []

    def get_media(self, fileId: str) -> _FakeRequest:
        return _FakeRequest(self._media_by_file_id.get(fileId, b""))

    def update(self, fileId: str, media_body) -> _FakeRequest:
        self.update_calls.append((fileId, media_body))
        return _FakeRequest({"id": fileId})


class _FakeDriveService:
    def __init__(self, media_by_file_id: dict[str, bytes] | None = None):
        self._files = _FakeFilesResource(media_by_file_id)

    def files(self) -> _FakeFilesResource:
        return self._files


# ---------- download_from_drive ----------


def test_download_from_drive_writes_local_file(tmp_path):
    service = _FakeDriveService({"file-abc": b"sqlite-bytes-here"})
    local_path = tmp_path / "state.db"

    drive.download_from_drive("file-abc", local_path, service=service)

    assert local_path.read_bytes() == b"sqlite-bytes-here"


def test_download_from_drive_creates_parent_directories(tmp_path):
    service = _FakeDriveService({"file-abc": b"content"})
    local_path = tmp_path / "nested" / "dir" / "state.db"

    drive.download_from_drive("file-abc", local_path, service=service)

    assert local_path.exists()
    assert local_path.read_bytes() == b"content"


def test_download_from_drive_empty_file_does_not_create_local_file(tmp_path):
    """사용자가 아직 아무 내용도 안 넣은 새 빈 파일 — 첫 실행에는 local_path를
    만들지 않아야 한다 (store.db.connect()가 새 빈 DB를 만든다)."""
    service = _FakeDriveService({"file-abc": b""})
    local_path = tmp_path / "state.db"

    drive.download_from_drive("file-abc", local_path, service=service)

    assert not local_path.exists()


def test_download_from_drive_missing_file_id_treated_as_empty(tmp_path):
    service = _FakeDriveService({})  # file_id가 media_by_file_id에 없음
    local_path = tmp_path / "state.db"

    drive.download_from_drive("unknown-file-id", local_path, service=service)

    assert not local_path.exists()


# ---------- upload_to_drive ----------


def test_upload_to_drive_calls_update_with_correct_file_id(tmp_path):
    service = _FakeDriveService()
    local_path = tmp_path / "state.db"
    local_path.write_bytes(b"updated-sqlite-bytes")

    drive.upload_to_drive("file-abc", local_path, service=service)

    assert len(service.files().update_calls) == 1
    file_id, media_body = service.files().update_calls[0]
    assert file_id == "file-abc"
    assert media_body is not None  # MediaFileUpload 객체 — 새 파일 생성 API는 호출하지 않는다


def test_upload_to_drive_missing_local_file_does_nothing(tmp_path):
    service = _FakeDriveService()
    local_path = tmp_path / "does_not_exist.db"

    drive.upload_to_drive("file-abc", local_path, service=service)

    assert service.files().update_calls == []


def test_upload_to_drive_never_calls_create():
    """서비스 계정은 저장 용량이 없어 새 파일을 만들 수 없다 — upload_to_drive는
    files().create()를 절대 호출하면 안 된다 (docs/design/live_advisor.md 5번)."""

    class _NoCreateFilesResource(_FakeFilesResource):
        def create(self, *args, **kwargs):
            raise AssertionError("upload_to_drive는 create()를 호출하면 안 된다 (빈 파일 덮어쓰기 방식)")

    class _NoCreateService(_FakeDriveService):
        def __init__(self):
            self._files = _NoCreateFilesResource()

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        local_path = Path(tmp) / "state.db"
        local_path.write_bytes(b"bytes")
        drive.upload_to_drive("file-abc", local_path, service=_NoCreateService())


# ---------- get_service / credentials ----------


def test_get_service_builds_credentials_and_calls_build(monkeypatch):
    info = {"type": "service_account", "client_email": "svc@example.com"}
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", json.dumps(info))

    calls = {}

    class _FakeCreds:
        pass

    def fake_from_service_account_info(loaded_info, scopes):
        calls["info"] = loaded_info
        calls["scopes"] = scopes
        return _FakeCreds()

    def fake_build(api, version, credentials, cache_discovery):
        calls["api"] = api
        calls["version"] = version
        calls["credentials"] = credentials
        calls["cache_discovery"] = cache_discovery
        return "fake-drive-service"

    import google.oauth2.service_account as service_account_module
    import googleapiclient.discovery as discovery_module

    monkeypatch.setattr(
        service_account_module.Credentials, "from_service_account_info", staticmethod(fake_from_service_account_info)
    )
    monkeypatch.setattr(discovery_module, "build", fake_build)

    service = drive.get_service()

    assert service == "fake-drive-service"
    assert calls["info"] == info
    assert calls["scopes"] == drive._SCOPES
    assert calls["api"] == "drive"
    assert calls["version"] == "v3"


def test_get_service_raises_without_credentials_env(monkeypatch):
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_JSON", raising=False)
    from data.sheets import SheetsConfigError

    with pytest.raises(SheetsConfigError):
        drive.get_service()
