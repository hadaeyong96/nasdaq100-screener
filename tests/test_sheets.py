"""data/sheets.py 테스트. 네트워크 없음 — gspread 클라이언트는 가짜 객체로
주입하고, 인증 함수는 환경변수만 검증한다 (docs/design/live_advisor.md 1단계 1a).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from data import sheets


class _FakeWorksheet:
    def __init__(self, records: list[dict]):
        self._records = records

    def get_all_records(self) -> list[dict]:
        return self._records


class _FakeSpreadsheet:
    def __init__(self, worksheets: dict[str, _FakeWorksheet]):
        self._worksheets = worksheets

    def worksheet(self, name: str) -> _FakeWorksheet:
        return self._worksheets[name]


class _FakeClient:
    def __init__(self, spreadsheet: _FakeSpreadsheet):
        self._spreadsheet = spreadsheet
        self.opened_with: str | None = None

    def open_by_key(self, sheet_id: str) -> _FakeSpreadsheet:
        self.opened_with = sheet_id
        return self._spreadsheet


def _fake_client(plan_records, fills_records) -> _FakeClient:
    return _FakeClient(
        _FakeSpreadsheet(
            {
                sheets.SHEET_PLAN: _FakeWorksheet(plan_records),
                sheets.SHEET_FILLS: _FakeWorksheet(fills_records),
            }
        )
    )


# ---------- parse_plan_records ----------


def test_parse_plan_records_korean_columns():
    df, errors = sheets.parse_plan_records(
        [{"티커": "NVDA", "계획금액": "3000000", "메모": "1차 진입 대기"}]
    )
    assert errors == []
    assert len(df) == 1
    row = df.iloc[0]
    assert row["ticker"] == "NVDA"
    assert row["budget_krw"] == 3_000_000.0
    assert row["memo"] == "1차 진입 대기"


def test_parse_plan_records_english_columns():
    df, errors = sheets.parse_plan_records([{"ticker": "avgo", "budget_krw": 1500000, "memo": ""}])
    assert errors == []
    assert df.iloc[0]["ticker"] == "AVGO"
    assert df.iloc[0]["budget_krw"] == 1_500_000.0


def test_parse_plan_records_accepts_comma_formatted_amount():
    df, errors = sheets.parse_plan_records([{"티커": "META", "계획금액": "3,000,000", "메모": ""}])
    assert errors == []
    assert df.iloc[0]["budget_krw"] == 3_000_000.0


def test_parse_plan_records_skips_blank_rows():
    df, errors = sheets.parse_plan_records([{"티커": "", "계획금액": "", "메모": ""}])
    assert df.empty
    assert errors == []


def test_parse_plan_records_reports_missing_ticker():
    df, errors = sheets.parse_plan_records([{"티커": "", "계획금액": "1000000", "메모": ""}])
    assert df.empty
    assert len(errors) == 1
    assert "1번째 줄" in errors[0]
    assert "티커가 비어 있음" in errors[0]


def test_parse_plan_records_reports_invalid_budget():
    df, errors = sheets.parse_plan_records([{"티커": "NVDA", "계획금액": "많이", "메모": ""}])
    assert df.empty
    assert "계획금액이 올바르지 않음" in errors[0]


def test_parse_plan_records_reports_negative_budget():
    df, errors = sheets.parse_plan_records([{"티커": "NVDA", "계획금액": "-100", "메모": ""}])
    assert df.empty
    assert "계획금액이 올바르지 않음" in errors[0]


def test_parse_plan_records_valid_and_invalid_rows_mixed_keeps_line_numbers():
    df, errors = sheets.parse_plan_records(
        [
            {"티커": "NVDA", "계획금액": "1000000", "메모": ""},
            {"티커": "", "계획금액": "500000", "메모": ""},
            {"티커": "AVGO", "계획금액": "2000000", "메모": "2차"},
        ]
    )
    assert len(df) == 2
    assert list(df["ticker"]) == ["NVDA", "AVGO"]
    assert len(errors) == 1
    assert "2번째 줄" in errors[0]


# ---------- read_sheets (fake client, no network) ----------


def test_read_sheets_combines_plan_and_fills(monkeypatch):
    monkeypatch.setenv("GOOGLE_SHEETS_ID", "sheet-123")
    client = _fake_client(
        plan_records=[{"티커": "NVDA", "계획금액": "3000000", "메모": ""}],
        fills_records=[
            {"날짜": "2026-09-10", "종목": "NVDA", "차수": "1차", "매수매도": "매수", "체결가": "180.25", "수량": "55"}
        ],
    )
    result = sheets.read_sheets(client=client)

    assert client.opened_with == "sheet-123"
    assert len(result.plan_df) == 1
    assert result.plan_df.iloc[0]["ticker"] == "NVDA"
    assert result.plan_errors == []

    assert len(result.fills.df) == 1
    assert result.fills.df.iloc[0]["ticker"] == "NVDA"
    assert result.fills.errors == []

    assert result.read_at_kst is not None
    assert str(result.read_at_kst.tzinfo) != "None"


def test_read_sheets_uses_sheet_id_env_var(monkeypatch):
    monkeypatch.setenv("GOOGLE_SHEETS_ID", "sheet-123")
    client = _fake_client(plan_records=[], fills_records=[])
    sheets.read_sheets(client=client)
    assert client.opened_with == "sheet-123"


def test_read_sheets_raises_without_sheet_id_env_var(monkeypatch):
    monkeypatch.delenv("GOOGLE_SHEETS_ID", raising=False)
    client = _fake_client(plan_records=[], fills_records=[])
    with pytest.raises(sheets.SheetsConfigError):
        sheets.read_sheets(client=client)


def test_read_sheets_empty_tabs_return_empty_frames(monkeypatch):
    monkeypatch.setenv("GOOGLE_SHEETS_ID", "sheet-123")
    client = _fake_client(plan_records=[], fills_records=[])
    result = sheets.read_sheets(client=client)
    assert result.plan_df.empty
    assert result.fills.df.empty


def test_read_sheets_propagates_plan_and_fills_errors(monkeypatch):
    monkeypatch.setenv("GOOGLE_SHEETS_ID", "sheet-123")
    client = _fake_client(
        plan_records=[{"티커": "", "계획금액": "1000000", "메모": ""}],
        fills_records=[
            {"날짜": "2026-09-10", "종목": "NVDA", "차수": "모름", "매수매도": "매수", "체결가": "180.25", "수량": "55"}
        ],
    )
    result = sheets.read_sheets(client=client)
    assert len(result.plan_errors) == 1
    assert len(result.fills.errors) == 1


# ---------- credentials env var handling ----------


def test_load_credentials_info_missing_env_raises(monkeypatch):
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_JSON", raising=False)
    with pytest.raises(sheets.SheetsConfigError):
        sheets.load_credentials_info()


def test_load_credentials_info_from_json_string(monkeypatch):
    info = {"type": "service_account", "client_email": "svc@example.com"}
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", json.dumps(info))
    loaded = sheets.load_credentials_info()
    assert loaded == info


def test_load_credentials_info_from_invalid_json_string_raises(monkeypatch):
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", "{not valid json")
    with pytest.raises(sheets.SheetsConfigError):
        sheets.load_credentials_info()


def test_load_credentials_info_from_file_path(monkeypatch, tmp_path):
    info = {"type": "service_account", "client_email": "svc@example.com"}
    key_path = tmp_path / "key.json"
    key_path.write_text(json.dumps(info), encoding="utf-8")
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", str(key_path))
    loaded = sheets.load_credentials_info()
    assert loaded == info


def test_load_credentials_info_from_missing_file_path_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", str(tmp_path / "no_such_key.json"))
    with pytest.raises(sheets.SheetsConfigError):
        sheets.load_credentials_info()


def test_get_client_builds_credentials_and_authorizes(monkeypatch):
    info = {"type": "service_account", "client_email": "svc@example.com"}
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", json.dumps(info))

    calls = {}

    class _FakeCreds:
        pass

    def fake_from_service_account_info(loaded_info, scopes):
        calls["info"] = loaded_info
        calls["scopes"] = scopes
        return _FakeCreds()

    def fake_authorize(creds):
        calls["creds"] = creds
        return "fake-gspread-client"

    import google.oauth2.service_account as service_account_module
    import gspread as gspread_module

    monkeypatch.setattr(service_account_module.Credentials, "from_service_account_info", staticmethod(fake_from_service_account_info))
    monkeypatch.setattr(gspread_module, "authorize", fake_authorize)

    client = sheets.get_client()
    assert client == "fake-gspread-client"
    assert calls["info"] == info
    assert calls["scopes"] == sheets._SCOPES
