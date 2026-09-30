"""data/sheets.py 테스트. 네트워크 없음 — gspread 클라이언트는 가짜 객체로
주입하고, 인증 함수는 환경변수만 검증한다 (docs/design/live_advisor.md 1단계 1a).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from data import fx, sheets
from notify import briefing


@pytest.fixture(autouse=True)
def _no_network_fx(monkeypatch):
    """환율 조회는 네트워크 없이 고정 값으로 (캐시 파일도 안 읽는다)."""
    monkeypatch.setattr(fx, "_load_cache", lambda: {})
    monkeypatch.setattr(
        sheets, "_default_fx_provider", lambda start, end: {"2026-09-09": 1390.0, "2026-09-10": 1392.5}
    )


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


# ---------- 실제 구글 시트 헤더 (2026-09-30 Actions 오류 재현) ----------

_REAL_PLAN_RECORDS = [
    {"티커": "ODFL", "계획금액(원)": "3,000,000", "등록일": "2026-09-25", "메모": "운송"},
    {"티커": "", "계획금액(원)": "", "등록일": "", "메모": ""},  # 빈 줄
    {"티커": "", "계획금액(원)": "", "등록일": "2026-09-26", "메모": "나중에"},  # 티커·금액 없는 줄
]
_REAL_FILL_HEADERS = ["날짜", "티커", "구분(매수/매도)", "수량", "체결가($)", "환율(원/$)", "수수료($)", "메모"]


def _fill(date, ticker, side, qty, price, fx_rate="", fee="", memo=""):
    return dict(zip(_REAL_FILL_HEADERS, [date, ticker, side, qty, price, fx_rate, fee, memo]))


def test_parse_plan_records_real_sheet_headers():
    df, errors = sheets.parse_plan_records(_REAL_PLAN_RECORDS)
    assert errors == []
    assert list(df["ticker"]) == ["ODFL"]
    assert df.iloc[0]["budget_krw"] == 3_000_000.0
    assert df.iloc[0]["memo"] == "운송"


def test_parse_fill_records_real_sheet_headers_no_header_error():
    result = sheets.parse_fill_records([_fill("2026-09-10", "odfl", "매수", "5", "$190.50", "1,391.2", "0.66")])
    assert result.errors == []
    row = result.df.iloc[0]
    assert row["ticker"] == "ODFL"
    assert row["unit"] == "1"  # 차수 열이 없으면 첫 매수 = 1차
    assert row["side"] == "buy"
    assert row["qty"] == 5
    assert row["price"] == 190.5
    assert row["fx_rate"] == 1391.2
    assert row["fee_usd"] == 0.66


def test_parse_fill_records_missing_required_column_names_korean_header():
    result = sheets.parse_fill_records([{"날짜": "2026-09-10", "티커": "ODFL", "수량": "5"}])
    assert result.df.empty
    assert "헤더 오류" in result.errors[0]
    assert "구분(매수/매도)" in result.errors[0] and "체결가($)" in result.errors[0]


def test_parse_fill_records_skips_blank_and_memo_only_rows():
    result = sheets.parse_fill_records(
        [
            _fill("", "", "", "", ""),
            _fill("", "", "", "", "", memo="메모만"),
            _fill("2026-09-10", "ODFL", "매수", "5", "190"),
        ]
    )
    assert result.errors == []
    assert len(result.df) == 1


def test_auto_units_follow_buy_order_and_reset_after_flat():
    result = sheets.parse_fill_records(
        [
            _fill("2026-09-01", "ODFL", "매수", "1", "100"),
            _fill("2026-09-02", "ODFL", "매수", "2", "110"),
            _fill("2026-09-03", "ODFL", "매수", "6", "120"),
            _fill("2026-09-04", "ODFL", "매수", "3", "130"),  # 3차 이후 → 3차 묶음에 합산
            _fill("2026-09-05", "ODFL", "매도", "12", "140"),  # 전량 매도(3차→2차→1차)
            _fill("2026-09-08", "ODFL", "매수", "4", "150"),  # 0주 뒤 → 다시 1차
        ]
    )
    assert result.errors == []
    df = result.df
    buys = df[df["side"] == "buy"]
    assert list(buys["unit"]) == ["1", "2", "6", "6", "1"]
    merged = buys.iloc[3]
    assert merged["qty"] == 9 and merged["price"] == pytest.approx((6 * 120 + 3 * 130) / 9)
    sells = df[df["side"] == "sell"]
    assert list(zip(sells["unit"], sells["qty"])) == [("6", 9), ("2", 2), ("1", 1)]
    assert sells["amount_usd"].sum() == pytest.approx(12 * 140)


def test_auto_units_oversell_warns():
    result = sheets.parse_fill_records(
        [_fill("2026-09-01", "ODFL", "매수", "2", "100"), _fill("2026-09-02", "ODFL", "매도", "5", "110")]
    )
    assert len(result.errors) == 1 and "3주 많음" in result.errors[0]


def test_read_sheets_fills_blank_fx_and_fee(monkeypatch):
    monkeypatch.setenv("GOOGLE_SHEETS_ID", "sheet-123")
    cfg = {"backtest": {"costs": {"commission_buy_pct": 0.07, "commission_sell_pct": 0.1}}}
    client = _fake_client(
        plan_records=_REAL_PLAN_RECORDS,
        fills_records=[
            _fill("2026-09-10", "ODFL", "매수", "10", "200"),  # 환율·수수료 빈 칸
            _fill("2026-09-11", "ODFL", "매도", "4", "210", "1400", "1.5"),  # 직접 적은 값은 유지
        ],
    )
    result = sheets.read_sheets(client=client, cfg=cfg)
    assert result.plan_errors == []
    assert result.fills.errors == []
    buy, sell = result.fills.df.iloc[0], result.fills.df.iloc[1]
    assert buy["fx_rate"] == 1392.5  # 체결일 종가
    assert buy["fee_usd"] == pytest.approx(10 * 200 * 0.07 / 100)
    assert sell["fx_rate"] == 1400.0 and sell["fee_usd"] == 1.5


def test_read_sheets_fx_uses_previous_close_when_date_missing(monkeypatch):
    monkeypatch.setenv("GOOGLE_SHEETS_ID", "sheet-123")
    client = _fake_client(plan_records=[], fills_records=[_fill("2026-09-12", "ODFL", "매수", "1", "200")])
    result = sheets.read_sheets(client=client)  # 9/12(토) 값 없음 → 9/10 종가
    assert result.fills.df.iloc[0]["fx_rate"] == 1392.5
    assert result.fills.df.iloc[0]["fee_usd"] == 0.0  # cfg 없으면 0%


def test_read_sheets_fx_network_failure_is_reported(monkeypatch):
    monkeypatch.setenv("GOOGLE_SHEETS_ID", "sheet-123")

    def _fail(start, end):
        raise ConnectionError("차단")

    client = _fake_client(plan_records=[], fills_records=[_fill("2026-09-10", "ODFL", "매수", "1", "200")])
    result = sheets.read_sheets(client=client, fx_provider=_fail)
    assert any("환율을 받지 못해" in e for e in result.fills.errors)
    assert len(result.fills.df) == 1  # 보유는 그대로 반영


def test_briefing_input_error_line():
    assert briefing.build_input_error_line([]) is None
    line = briefing.build_input_error_line(["계획 기록 오류: 1번째 줄 - 계획금액이 올바르지 않음('')", "x"])
    assert line.startswith("⚠️ 시트 오류: 계획 기록 오류")
    assert line.endswith("외 1건")
    assert "\n" not in line


# ---------- 2026-09-30 확정 시트 형식 (계획 8열, 체결 7열, 수식 때문에 1000행까지 빈 줄) ----------


def _plan(ticker, budget, reg="", memo=""):
    return {"티커": ticker, "계획금액": budget, "등록일": reg, "1차금액": "", "2차금액": "", "3차금액": "",
            "합계": "", "메모": memo}


def test_plan_tab_final_format_with_formula_blank_rows():
    formula_blank = {"티커": "", "계획금액": "", "등록일": "", "1차금액": 0, "2차금액": 0, "3차금액": 0, "합계": 0, "메모": ""}
    records = [_plan("ODFL", "3,000,000", "2026-09-28", "운송"), _plan("NVDA", 5000000)] + [formula_blank] * 998
    df, errors = sheets.parse_plan_records(records)
    assert errors == []
    assert list(df["ticker"]) == ["ODFL", "NVDA"]
    assert list(df["budget_krw"]) == [3_000_000.0, 5_000_000.0]


def test_fills_tab_final_format_with_unit_column_and_blank_rows():
    cols = ["날짜", "종목", "매수매도", "수량", "체결가", "차수", "메모"]
    blank = dict.fromkeys(cols, "")
    records = [dict(zip(cols, ["2026-09-29", "ODFL", "매수", 4, 178.41, "1차", ""]))] + [blank] * 999
    result = sheets.parse_fill_records(records)
    assert result.errors == []
    assert len(result.df) == 1
    row = result.df.iloc[0]
    assert (row["ticker"], row["unit"], row["side"], row["qty"], row["price"]) == ("ODFL", "1", "buy", 4, 178.41)
