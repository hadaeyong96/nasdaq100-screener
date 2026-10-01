"""계획 탭 자동 기록 테스트 (docs/design/auto_plan.md) — 가짜 시트 클라이언트, 네트워크 없이."""

from __future__ import annotations

import copy
from datetime import date

import pandas as pd
import pytest

from core import auto_plan as ap
from data import sheets

_RUN_DATE = date(2026, 10, 2)
_SETTINGS = {"enabled": True, "default_budget_krw": 3_000_000.0, "max_new_per_day": 5}
_HEADER = ["티커", "계획금액(원)", "등록일", "기준가($)", "환율", "1차(주)", "2차(주)", "3차(주)", "합계(주)", "보유(주)", "남은(주)", "메모"]
_INPUT_HEADERS = {"티커", "계획금액(원)", "등록일", "기준가($)", "메모"}


def _signal(ticker, score, stage="A1", limit=100.0, stop=95.0, label="1차 정찰"):
    return {"ticker": ticker, "stage": stage, "stage_label": label, "limit": limit, "stop": stop, "score": score}


# ── 가짜 시트 ───────────────────────────────────────────────────────────────


class FakeWorksheet:
    """gspread Worksheet 흉내: 1000행까지 수식 칸이 미리 채워진 상태를 재현한다."""

    def __init__(self, title, header, rows, n_rows=1000, fail_on_write=False):
        self.title = title
        self.header = list(header)
        self.values = [list(header)] + [self._pad(r) for r in rows]
        while len(self.values) < n_rows:  # 수식만 있는 빈 줄(입력 칸은 빈 값, 수식 칸은 계산 결과)
            self.values.append(["" if h in _INPUT_HEADERS else "0" for h in self.header])
        self.updates: list[dict] = []
        self.fail_on_write = fail_on_write

    def _pad(self, row):
        if isinstance(row, dict):
            return [str(row.get(h, "0" if h not in _INPUT_HEADERS else "")) for h in self.header]
        return list(row)

    def get_all_values(self):
        return copy.deepcopy(self.values)

    def get_all_records(self):
        return [dict(zip(self.header, r)) for r in self.values[1:]]

    def batch_update(self, data, value_input_option=None):
        if self.fail_on_write:
            raise RuntimeError("403 권한 없음")
        for item in data:
            col_letters = "".join(ch for ch in item["range"] if ch.isalpha())
            row = int("".join(ch for ch in item["range"] if ch.isdigit()))
            col = 0
            for ch in col_letters:
                col = col * 26 + (ord(ch) - 64)
            self.values[row - 1][col - 1] = str(item["values"][0][0])
            self.updates.append({"row": row, "col": col, "header": self.header[col - 1], "value": item["values"][0][0]})


class FakeSpreadsheet:
    def __init__(self, tabs):
        self.tabs = tabs

    def worksheet(self, name):
        return self.tabs[name]


class FakeClient:
    def __init__(self, plan_ws, fills_ws=None):
        self.plan_ws = plan_ws
        self.fills_ws = fills_ws or FakeWorksheet("체결", ["날짜", "티커", "구분(매수/매도)", "수량", "체결가($)", "메모"], [], n_rows=1)
        self.spreadsheet = FakeSpreadsheet({"계획": plan_ws, "체결": self.fills_ws})

    def open_by_key(self, sheet_id):
        return self.spreadsheet

    def all_updates(self):
        return self.plan_ws.updates + self.fills_ws.updates


def _plan_ws(rows=(), header=_HEADER, **kw):
    return FakeWorksheet("계획", header, list(rows), **kw)


# ── core: 추가할 줄 고르기 ─────────────────────────────────────────────────


def test_settings_default_disabled_without_section():
    assert ap.auto_plan_settings({})["enabled"] is False


def test_select_skips_tickers_already_in_plan():
    rows = ap.select_new_plan_rows([_signal("AAPL", 30), _signal("MSFT", 20)], ["aapl"], _SETTINGS, _RUN_DATE)
    assert [r["ticker"] for r in rows] == ["MSFT"]


def test_select_max_per_day_by_score():
    signals = [_signal(t, s) for t, s in [("A", 10), ("B", 50), ("C", 30), ("D", 40), ("E", 20), ("F", 60), ("G", 5)]]
    rows = ap.select_new_plan_rows(signals, [], _SETTINGS, _RUN_DATE)
    assert [r["ticker"] for r in rows] == ["F", "B", "D", "C", "E"]


def test_select_row_contents_and_memo():
    rows = ap.select_new_plan_rows([_signal("NVDA", 20, stage="A2", limit=181.8, stop=170.456, label="2차 확인")], [], _SETTINGS, _RUN_DATE)
    assert rows == [
        {
            "ticker": "NVDA", "budget_krw": 3_000_000.0, "reg_date": "2026-10-02", "ref_price": 181.8,
            "memo": "자동 추가 2026-10-02 · 2차 확인 · 손절 $170.46",
        }
    ]


def test_select_skips_signal_without_stop_and_non_entry_kinds():
    signals = [_signal("A", 10, stop=None), _signal("B", 10, stage="STOP")]
    assert ap.select_new_plan_rows(signals, [], _SETTINGS, _RUN_DATE) == []


def test_merge_plan_rows_keeps_existing_and_adds_new():
    plan = pd.DataFrame({"ticker": ["AAPL"], "budget_krw": [5_000_000.0], "ref_price": [200.0], "memo": ["내 메모"]})
    new = ap.select_new_plan_rows([_signal("AAPL", 9), _signal("MSFT", 8)], [], _SETTINGS, _RUN_DATE)
    merged = ap.merge_plan_rows(plan, new)
    assert merged.set_index("ticker")["budget_krw"].to_dict() == {"AAPL": 5_000_000.0, "MSFT": 3_000_000.0}


# ── core: 만료 메모 ─────────────────────────────────────────────────────────

_DATES = pd.bdate_range("2026-10-01", periods=30)
_AUTO_MEMO = "자동 추가 2026-10-02 · 1차 정찰 · 손절 $95.00"


def _expiry(memo=_AUTO_MEMO, held=0, as_of=_DATES[12], expiry_days=10):
    return ap.select_expiry_memos(
        [{"ticker": "NVDA", "memo": memo}], {"NVDA": held}, {"NVDA": _DATES}, as_of, date(2026, 10, 20), expiry_days, "1차 정찰"
    )


def test_expiry_marks_when_all_conditions_met():
    out = _expiry()
    assert out == [{"ticker": "NVDA", "old_memo": _AUTO_MEMO, "new_memo": _AUTO_MEMO + " · 만료 2026-10-20"}]


def test_expiry_not_before_period():
    # 10-02부터 거래일 9개(10-02~10-14) — 아직 만료 아님
    assert _expiry(as_of=_DATES[9]) == []
    assert len(_expiry(as_of=_DATES[10])) == 1


def test_expiry_not_when_holding():
    assert _expiry(held=3) == []


def test_expiry_not_when_memo_changed_by_user():
    assert _expiry(memo=_AUTO_MEMO + " 계속 지켜봄") == []
    assert _expiry(memo="자동 추가 2026-10-02 · 2차 확인 · 손절 $95.00") == []


def test_expiry_only_once():
    once = _expiry()[0]["new_memo"]
    assert _expiry(memo=once) == []


# ── data/sheets: 계획 탭 쓰기 ────────────────────────────────────────────────


def _existing_rows():
    return [
        {"티커": "AAPL", "계획금액(원)": "5,000,000", "등록일": "2026-09-20", "기준가($)": "201.5", "메모": "내 계획"},
        {"티커": "MSFT", "계획금액(원)": "2,000,000", "등록일": "2026-09-21", "기준가($)": "", "메모": ""},
    ]


def test_write_uses_first_row_with_empty_ticker_in_1000_row_formula_sheet():
    ws = _plan_ws(_existing_rows())
    rows = ap.select_new_plan_rows([_signal("NVDA", 20), _signal("AMD", 10)], [], _SETTINGS, _RUN_DATE)
    result = sheets.write_plan_rows(ws, rows)
    assert [(r["ticker"], r["row"]) for r in result.written] == [("NVDA", 4), ("AMD", 5)]
    assert ws.values[3][0] == "NVDA" and ws.values[4][0] == "AMD"
    assert len(ws.values) == 1000  # 맨 아래에 붙이지 않음


def test_write_never_touches_formula_columns():
    ws = _plan_ws(_existing_rows())
    rows = ap.select_new_plan_rows([_signal("NVDA", 20), _signal("AMD", 10)], [], _SETTINGS, _RUN_DATE)
    sheets.write_plan_rows(ws, rows)
    assert ws.updates
    formula_writes = [u for u in ws.updates if u["header"] not in _INPUT_HEADERS]
    assert formula_writes == []


def test_write_finds_columns_by_header_when_order_changes():
    header = ["메모", "환율", "기준가($)", "1차(주)", "티커", "등록일", "남은(주)", "계획금액(원)"]
    ws = FakeWorksheet("계획", header, [])
    rows = ap.select_new_plan_rows([_signal("NVDA", 20, limit=123.45)], [], _SETTINGS, _RUN_DATE)
    sheets.write_plan_rows(ws, rows)
    written = {u["header"]: u["value"] for u in ws.updates}
    assert written == {
        "티커": "NVDA", "계획금액(원)": 3000000, "등록일": "2026-10-02", "기준가($)": 123.45,
        "메모": "자동 추가 2026-10-02 · 1차 정찰 · 손절 $95.00",
    }


def test_existing_rows_with_user_edits_are_kept():
    ws = _plan_ws(_existing_rows())
    before = copy.deepcopy(ws.values[1:3])
    rows = ap.select_new_plan_rows([_signal("NVDA", 20)], ["AAPL", "MSFT"], _SETTINGS, _RUN_DATE)
    sheets.write_plan_rows(ws, rows)
    assert ws.values[1:3] == before
    assert all(u["row"] >= 4 for u in ws.updates)


def test_write_skips_ticker_present_in_sheet_even_if_invalid_row():
    # 계획금액 오류로 계획 DataFrame에서는 빠진 줄 — 시트 원본 기준으로 다시 쓰지 않는다
    ws = _plan_ws([{"티커": "NVDA", "계획금액(원)": "abc", "메모": ""}])
    rows = ap.select_new_plan_rows([_signal("NVDA", 20)], [], _SETTINGS, _RUN_DATE)
    result = sheets.write_plan_rows(ws, rows)
    assert result.written == [] and result.already_present == ["NVDA"]
    assert ws.updates == []


def test_write_does_not_overwrite_row_with_input_leftovers():
    # 티커는 비었지만 메모가 남은 줄은 건너뛴다
    ws = _plan_ws([{"티커": "", "메모": "나중에 볼 종목"}])
    rows = ap.select_new_plan_rows([_signal("NVDA", 20)], [], _SETTINGS, _RUN_DATE)
    result = sheets.write_plan_rows(ws, rows)
    assert result.written[0]["row"] == 3


def test_write_to_other_tab_raises():
    fills_ws = FakeWorksheet("체결", ["날짜", "티커", "메모"], [], n_rows=1)
    rows = ap.select_new_plan_rows([_signal("NVDA", 20)], [], _SETTINGS, _RUN_DATE)
    with pytest.raises(sheets.SheetsWriteForbiddenError):
        sheets.write_plan_rows(fills_ws, rows)
    with pytest.raises(sheets.SheetsWriteForbiddenError):
        sheets.append_expiry_memos(fills_ws, [{"ticker": "NVDA", "old_memo": "x", "new_memo": "y"}])
    with pytest.raises(sheets.SheetsWriteForbiddenError):
        sheets._plan_worksheet(FakeSpreadsheet({"체결": fills_ws}), "체결")
    assert fills_ws.updates == []


def test_write_missing_input_header_raises():
    ws = FakeWorksheet("계획", ["티커", "계획금액(원)", "메모"], [])
    with pytest.raises(sheets.PlanHeaderError):
        sheets.write_plan_rows(ws, ap.select_new_plan_rows([_signal("NVDA", 20)], [], _SETTINGS, _RUN_DATE))


def test_append_expiry_memo_only_memo_cell_and_only_if_unchanged():
    ws = _plan_ws([{"티커": "NVDA", "계획금액(원)": "3000000", "메모": _AUTO_MEMO}, {"티커": "AMD", "메모": "사용자 메모"}])
    done = sheets.append_expiry_memos(
        ws,
        [
            {"ticker": "NVDA", "old_memo": _AUTO_MEMO, "new_memo": _AUTO_MEMO + " · 만료 2026-10-20"},
            {"ticker": "AMD", "old_memo": "자동 추가 2026-10-02 · 1차 정찰 · 손절 $1.00", "new_memo": "x"},  # 그새 바뀜
        ],
    )
    assert done == ["NVDA"]
    assert ws.updates == [{"row": 2, "col": 12, "header": "메모", "value": _AUTO_MEMO + " · 만료 2026-10-20"}]


def test_scopes_allow_sheet_write_but_drive_stays_readonly():
    assert "https://www.googleapis.com/auth/spreadsheets" in sheets._SCOPES
    assert "https://www.googleapis.com/auth/spreadsheets.readonly" not in sheets._SCOPES
    assert "https://www.googleapis.com/auth/drive.readonly" in sheets._SCOPES


# ── engine: run() 통합 ─────────────────────────────────────────────────────


def _patch_run_with_signal(monkeypatch, tmp_path, score=20):
    """tests/test_live_advisor.py의 run() 가짜 IO에 오늘 TEST 종목 A1 신호를 하나 끼워 넣는다."""
    from tests.test_live_advisor import _patch_run_io

    engine_daily = _patch_run_io(monkeypatch, tmp_path)
    monkeypatch.setenv("GOOGLE_SHEETS_ID", "sheet-123")
    original = engine_daily.simulate_since

    def _with_signal(indicator_map, *args, **kwargs):
        sim = original(indicator_map, *args, **kwargs)
        last = indicator_map["TEST"].index[-1]
        close = float(indicator_map["TEST"].loc[last, "close"])
        sim["today_events"].append({"date": last, "ticker": "TEST", "kind": "A1", "unit": "1", "price": close, "score": score})
        sim["states"]["TEST"]["stop"] = round(close * 0.9, 2)
        return sim

    monkeypatch.setattr(engine_daily, "simulate_since", _with_signal)
    return engine_daily


def _cfg_with(cfg, **auto):
    out = copy.deepcopy(cfg)
    out.setdefault("live", {})["auto_plan"] = {**out.get("live", {}).get("auto_plan", {}), **auto}
    return out


def _test_buy_row(summary):
    return next(r for rows in summary["buy_groups"].values() for r in rows if r["ticker"] == "TEST")


def test_run_live_writes_plan_and_sizes_same_run(monkeypatch, tmp_path, cfg):
    engine_daily = _patch_run_with_signal(monkeypatch, tmp_path)
    client = FakeClient(_plan_ws(_existing_rows()))
    summary = engine_daily.run(_cfg_with(cfg, enabled=True), "live", do_replay=False, dry_run=False, sheets_client=client)
    assert summary["auto_plan"]["written"] == ["TEST"]
    assert client.plan_ws.values[3][0] == "TEST"
    assert summary["plan_by_ticker"]["TEST"] == 3_000_000.0
    assert _test_buy_row(summary)["qty"] > 0
    assert client.fills_ws.updates == []


def test_run_live_write_failure_warns_and_still_sizes(monkeypatch, tmp_path, cfg):
    engine_daily = _patch_run_with_signal(monkeypatch, tmp_path)
    client = FakeClient(_plan_ws(_existing_rows(), fail_on_write=True))
    summary = engine_daily.run(_cfg_with(cfg, enabled=True), "live", do_replay=False, dry_run=False, sheets_client=client)
    assert any(w.startswith("계획 자동 기록 실패:") for w in summary["warnings"])
    assert summary["report_path"]  # 보고서(브리핑 첨부)는 그대로 만들어짐
    assert summary["plan_by_ticker"]["TEST"] == 3_000_000.0
    assert _test_buy_row(summary)["qty"] > 0


def test_run_live_disabled_writes_nothing_and_matches_previous_behavior(monkeypatch, tmp_path, cfg):
    engine_daily = _patch_run_with_signal(monkeypatch, tmp_path)

    def _should_not_be_called(*a, **k):
        raise AssertionError("enabled: false면 apply_auto_plan을 부르면 안 된다")

    monkeypatch.setattr(engine_daily, "apply_auto_plan", _should_not_be_called)
    client = FakeClient(_plan_ws(_existing_rows()))
    summary = engine_daily.run(_cfg_with(cfg, enabled=False), "live", do_replay=False, dry_run=False, sheets_client=client)
    assert client.all_updates() == []
    assert "auto_plan" not in summary
    assert set(summary["plan_by_ticker"]) == {"AAPL", "MSFT"}
    row = _test_buy_row(summary)
    assert row["qty"] == 0 and "계획 없음" in row["note"]


def test_run_paper_never_writes(monkeypatch, tmp_path, cfg):
    from store import db as db_module

    engine_daily = _patch_run_with_signal(monkeypatch, tmp_path)
    monkeypatch.setattr(db_module, "PAPER_DB_PATH", tmp_path / "paper_state.db")
    monkeypatch.setattr(engine_daily, "load_fills", lambda: sheets.FillsResult())
    client = FakeClient(_plan_ws(_existing_rows()))
    summary = engine_daily.run(_cfg_with(cfg, enabled=True), "paper", do_replay=False, dry_run=False, sheets_client=client)
    assert client.all_updates() == []
    assert "auto_plan" not in summary


def test_run_live_dry_run_does_not_write_but_merges(monkeypatch, tmp_path, cfg):
    engine_daily = _patch_run_with_signal(monkeypatch, tmp_path)
    client = FakeClient(_plan_ws(_existing_rows()))
    summary = engine_daily.run(_cfg_with(cfg, enabled=True), "live", do_replay=False, dry_run=True, sheets_client=client)
    assert client.all_updates() == []
    assert summary["plan_by_ticker"]["TEST"] == 3_000_000.0


def test_user_edited_auto_row_is_not_rewritten_next_day():
    # 자동 추가된 줄의 계획금액·기준가를 사용자가 바꾼 뒤, 다음 날 같은 종목 신호가 또 나도 그대로 둔다
    ws = _plan_ws([{"티커": "NVDA", "계획금액(원)": "7,000,000", "등록일": "2026-10-02", "기준가($)": "150", "메모": _AUTO_MEMO}])
    plan_df, _ = sheets.parse_plan_records(ws.get_all_records())
    rows = ap.select_new_plan_rows([_signal("NVDA", 99, limit=111.0)], list(plan_df["ticker"]), _SETTINGS, date(2026, 10, 3))
    assert rows == []
    assert sheets.write_plan_rows(ws, ap.select_new_plan_rows([_signal("NVDA", 99)], [], _SETTINGS, date(2026, 10, 3))).written == []
    assert ws.updates == []
    assert ws.values[1][1] == "7,000,000" and ws.values[1][3] == "150"
