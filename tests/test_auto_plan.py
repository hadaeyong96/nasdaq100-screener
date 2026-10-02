"""계획 탭 자동 기록 v2 테스트 (docs/design/auto_plan.md) — 가짜 시트 클라이언트, 네트워크 없이."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from core import auto_plan as ap
from data import sheets

_HEADER = ["티커", "계획금액(원)", "등록일", "기준가($)", "환율", "1차(주)", "2차(주)", "3차(주)", "합계(주)", "보유(주)", "남은(주)", "메모"]
_INPUT_HEADERS = {"티커", "계획금액(원)", "등록일", "기준가($)", "메모"}
_FILLS_HEADER = ["날짜", "티커", "구분(매수/매도)", "수량", "체결가($)", "환율(원/$)", "차수", "메모"]
_BUDGET = 3_000_000.0


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


class FakePortfolioWorksheet:
    """투자현황 탭 흉내 — A열 항목 이름, B열 값. 쓰기 메서드가 불리면 기록한다."""

    def __init__(self, values):
        self.title = "투자현황"
        self.values = [list(v) for v in values]
        self.updates: list = []

    def get_all_values(self):
        return copy.deepcopy(self.values)

    def batch_update(self, data, value_input_option=None):
        self.updates.append(data)


class FakeSpreadsheet:
    def __init__(self, tabs):
        self.tabs = tabs

    def worksheet(self, name):
        return self.tabs[name]  # 없는 탭은 KeyError(gspread의 WorksheetNotFound 대신)


class FakeClient:
    def __init__(self, plan_ws, fills_ws=None, portfolio_ws=None):
        self.plan_ws = plan_ws
        self.fills_ws = fills_ws or FakeWorksheet("체결", _FILLS_HEADER, [], n_rows=1)
        self.portfolio_ws = portfolio_ws
        tabs = {"계획": plan_ws, "체결": self.fills_ws}
        if portfolio_ws is not None:
            tabs["투자현황"] = portfolio_ws
        self.spreadsheet = FakeSpreadsheet(tabs)

    def open_by_key(self, sheet_id):
        return self.spreadsheet

    def all_updates(self):
        portfolio_updates = self.portfolio_ws.updates if self.portfolio_ws is not None else []
        return self.plan_ws.updates + self.fills_ws.updates + portfolio_updates


def _plan_ws(rows=(), header=_HEADER, **kw):
    return FakeWorksheet("계획", header, list(rows), **kw)


def _fills_ws(rows):
    return FakeWorksheet("체결", _FILLS_HEADER, list(rows), n_rows=len(rows) + 1)


def _fill(date, ticker, qty, price, side="매수", unit="1차", fx="1400"):
    return {"날짜": date, "티커": ticker, "구분(매수/매도)": side, "수량": str(qty), "체결가($)": str(price),
            "환율(원/$)": fx, "차수": unit, "메모": ""}


def _portfolio_ws(total="10,000,000", budget="3,000,000"):
    return FakePortfolioWorksheet([["항목", "값"], ["총 투자금", total], ["종목당 계획금액", budget]])


def _fills_df(rows):
    """[(date, ticker, side, qty, price, fx)] → fills DataFrame (data/fills.py 열 이름)."""
    return pd.DataFrame(
        [{"date": pd.Timestamp(d), "ticker": t, "side": s, "qty": q, "price": p, "fx_rate": fx} for d, t, s, q, p, fx in rows]
    )


def _plan_row(ticker, ref_price=27.55, reg_date="2026-10-01"):
    return {"ticker": ticker, "budget_krw": _BUDGET, "reg_date": reg_date, "ref_price": ref_price,
            "memo": ap.build_first_buy_memo(reg_date)}


# ── data/sheets: 투자현황 탭 읽기 (규칙 A) ─────────────────────────────────────


@pytest.fixture
def _sheet_id(monkeypatch):
    monkeypatch.setenv("GOOGLE_SHEETS_ID", "sheet-123")


def test_portfolio_tab_missing_returns_none_and_warns(_sheet_id):
    portfolio, warnings = sheets.read_portfolio(FakeClient(_plan_ws()))
    assert portfolio == {"total_krw": None, "per_ticker_budget_krw": None}
    assert warnings == ["투자현황 탭에 종목당 계획금액이 없어 기본 수량을 계산하지 않았습니다"]


def test_portfolio_empty_budget_cell_warns(_sheet_id):
    portfolio, warnings = sheets.read_portfolio(FakeClient(_plan_ws(), portfolio_ws=_portfolio_ws(budget="")))
    assert portfolio == {"total_krw": 10_000_000.0, "per_ticker_budget_krw": None}
    assert warnings == [sheets.PORTFOLIO_BUDGET_MISSING_WARNING]


def test_portfolio_comma_numbers_and_labels_with_parens_spaces():
    values = [["총 투자금(원)", "12,345,678"], [" 종목당  계획금액 (원)", "₩3,000,000"]]
    assert sheets.parse_portfolio_values(values) == {"total_krw": 12_345_678.0, "per_ticker_budget_krw": 3_000_000.0}


def test_portfolio_row_order_changed(_sheet_id):
    ws = FakePortfolioWorksheet([["메모", "x"], ["종목당 계획금액", "2,500,000"], [], ["기타", "1"], ["총 투자금", "50000000"]])
    portfolio, warnings = sheets.read_portfolio(FakeClient(_plan_ws(), portfolio_ws=ws))
    assert portfolio == {"total_krw": 50_000_000.0, "per_ticker_budget_krw": 2_500_000.0}
    assert warnings == []


# ── core: 첫 매수 계획 줄 (규칙 D) ─────────────────────────────────────────────


def test_first_buy_row_uses_first_buy_date_and_price():
    fills = _fills_df([
        ("2026-10-05", "CPRT", "buy", 2, 29.10, 1400),
        ("2026-10-01", "CPRT", "buy", 1, 27.55, 1390),
    ])
    rows = ap.select_first_buy_plan_rows(fills, [], _BUDGET)
    assert rows == [{"ticker": "CPRT", "budget_krw": _BUDGET, "reg_date": "2026-10-01", "ref_price": 27.55,
                     "memo": "자동 추가 · 첫 매수 2026-10-01"}]


def test_first_buy_row_skipped_when_plan_exists():
    fills = _fills_df([("2026-10-01", "CPRT", "buy", 1, 27.55, 1400)])
    assert ap.select_first_buy_plan_rows(fills, ["cprt"], _BUDGET) == []


def test_first_buy_row_skipped_when_fully_sold():
    fills = _fills_df([("2026-10-01", "CPRT", "buy", 1, 27.55, 1400), ("2026-10-02", "CPRT", "sell", 1, 28.0, 1400)])
    assert ap.select_first_buy_plan_rows(fills, [], _BUDGET) == []


def test_invested_principal_buy_minus_sell():
    fills = _fills_df([
        ("2026-10-01", "CPRT", "buy", 10, 30.0, 1400),
        ("2026-10-02", "CPRT", "sell", 4, 35.0, 1500),
        ("2026-10-02", "AAPL", "buy", 1, 200.0, float("nan")),
    ])
    principal, missing = ap.invested_principal_krw(fills)
    assert principal == pytest.approx(10 * 30 * 1400 - 4 * 35 * 1500)
    assert missing == 1


def test_cash_shortfall_total_and_row():
    out = ap.cash_shortfall([400_000, 400_000], 500_000)
    assert out == {"total_krw": 800_000.0, "total_exceeds": True, "row_exceeds": [False, False]}
    out = ap.cash_shortfall([600_000, 100_000], 500_000)
    assert out["total_exceeds"] is True and out["row_exceeds"] == [True, False]
    assert ap.cash_shortfall([100_000], 500_000) == {"total_krw": 100_000.0, "total_exceeds": False, "row_exceeds": [False]}


def test_merge_plan_rows_keeps_existing_and_adds_new():
    plan = pd.DataFrame({"ticker": ["AAPL"], "budget_krw": [5_000_000.0], "ref_price": [200.0], "memo": ["내 메모"]})
    merged = ap.merge_plan_rows(plan, [_plan_row("AAPL"), _plan_row("CPRT")])
    assert merged.set_index("ticker")["budget_krw"].to_dict() == {"AAPL": 5_000_000.0, "CPRT": _BUDGET}
    assert merged.set_index("ticker").loc["AAPL", "ref_price"] == 200.0


def test_settings_default_disabled_without_section():
    assert ap.auto_plan_settings({}) == {"enabled": False}


# ── data/sheets: 계획 탭 쓰기 (규칙 F, v1과 같음) ─────────────────────────────


def _existing_rows():
    return [
        {"티커": "AAPL", "계획금액(원)": "5,000,000", "등록일": "2026-09-20", "기준가($)": "201.5", "메모": "내 계획"},
        {"티커": "MSFT", "계획금액(원)": "2,000,000", "등록일": "2026-09-21", "기준가($)": "", "메모": ""},
    ]


def test_write_uses_first_row_with_empty_ticker_in_1000_row_formula_sheet():
    ws = _plan_ws(_existing_rows())
    result = sheets.write_plan_rows(ws, [_plan_row("NVDA"), _plan_row("AMD")])
    assert [(r["ticker"], r["row"]) for r in result.written] == [("NVDA", 4), ("AMD", 5)]
    assert ws.values[3][0] == "NVDA" and ws.values[4][0] == "AMD"
    assert len(ws.values) == 1000  # 맨 아래에 붙이지 않음


def test_write_never_touches_formula_columns():
    ws = _plan_ws(_existing_rows())
    sheets.write_plan_rows(ws, [_plan_row("NVDA"), _plan_row("AMD")])
    assert ws.updates
    formula_writes = [u for u in ws.updates if u["header"] not in _INPUT_HEADERS]
    assert formula_writes == []


def test_write_finds_columns_by_header_when_order_changes():
    header = ["메모", "환율", "기준가($)", "1차(주)", "티커", "등록일", "남은(주)", "계획금액(원)"]
    ws = FakeWorksheet("계획", header, [])
    sheets.write_plan_rows(ws, [_plan_row("CPRT", ref_price=27.55, reg_date="2026-10-01")])
    written = {u["header"]: u["value"] for u in ws.updates}
    assert written == {
        "티커": "CPRT", "계획금액(원)": 3000000, "등록일": "2026-10-01", "기준가($)": 27.55,
        "메모": "자동 추가 · 첫 매수 2026-10-01",
    }


def test_existing_rows_with_user_edits_are_kept():
    ws = _plan_ws(_existing_rows())
    before = copy.deepcopy(ws.values[1:3])
    sheets.write_plan_rows(ws, [_plan_row("NVDA")])
    assert ws.values[1:3] == before
    assert all(u["row"] >= 4 for u in ws.updates)


def test_write_skips_ticker_present_in_sheet_even_if_invalid_row():
    # 계획금액 오류로 계획 DataFrame에서는 빠진 줄 — 시트 원본 기준으로 다시 쓰지 않는다
    ws = _plan_ws([{"티커": "NVDA", "계획금액(원)": "abc", "메모": ""}])
    result = sheets.write_plan_rows(ws, [_plan_row("NVDA")])
    assert result.written == [] and result.already_present == ["NVDA"]
    assert ws.updates == []


def test_write_does_not_overwrite_row_with_input_leftovers():
    # 티커는 비었지만 메모가 남은 줄은 건너뛴다
    ws = _plan_ws([{"티커": "", "메모": "나중에 볼 종목"}])
    result = sheets.write_plan_rows(ws, [_plan_row("NVDA")])
    assert result.written[0]["row"] == 3


def test_write_to_other_tab_raises():
    fills_ws = FakeWorksheet("체결", ["날짜", "티커", "메모"], [], n_rows=1)
    portfolio_ws = _portfolio_ws()
    with pytest.raises(sheets.SheetsWriteForbiddenError):
        sheets.write_plan_rows(fills_ws, [_plan_row("NVDA")])
    with pytest.raises(sheets.SheetsWriteForbiddenError):
        sheets.write_plan_rows(portfolio_ws, [_plan_row("NVDA")])
    with pytest.raises(sheets.SheetsWriteForbiddenError):
        sheets._plan_worksheet(FakeSpreadsheet({"체결": fills_ws}), "체결")
    with pytest.raises(sheets.SheetsWriteForbiddenError):
        sheets._plan_worksheet(FakeSpreadsheet({"투자현황": portfolio_ws}), "투자현황")
    assert fills_ws.updates == [] and portfolio_ws.updates == []


def test_write_missing_input_header_raises():
    ws = FakeWorksheet("계획", ["티커", "계획금액(원)", "메모"], [])
    with pytest.raises(sheets.PlanHeaderError):
        sheets.write_plan_rows(ws, [_plan_row("NVDA")])


def test_scopes_allow_sheet_write_but_drive_stays_readonly():
    assert "https://www.googleapis.com/auth/spreadsheets" in sheets._SCOPES
    assert "https://www.googleapis.com/auth/spreadsheets.readonly" not in sheets._SCOPES
    assert "https://www.googleapis.com/auth/drive.readonly" in sheets._SCOPES


# ── engine: run() 통합 ─────────────────────────────────────────────────────


def _patch_run_with_signal(monkeypatch, tmp_path, score=20, signal=True):
    """tests/test_live_advisor.py의 run() 가짜 IO에 오늘 TEST 종목 A1 신호를 하나 끼워 넣는다."""
    from tests.test_live_advisor import _patch_run_io

    engine_daily = _patch_run_io(monkeypatch, tmp_path)
    monkeypatch.setenv("GOOGLE_SHEETS_ID", "sheet-123")
    if not signal:
        return engine_daily
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


def _run(engine_daily, cfg, client, enabled=True, mode="live", dry_run=False):
    return engine_daily.run(_cfg_with(cfg, enabled=enabled), mode, do_replay=False, dry_run=dry_run, sheets_client=client)


def test_run_default_budget_sizes_unplanned_recommendation(monkeypatch, tmp_path, cfg):
    engine_daily = _patch_run_with_signal(monkeypatch, tmp_path)
    client = FakeClient(_plan_ws(_existing_rows()), portfolio_ws=_portfolio_ws())
    summary = _run(engine_daily, cfg, client)
    row = _test_buy_row(summary)
    assert row["qty"] > 0 and row["amount_krw"] > 0
    assert "기본 금액" in row["note"] and "계획 없음" not in row["note"]
    # 기본 금액 종목은 판정 표·계획에 들어가지 않고, 시트에도 안 쓴다
    assert "TEST" not in summary["plan_by_ticker"]
    assert all(r["ticker"] != "TEST" for r in summary["live_judgment_rows"])
    assert client.all_updates() == []
    assert summary["auto_plan"]["selected"] == []


def test_run_plan_budget_takes_priority_over_default(monkeypatch, tmp_path, cfg):
    engine_daily = _patch_run_with_signal(monkeypatch, tmp_path)
    plan = [{"티커": "TEST", "계획금액(원)": "9,000,000", "등록일": "2026-09-20", "기준가($)": "50", "메모": ""}]
    client = FakeClient(_plan_ws(plan), portfolio_ws=_portfolio_ws(budget="1,000,000", total="100,000,000"))
    summary = _run(engine_daily, cfg, client)
    row = _test_buy_row(summary)
    # 계획금액 9,000,000 · 기준가 $50 · 환율 1400 → 총 128주, 1차 1:2:6 비중
    from core import sizing

    expected = sizing.plan_tranche_qty(9_000_000, "A1", row["limit"], 1400.0, ref_price=50.0)["qty"]
    assert row["qty"] == expected
    assert "기본 금액" not in row["note"]
    assert summary["plan_by_ticker"]["TEST"] == 9_000_000.0


def test_run_portfolio_missing_keeps_zero_qty_and_warns(monkeypatch, tmp_path, cfg):
    engine_daily = _patch_run_with_signal(monkeypatch, tmp_path)
    client = FakeClient(_plan_ws(_existing_rows()))  # 투자현황 탭 없음
    summary = _run(engine_daily, cfg, client)
    row = _test_buy_row(summary)
    assert row["qty"] == 0 and "계획 없음" in row["note"]
    assert sheets.PORTFOLIO_BUDGET_MISSING_WARNING in summary["warnings"]
    assert not any(w.startswith("현금 부족") for w in summary["warnings"])


def test_run_cash_shortfall_marks_row_and_warns(monkeypatch, tmp_path, cfg):
    engine_daily = _patch_run_with_signal(monkeypatch, tmp_path)
    client = FakeClient(_plan_ws(_existing_rows()), portfolio_ws=_portfolio_ws(total="100,000"))
    summary = _run(engine_daily, cfg, client)
    row = _test_buy_row(summary)
    assert row["amount_krw"] > 100_000
    assert "현금 부족" in row["note"]
    assert sum(w.startswith("현금 부족: 오늘 추천 투입금액 합계") for w in summary["warnings"]) == 1


def test_run_cash_remaining_subtracts_invested_principal(monkeypatch, tmp_path, cfg):
    # 총 투자금 300만 − 투자 원금(10주×$200×1400=280만) = 남은 현금 20만 → 1차 추천 금액이 넘는다
    engine_daily = _patch_run_with_signal(monkeypatch, tmp_path)
    plan = [{"티커": "AAPL", "계획금액(원)": "5,000,000", "등록일": "2026-09-20", "기준가($)": "", "메모": ""}]
    fills = _fills_ws([_fill("2026-09-22", "AAPL", 10, 200)])
    client = FakeClient(_plan_ws(plan), fills_ws=fills, portfolio_ws=_portfolio_ws(total="3,000,000"))
    summary = _run(engine_daily, cfg, client)
    row = _test_buy_row(summary)
    assert 200_000 < row["amount_krw"] < 3_000_000
    assert "현금 부족" in row["note"]


@pytest.mark.parametrize("total", ["100,000,000", ""])
def test_run_cash_no_warning_when_enough_or_total_missing(monkeypatch, tmp_path, cfg, total):
    engine_daily = _patch_run_with_signal(monkeypatch, tmp_path)
    client = FakeClient(_plan_ws(_existing_rows()), portfolio_ws=_portfolio_ws(total=total))
    summary = _run(engine_daily, cfg, client)
    assert _test_buy_row(summary)["qty"] > 0
    assert "현금 부족" not in _test_buy_row(summary)["note"]
    assert not any(w.startswith("현금 부족") for w in summary["warnings"])


def test_run_first_buy_writes_plan_row_and_sizes_same_run(monkeypatch, tmp_path, cfg):
    engine_daily = _patch_run_with_signal(monkeypatch, tmp_path, signal=False)
    seen = {}
    original = engine_daily.compute_live_judgments

    def _capture(states, today_events, indicator_map, as_of_by_ticker, name_map, plan_by_ticker, *a, **k):
        seen["plan"] = dict(plan_by_ticker)
        seen["ref"] = dict(k.get("plan_ref_price_by_ticker") or {})
        return original(states, today_events, indicator_map, as_of_by_ticker, name_map, plan_by_ticker, *a, **k)

    monkeypatch.setattr(engine_daily, "compute_live_judgments", _capture)
    fills = _fills_ws([_fill("2026-09-23", "CPRT", 1, 27.55), _fill("2026-09-24", "CPRT", 2, 28.10, unit="2차")])
    client = FakeClient(_plan_ws(_existing_rows()), fills_ws=fills, portfolio_ws=_portfolio_ws())
    summary = _run(engine_daily, cfg, client)
    assert summary["auto_plan"]["written"] == ["CPRT"]
    written = {u["header"]: u["value"] for u in client.plan_ws.updates}
    assert written == {
        "티커": "CPRT", "계획금액(원)": 3000000, "등록일": "2026-09-23", "기준가($)": 27.55,
        "메모": "자동 추가 · 첫 매수 2026-09-23",
    }
    assert {u["row"] for u in client.plan_ws.updates} == {4}
    # 같은 실행의 판정·수량에 바로 반영(메모리 병합): 계획금액·기준가 = 첫 매수가
    assert summary["plan_by_ticker"]["CPRT"] == _BUDGET
    assert seen["plan"]["CPRT"] == _BUDGET and seen["ref"]["CPRT"] == 27.55
    assert fills.updates == [] and client.portfolio_ws.updates == []


def test_run_signal_only_without_buy_writes_no_plan_row(monkeypatch, tmp_path, cfg):
    engine_daily = _patch_run_with_signal(monkeypatch, tmp_path)
    client = FakeClient(_plan_ws(_existing_rows()), portfolio_ws=_portfolio_ws())
    _run(engine_daily, cfg, client)
    assert client.plan_ws.updates == []


def test_run_first_buy_skips_existing_plan_cash_and_sold_out(monkeypatch, tmp_path, cfg):
    engine_daily = _patch_run_with_signal(monkeypatch, tmp_path, signal=False)
    fills = _fills_ws([
        _fill("2026-09-22", "AAPL", 1, 200),  # 이미 계획 있음
        _fill("2026-09-22", "QQQM", 5, 210, unit="대기자금"),  # 대기자금 제외
        _fill("2026-09-22", "AMD", 2, 150),  # 보유 0
        _fill("2026-09-23", "AMD", 2, 155, side="매도"),
    ])
    client = FakeClient(_plan_ws(_existing_rows()), fills_ws=fills, portfolio_ws=_portfolio_ws())
    summary = _run(engine_daily, cfg, client)
    assert client.plan_ws.updates == []
    assert summary["auto_plan"]["selected"] == []


def test_run_first_buy_without_budget_does_not_write_and_warns(monkeypatch, tmp_path, cfg):
    engine_daily = _patch_run_with_signal(monkeypatch, tmp_path, signal=False)
    fills = _fills_ws([_fill("2026-09-23", "CPRT", 1, 27.55)])
    client = FakeClient(_plan_ws(_existing_rows()), fills_ws=fills, portfolio_ws=_portfolio_ws(budget=""))
    summary = _run(engine_daily, cfg, client)
    assert client.all_updates() == []
    assert any(w.startswith("계획 자동 기록 생략:") and "CPRT" in w for w in summary["warnings"])


def test_run_write_failure_warns_and_still_merges(monkeypatch, tmp_path, cfg):
    engine_daily = _patch_run_with_signal(monkeypatch, tmp_path, signal=False)
    fills = _fills_ws([_fill("2026-09-23", "CPRT", 1, 27.55)])
    client = FakeClient(_plan_ws(_existing_rows(), fail_on_write=True), fills_ws=fills, portfolio_ws=_portfolio_ws())
    summary = _run(engine_daily, cfg, client)
    assert any(w.startswith("계획 자동 기록 실패:") for w in summary["warnings"])
    assert summary["report_path"]  # 보고서(브리핑 첨부)는 그대로 만들어짐
    assert summary["plan_by_ticker"]["CPRT"] == _BUDGET


def test_run_disabled_writes_nothing_and_matches_previous_behavior(monkeypatch, tmp_path, cfg):
    engine_daily = _patch_run_with_signal(monkeypatch, tmp_path)

    def _should_not_be_called(*a, **k):
        raise AssertionError("enabled: false면 투자현황 읽기·apply_auto_plan을 부르면 안 된다")

    monkeypatch.setattr(engine_daily, "apply_auto_plan", _should_not_be_called)
    monkeypatch.setattr(engine_daily.sheets, "read_portfolio", _should_not_be_called)
    fills = _fills_ws([_fill("2026-09-23", "CPRT", 1, 27.55)])
    client = FakeClient(_plan_ws(_existing_rows()), fills_ws=fills, portfolio_ws=_portfolio_ws(total="100"))
    summary = _run(engine_daily, cfg, client, enabled=False)
    assert client.all_updates() == []
    assert "auto_plan" not in summary
    assert set(summary["plan_by_ticker"]) == {"AAPL", "MSFT"}
    row = _test_buy_row(summary)
    assert row["qty"] == 0 and "계획 없음" in row["note"] and "현금 부족" not in row["note"]


def test_run_paper_never_writes(monkeypatch, tmp_path, cfg):
    from store import db as db_module

    engine_daily = _patch_run_with_signal(monkeypatch, tmp_path)
    monkeypatch.setattr(db_module, "PAPER_DB_PATH", tmp_path / "paper_state.db")
    monkeypatch.setattr(engine_daily, "load_fills", lambda: sheets.FillsResult())
    fills = _fills_ws([_fill("2026-09-23", "CPRT", 1, 27.55)])
    client = FakeClient(_plan_ws(_existing_rows()), fills_ws=fills, portfolio_ws=_portfolio_ws())
    summary = _run(engine_daily, cfg, client, mode="paper")
    assert client.all_updates() == []
    assert "auto_plan" not in summary


def test_run_dry_run_does_not_write_but_merges(monkeypatch, tmp_path, cfg):
    engine_daily = _patch_run_with_signal(monkeypatch, tmp_path, signal=False)
    fills = _fills_ws([_fill("2026-09-23", "CPRT", 1, 27.55)])
    client = FakeClient(_plan_ws(_existing_rows()), fills_ws=fills, portfolio_ws=_portfolio_ws())
    summary = _run(engine_daily, cfg, client, dry_run=True)
    assert client.all_updates() == []
    assert summary["plan_by_ticker"]["CPRT"] == _BUDGET


def test_user_edited_auto_row_is_not_rewritten():
    # 자동 추가된 줄의 계획금액·기준가를 사용자가 바꾼 뒤에도, 추가 매수가 있어도 그대로 둔다
    ws = _plan_ws([{"티커": "CPRT", "계획금액(원)": "7,000,000", "등록일": "2026-10-01", "기준가($)": "30",
                    "메모": "자동 추가 · 첫 매수 2026-10-01"}])
    plan_df, _ = sheets.parse_plan_records(ws.get_all_records())
    fills = _fills_df([("2026-10-01", "CPRT", "buy", 1, 27.55, 1400), ("2026-10-02", "CPRT", "buy", 2, 28.0, 1400)])
    assert ap.select_first_buy_plan_rows(fills, list(plan_df["ticker"]), _BUDGET) == []
    assert sheets.write_plan_rows(ws, ap.select_first_buy_plan_rows(fills, [], _BUDGET)).written == []
    assert ws.updates == []
    assert ws.values[1][1] == "7,000,000" and ws.values[1][3] == "30"
