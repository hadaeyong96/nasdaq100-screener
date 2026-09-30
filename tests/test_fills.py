"""data/fills.py 로더와 core.state.apply_fill 체결 반영 테스트. 네트워크 없음.

P3: 한글·영어 열 이름, 날짜 3형식(-, /, .), BOM·CP949 인코딩, 오류 줄 건너뛰기.
P3.4: fills.xlsx("체결기록" 시트만, 다른 시트 무시), 엑셀 날짜·글자 날짜, 빈 줄,
대기자금(QQQM) 행, 파일 잠김 경고, xlsx/csv 우선순위.
"""

from __future__ import annotations

import datetime as dt

import openpyxl
import pandas as pd
import pytest

from core import state as st
from data.fills import fills_for, load_fills, load_plan, parse_plan_records, summarize_cash_rows


def test_load_fills_empty_file_returns_empty_frame(tmp_path):
    path = tmp_path / "fills.csv"
    path.write_text("날짜,종목,차수,매수매도,체결가,수량\n", encoding="utf-8")
    result = load_fills(path)
    assert result.df.empty
    assert result.errors == []


def test_load_fills_missing_file_returns_empty_frame(tmp_path):
    result = load_fills(tmp_path / "no_such_file.csv")
    assert result.df.empty
    assert result.errors == []


def test_load_fills_parses_korean_columns(tmp_path):
    path = tmp_path / "fills.csv"
    path.write_text(
        "날짜,종목,차수,매수매도,체결가,수량\n2026-09-10,NVDA,1차,매수,180.25,55\n",
        encoding="utf-8",
    )
    result = load_fills(path)
    assert result.errors == []
    assert len(result.df) == 1
    row = result.df.iloc[0]
    assert row["ticker"] == "NVDA"
    assert row["unit"] == "1"
    assert row["side"] == "buy"
    assert row["price"] == 180.25
    assert row["qty"] == 55

    matched = fills_for(result.df, "NVDA", pd.Timestamp("2026-09-10"))
    assert len(matched) == 1
    assert matched[0]["qty"] == 55


def test_load_fills_parses_legacy_english_columns(tmp_path):
    path = tmp_path / "fills.csv"
    path.write_text(
        "date,ticker,unit,side,price,qty\n2026-09-10,NVDA,1,buy,180.25,55\n",
        encoding="utf-8",
    )
    result = load_fills(path)
    assert result.errors == []
    assert len(result.df) == 1
    assert result.df.iloc[0]["unit"] == "1"
    assert result.df.iloc[0]["side"] == "buy"


def test_load_fills_all_unit_and_side_codes(tmp_path):
    path = tmp_path / "fills.csv"
    path.write_text(
        "날짜,종목,차수,매수매도,체결가,수량\n"
        "2026-09-10,AAA,1차,매수,10,1\n"
        "2026-09-11,AAA,2차,매수,11,1\n"
        "2026-09-12,AAA,3차,매수,12,1\n"
        "2026-09-13,AAA,재진입,매수,13,1\n"
        "2026-09-14,AAA,3차,매도,14,1\n",
        encoding="utf-8",
    )
    result = load_fills(path)
    assert result.errors == []
    assert list(result.df["unit"]) == ["1", "2", "6", "9", "6"]
    assert list(result.df["side"]) == ["buy", "buy", "buy", "buy", "sell"]


def test_load_fills_accepts_three_date_formats(tmp_path):
    path = tmp_path / "fills.csv"
    path.write_text(
        "날짜,종목,차수,매수매도,체결가,수량\n"
        "2026-09-10,AAA,1차,매수,10,1\n"
        "2026/9/11,AAA,1차,매수,10,1\n"
        "2026.9.12,AAA,1차,매수,10,1\n",
        encoding="utf-8",
    )
    result = load_fills(path)
    assert result.errors == []
    assert list(result.df["date"]) == [
        pd.Timestamp("2026-09-10"),
        pd.Timestamp("2026-09-11"),
        pd.Timestamp("2026-09-12"),
    ]


def test_load_fills_reads_utf8_bom(tmp_path):
    path = tmp_path / "fills.csv"
    path.write_bytes(
        "날짜,종목,차수,매수매도,체결가,수량\n2026-09-10,NVDA,1차,매수,180.25,55\n".encode("utf-8-sig")
    )
    result = load_fills(path)
    assert result.errors == []
    assert len(result.df) == 1
    assert result.df.iloc[0]["ticker"] == "NVDA"


def test_load_fills_reads_cp949(tmp_path):
    path = tmp_path / "fills.csv"
    path.write_bytes(
        "날짜,종목,차수,매수매도,체결가,수량\n2026-09-10,NVDA,1차,매수,180.25,55\n".encode("cp949")
    )
    result = load_fills(path)
    assert result.errors == []
    assert len(result.df) == 1
    assert result.df.iloc[0]["ticker"] == "NVDA"


def test_load_fills_skips_bad_rows_and_reports_errors(tmp_path):
    path = tmp_path / "fills.csv"
    path.write_text(
        "날짜,종목,차수,매수매도,체결가,수량\n"
        "2026-09-10,NVDA,1차,매수,180.25,55\n"  # 정상
        "2026-13-99,NVDA,1차,매수,180.25,55\n"  # 날짜 오류
        "2026-09-11,AAPL,9차,매수,180.25,55\n"  # 차수 오류
        "2026-09-12,AAPL,1차,보류,180.25,55\n"  # 매수매도 오류
        "2026-09-13,AAPL,1차,매수,180.25,-5\n"  # 수량 음수
        "2026-09-14,AAPL,1차,매수,abc,5\n",  # 체결가 오류
        encoding="utf-8",
    )
    result = load_fills(path)
    assert len(result.df) == 1  # 정상 줄만 남는다
    assert len(result.errors) == 5
    for line in result.errors:
        assert line.startswith("체결 기록 오류: ")
    assert "2번째" in result.errors[0]


def test_apply_fill_records_real_price_and_qty(cfg):
    state = st.init_state("NVDA")
    state["state"] = "정찰"
    state["units"] = {"1": 0}
    state["entries"] = {"1": None}

    fill = {"unit": "1", "side": "buy", "price": 180.25, "qty": 55}
    new_state = st.apply_fill(state, fill, cfg)

    assert new_state["units"]["1"] == 55
    assert new_state["entries"]["1"] == 180.25


def test_apply_fill_unfilled_a1_returns_to_waiting_without_cooldown(cfg):
    """정찰 단계에서 A1이 체결 안 됨(qty=0)이면 쿨다운 없이 대기로 돌린다."""
    state = st.init_state("NVDA")
    state["state"] = "정찰"
    state["a1_date"] = pd.Timestamp("2026-09-10")
    state["stop"] = 170.0
    state["units"] = {"1": 0}
    state["entries"] = {"1": None}
    state["cooldown_until"] = pd.Timestamp("2026-09-20")  # 혹시 남아 있던 값도 지워져야 한다

    fill = {"unit": "1", "side": "buy", "price": 180.25, "qty": 0}
    new_state = st.apply_fill(state, fill, cfg)

    assert new_state["state"] == "대기"
    assert new_state["a1_date"] is None
    assert new_state["cooldown_until"] is None
    assert "1" not in new_state["units"]


def test_apply_fill_sell_reduces_held_qty(cfg):
    state = st.init_state("NVDA")
    state["units"] = {"1": 55}
    fill = {"unit": "1", "side": "sell", "price": 228.87, "qty": 55}
    new_state = st.apply_fill(state, fill, cfg)
    assert new_state["units"]["1"] == 0


# ── P3.4: data/fills.xlsx ────────────────────────────────────────────────


def _write_fills_xlsx(path, fills_rows, usage_row=("사용법 텍스트", "체결기록 시트만 입력하세요"), example_rows=(("2026-01-01", "QQQM", "대기자금", "매수", 100, 1),)):
    """사용법·체결기록·작성예시 세 시트를 가진 fills.xlsx를 만든다 (사용자 양식 재현)."""
    headers = ["날짜", "종목", "차수", "매수매도", "체결가", "수량", "금액($)", "메모"]
    wb = openpyxl.Workbook()

    ws_usage = wb.active
    ws_usage.title = "사용법"
    ws_usage.append(list(usage_row))

    ws_fills = wb.create_sheet("체결기록")
    ws_fills.append(headers)
    for row in fills_rows:
        ws_fills.append(list(row))

    ws_example = wb.create_sheet("작성예시")
    ws_example.append(headers)
    for row in example_rows:
        ws_example.append(list(row))

    wb.save(path)


def test_load_fills_xlsx_reads_only_체결기록_sheet_ignores_others(tmp_path):
    path = tmp_path / "fills.xlsx"
    _write_fills_xlsx(
        path,
        fills_rows=[("2026-09-10", "NVDA", "1차", "매수", 180.25, 55)],
        example_rows=[("2026-01-01", "BADTICKER", "9999차", "매수", -1, -1)],  # 작성예시는 읽지 않아야 함(오류 안 남아야 함)
    )
    result = load_fills(path)
    assert result.errors == []
    assert len(result.df) == 1
    assert result.df.iloc[0]["ticker"] == "NVDA"


def test_load_fills_xlsx_accepts_excel_date_and_text_date(tmp_path):
    path = tmp_path / "fills.xlsx"
    _write_fills_xlsx(
        path,
        fills_rows=[
            ("2026-09-10", "NVDA", "1차", "매수", 180.25, 55),  # 글자 날짜
            (dt.datetime(2026, 9, 22), "NVDA", "1차", "매도", 228.87, 55),  # 엑셀 날짜(datetime)
        ],
    )
    result = load_fills(path)
    assert result.errors == []
    assert list(result.df["date"]) == [pd.Timestamp("2026-09-10"), pd.Timestamp("2026-09-22")]


def test_load_fills_xlsx_skips_blank_rows(tmp_path):
    path = tmp_path / "fills.xlsx"
    _write_fills_xlsx(
        path,
        fills_rows=[
            ("2026-09-10", "NVDA", "1차", "매수", 180.25, 55),
            (None, None, None, None, None, None),  # 빈 줄
            ("2026-09-11", "AAPL", "1차", "매수", 220.0, 10),
        ],
    )
    result = load_fills(path)
    assert result.errors == []
    assert len(result.df) == 2


def test_load_fills_xlsx_cash_rows_go_to_대기자금_bucket_not_signal_df(tmp_path):
    path = tmp_path / "fills.xlsx"
    _write_fills_xlsx(
        path,
        fills_rows=[
            ("2026-09-10", "NVDA", "1차", "매수", 180.25, 55),
            ("2026-09-24", "QQQM", "대기자금", "매수", 271.4, 40),
        ],
    )
    result = load_fills(path)
    assert result.errors == []
    assert list(result.df["ticker"]) == ["NVDA"]  # 신호 판정용 df에는 QQQM이 없다
    assert len(result.cash_rows) == 1
    assert result.cash_rows.iloc[0]["ticker"] == "QQQM"


def test_load_fills_xlsx_locked_file_warns_without_crashing(tmp_path):
    path = tmp_path / "fills.xlsx"
    _write_fills_xlsx(path, fills_rows=[("2026-09-10", "NVDA", "1차", "매수", 180.25, 55)])
    lock_file = tmp_path / "~$fills.xlsx"
    lock_file.write_text("")

    result = load_fills(path)
    assert result.df.empty
    assert len(result.errors) == 1
    assert "열려 있음" in result.errors[0]


def test_load_fills_prefers_xlsx_over_csv_with_warning(tmp_path, monkeypatch):
    import data.fills as fills_module

    xlsx_path = tmp_path / "fills.xlsx"
    csv_path = tmp_path / "fills.csv"
    _write_fills_xlsx(xlsx_path, fills_rows=[("2026-09-10", "NVDA", "1차", "매수", 180.25, 55)])
    csv_path.write_text("날짜,종목,차수,매수매도,체결가,수량\n2026-09-10,AAPL,1차,매수,220.0,10\n", encoding="utf-8")

    monkeypatch.setattr(fills_module, "FILLS_XLSX", xlsx_path)
    monkeypatch.setattr(fills_module, "FILLS_CSV", csv_path)

    result = load_fills()
    assert list(result.df["ticker"]) == ["NVDA"]  # xlsx 우선
    assert any("모두 있어" in e for e in result.errors)


# ── 라이브 어드바이저 1단계: fills.xlsx의 "계획" 시트 (load_plan) ───────────


def _write_plan_xlsx(path, plan_rows, include_plan_sheet=True):
    """fills.xlsx와 같은 파일에 "계획" 시트를 (선택적으로) 추가한 파일을 만든다."""
    headers = ["티커", "계획금액", "메모"]
    wb = openpyxl.Workbook()
    ws_fills = wb.active
    ws_fills.title = "체결기록"
    ws_fills.append(["날짜", "종목", "차수", "매수매도", "체결가", "수량"])

    if include_plan_sheet:
        ws_plan = wb.create_sheet("계획")
        ws_plan.append(headers)
        for row in plan_rows:
            ws_plan.append(list(row))

    wb.save(path)


def test_load_plan_reads_계획_sheet(tmp_path):
    path = tmp_path / "fills.xlsx"
    _write_plan_xlsx(path, plan_rows=[("NVDA", 3_000_000, "1차 대기")])
    plan_df, errors = load_plan(path)
    assert errors == []
    assert len(plan_df) == 1
    row = plan_df.iloc[0]
    assert row["ticker"] == "NVDA"
    assert row["budget_krw"] == 3_000_000.0
    assert row["memo"] == "1차 대기"


def test_load_plan_missing_sheet_returns_empty_without_error(tmp_path):
    """계획 시트는 선택 사항 — 기존 fills.xlsx(계획 시트 없음)를 그대로 써도 오류가 아니다."""
    path = tmp_path / "fills.xlsx"
    _write_plan_xlsx(path, plan_rows=[], include_plan_sheet=False)
    plan_df, errors = load_plan(path)
    assert plan_df.empty
    assert errors == []


def test_load_plan_missing_file_returns_empty_without_error(tmp_path):
    plan_df, errors = load_plan(tmp_path / "no_such_file.xlsx")
    assert plan_df.empty
    assert errors == []


def test_load_plan_reports_invalid_rows(tmp_path):
    path = tmp_path / "fills.xlsx"
    _write_plan_xlsx(path, plan_rows=[("", 1_000_000, ""), ("AVGO", "많이", "")])
    plan_df, errors = load_plan(path)
    assert plan_df.empty
    assert len(errors) == 2
    assert "티커가 비어 있음" in errors[0]
    assert "계획금액이 올바르지 않음" in errors[1]


def test_load_plan_defaults_to_project_fills_xlsx_path(monkeypatch, tmp_path):
    import data.fills as fills_module

    path = tmp_path / "fills.xlsx"
    _write_plan_xlsx(path, plan_rows=[("NVDA", 3_000_000, "")])
    monkeypatch.setattr(fills_module, "FILLS_XLSX", path)

    plan_df, errors = load_plan()
    assert errors == []
    assert list(plan_df["ticker"]) == ["NVDA"]


# ── 계획 탭 개편(2026-09-30): 주식 수 기준 새 머리글 ─────────────────────────
# 티커, 계획금액, 등록일, 기준가($), 환율, 1차(주), 2차(주), 3차(주), 합계(주),
# 보유(주), 남은(주), 메모 — 환율·1차(주)~남은(주)는 시트 수식이라 읽지 않는다
# (core.sizing.plan_tranche_qty가 budget_krw·ref_price·그날 환율로 직접 다시
# 계산한다). parse_plan_records는 data/sheets.py(구글 시트)와 load_plan(xlsx)
# 양쪽이 공유하므로, 여기서는 파일 없이 레코드 dict로 직접 테스트한다.


def _new_header_plan_record(**overrides) -> dict:
    record = {
        "티커": "ODFL",
        "계획금액": "3,000,000",
        "등록일": "2026-09-01",
        "기준가($)": "178.41",
        "환율": "1,362",  # 시트 수식 결과 — 읽지 않아야 한다
        "1차(주)": "1",  # 시트 수식 결과 — 읽지 않아야 한다(코드가 직접 다시 계산)
        "2차(주)": "3",
        "3차(주)": "8",
        "합계(주)": "12",
        "보유(주)": "4",
        "남은(주)": "8",
        "메모": "",
    }
    record.update(overrides)
    return record


def test_parse_plan_records_reads_new_share_based_headers():
    plan_df, errors = parse_plan_records([_new_header_plan_record()])
    assert errors == []
    assert len(plan_df) == 1
    row = plan_df.iloc[0]
    assert row["ticker"] == "ODFL"
    assert row["budget_krw"] == 3_000_000.0
    assert row["ref_price"] == 178.41
    assert row["memo"] == ""


def test_parse_plan_records_ignores_formula_columns_not_just_ref_price():
    """등록일·환율·1차(주)~남은(주)는 시트 수식이라, 값이 계산과 안 맞아도(고의로
    틀리게 넣어도) 파싱 결과(ref_price)에 영향을 주면 안 된다."""
    record = _new_header_plan_record(**{"환율": "9999", "1차(주)": "999", "합계(주)": "999"})
    plan_df, errors = parse_plan_records([record])
    assert errors == []
    assert plan_df.iloc[0]["ref_price"] == 178.41  # 시트의 "1차(주)" 등과 무관


def test_parse_plan_records_ref_price_blank_means_use_old_method():
    """기준가($)가 비어 있으면(선택 열) None — core.sizing.plan_tranche_qty가
    지금 방식(그날 지정가 기준)으로 폴백한다."""
    record = _new_header_plan_record(**{"기준가($)": ""})
    plan_df, errors = parse_plan_records([record])
    assert errors == []
    assert pd.isna(plan_df.iloc[0]["ref_price"]) or plan_df.iloc[0]["ref_price"] is None


def test_parse_plan_records_invalid_ref_price_is_reported():
    record = _new_header_plan_record(**{"기준가($)": "많이"})
    plan_df, errors = parse_plan_records([record])
    assert plan_df.empty
    assert "기준가가 올바르지 않음" in errors[0]


def test_load_fills_falls_back_to_csv_when_no_xlsx(tmp_path, monkeypatch):
    import data.fills as fills_module

    csv_path = tmp_path / "fills.csv"
    csv_path.write_text("날짜,종목,차수,매수매도,체결가,수량\n2026-09-10,AAPL,1차,매수,220.0,10\n", encoding="utf-8")

    monkeypatch.setattr(fills_module, "FILLS_XLSX", tmp_path / "no_fills.xlsx")
    monkeypatch.setattr(fills_module, "FILLS_CSV", csv_path)

    result = load_fills()
    assert result.errors == []
    assert list(result.df["ticker"]) == ["AAPL"]


def test_summarize_cash_rows_tracks_weighted_average_and_partial_sell():
    cash_rows = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-09-24"), "ticker": "QQQM", "unit": "cash", "side": "buy", "price": 271.4, "qty": 40},
            {"date": pd.Timestamp("2026-10-08"), "ticker": "QQQM", "unit": "cash", "side": "buy", "price": 275.0, "qty": 17},
        ]
    )
    summary = summarize_cash_rows(cash_rows)
    assert summary["qty"] == 57
    assert summary["avg_price"] == pytest.approx((271.4 * 40 + 275.0 * 17) / 57)


def test_summarize_cash_rows_empty_returns_none():
    assert summarize_cash_rows(pd.DataFrame(columns=["date", "ticker", "unit", "side", "price", "qty"])) is None
