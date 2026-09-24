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
from data.fills import fills_for, load_fills, summarize_cash_rows


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
