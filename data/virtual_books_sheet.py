"""가상 장부 구글 시트 쓰기 — 전략비교_일일·매매기록·전략요약 세 탭만 (docs/design/virtual_books.md).

data/sheets.py(계획·체결)와 별도 모듈이다: 쓸 시트 ID는 config virtual_books.sheet_id_env가
가리키는 환경변수(기본 VIRTUAL_BOOKS_SHEETS_ID)에서 읽고 — 먼저 사본 시트로 시험한 뒤 사용자가
실제 시트로 바꾼다 — 이 세 탭 밖에는 쓰지 못하게 코드로 막는다(계획·체결 탭 보호).

장부는 매 실행 시작일부터 다시 계산하므로 탭마다 통째로 지우고 다시 쓴다 — 같은 날 두 번 실행해도
행이 중복되지 않는다. 탭이 없으면 만든다. 인증은 data/sheets.py의 서비스 계정을 그대로 쓴다
(인증 정보 값은 출력·로그하지 않는다).
"""

from __future__ import annotations

import os

TAB_DAILY = "전략비교_일일"
TAB_TRADES = "매매기록"
TAB_SUMMARY = "전략요약"
ALLOWED_TABS = (TAB_DAILY, TAB_TRADES, TAB_SUMMARY)


class VirtualBooksSheetForbiddenError(RuntimeError):
    """허용된 세 탭 밖에 쓰려 할 때."""


def sheet_id_from_env(env_name: str) -> str | None:
    """환경변수에서 시트 ID를 읽는다 (없으면 None — 호출부가 경고 후 시트 기록만 건너뛴다)."""
    return (os.environ.get(env_name) or "").strip() or None


def _worksheet(spreadsheet, name: str, rows: int, cols: int):
    if name not in ALLOWED_TABS:
        raise VirtualBooksSheetForbiddenError(f"가상 장부 탭 외에는 쓸 수 없습니다: {name!r}")
    import gspread

    try:
        ws = spreadsheet.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=name, rows=max(rows, 100), cols=max(cols, 10))
    if getattr(ws, "title", name) not in ALLOWED_TABS:
        raise VirtualBooksSheetForbiddenError(f"가상 장부 탭 외에는 쓸 수 없습니다: {ws.title!r}")
    return ws


def write_tables(client, sheet_id: str, tables: dict[str, list[list]]) -> dict[str, int]:
    """탭별 표를 통째로 다시 쓴다.

    입력: client(gspread 클라이언트 — open_by_key), sheet_id, tables({탭 이름: 행 목록(첫 행 머리글)})
    출력: {탭 이름: 쓴 행 수(머리글 포함)}
    예외: 허용 탭이 아니면 VirtualBooksSheetForbiddenError (아무 탭도 쓰기 전에 확인)
    """
    bad = [t for t in tables if t not in ALLOWED_TABS]
    if bad:
        raise VirtualBooksSheetForbiddenError(f"가상 장부 탭 외에는 쓸 수 없습니다: {bad}")
    spreadsheet = client.open_by_key(sheet_id)
    out = {}
    for name, rows in tables.items():
        width = max((len(r) for r in rows), default=1)
        padded = [list(r) + [""] * (width - len(r)) for r in rows]
        ws = _worksheet(spreadsheet, name, len(padded) + 10, width)
        ws.clear()
        need_rows, need_cols = len(padded) + 10, width
        if getattr(ws, "row_count", need_rows) < need_rows or getattr(ws, "col_count", need_cols) < need_cols:
            ws.resize(rows=max(need_rows, getattr(ws, "row_count", 0)), cols=max(need_cols, getattr(ws, "col_count", 0)))
        if padded:
            ws.update(values=padded, range_name="A1", value_input_option="RAW")
        out[name] = len(padded)
    return out


def read_tables(client, sheet_id: str, tabs: tuple[str, ...] = ALLOWED_TABS) -> dict[str, list[list]]:
    """가상 장부 탭을 읽기만 한다 (DB 복구용 — engine.picks_books.recover_from_tables).

    입력: client, sheet_id, tabs(허용 세 탭 중) / 출력: {탭 이름: get_all_values() 결과}
    예외: 허용 탭이 아니면 VirtualBooksSheetForbiddenError
    """
    bad = [t for t in tabs if t not in ALLOWED_TABS]
    if bad:
        raise VirtualBooksSheetForbiddenError(f"가상 장부 탭 외에는 읽지 않습니다: {bad}")
    spreadsheet = client.open_by_key(sheet_id)
    return {t: spreadsheet.worksheet(t).get_all_values() for t in tabs}
