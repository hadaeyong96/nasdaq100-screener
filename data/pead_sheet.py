"""전략공장 시트(GOOGLE_SHEETS_ID)는 실적급등 기록 전용 (2026-11-13, 사용자 결정) — 탭 「일별 성적」·「매매이력」 두 개에만 쓴다.

줄 내용은 core/pead_sheet.py가 만든다. 쓰는 방식은 웅덩이 탭(data/puddle_v32_sheet.py)의 upsert를 그대로 쓴다:
1행 제목, 2행 머리글(매번 맞춤), 3행부터 키가 같으면 그 줄을 덮어쓰고 없으면 끝에 붙인다.
「일별 성적」 키 = 날짜, 「매매이력」 키 = 티커+매수일. 그 밖의 탭은 만들지도 쓰지도 않는다.
웅덩이·QQQM 탭(RETIRED_TABS)은 2026-11-13 백업 뒤 지웠고, 이 모듈은 그 탭들을 다시 만들지 않는다.
"""

from __future__ import annotations

from core import pead_sheet as ps
from data import puddle_v32_sheet as _base

DAILY_TAB = "일별 성적"
TRADE_TAB = "매매이력"
TABS = {DAILY_TAB: (ps.DAILY_HEADER, (0,)), TRADE_TAB: (ps.TRADE_HEADER, ps.TRADE_KEY)}
RETIRED_TABS = ("QQQM 모으기", "웅덩이 v3.2 (가상·시총)", "웅덩이 반등 확인 v2 (가상·시총)",
                "웅덩이 v3.1 (가상)", "웅덩이 반등 확인 (가상)")
WrongTabError = _base.WrongTabError


def _col(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def open_tab(spreadsheet, tab: str):
    """탭 열기 — 없으면 만들고 2행까지 고정·머리글 굵게. 출력: (워크시트, 새로 만들었는가). TABS 밖 이름은 거부."""
    if tab not in TABS:
        raise WrongTabError(f"실적급등 탭({', '.join(TABS)})만 열 수 있습니다: {tab}")
    try:
        return spreadsheet.worksheet(tab), False
    except Exception:  # gspread.WorksheetNotFound
        header = TABS[tab][0]
        ws = spreadsheet.add_worksheet(title=tab, rows=400, cols=len(header))
        ws.freeze(rows=2)
        ws.format(f"A2:{_col(len(header))}2", {"textFormat": {"bold": True}})
        return ws, True


def write_tab(spreadsheet, tab: str, rows: list[list]) -> int:
    """한 탭에 rows를 upsert. 출력: 쓴 줄 수."""
    header, key = TABS[tab]
    ws, _ = open_tab(spreadsheet, tab)
    return _base.upsert(ws, rows, tab, ps.TITLE, header, key=key)


def write_all(client, sheet_id: str, daily: list[list], trades: list[list]) -> dict:
    """두 탭 모두 쓴다. 출력: {탭: 쓴 줄 수}."""
    sp = client.open_by_key(sheet_id)
    return {DAILY_TAB: write_tab(sp, DAILY_TAB, daily), TRADE_TAB: write_tab(sp, TRADE_TAB, trades)}
