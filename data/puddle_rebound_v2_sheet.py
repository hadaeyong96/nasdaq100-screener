"""사용자 시트(GOOGLE_SHEETS_ID, 「전략공장」)의 탭 「웅덩이 반등 확인 v2 (가상·시총)」 하나에만 쓴다 (2026-10-27, 사용자 승인).

탭 교체(사전 등록 반등 확인 v2 2-6-4): 새 탭이 없고 옛 「웅덩이 반등 확인 (가상)」 탭이 있으면 이름을 바꾸고 비운 뒤 새로 기록한다.
쓰는 방식은 「웅덩이 v3.2 (가상·시총)」 탭(data/puddle_v32_sheet.py)과 같다. 다른 탭은 읽기만.
"""

from __future__ import annotations

from data import puddle_v32_sheet as _base

TAB = "웅덩이 반등 확인 v2 (가상·시총)"
OLD_TAB = "웅덩이 반등 확인 (가상)"
TITLE = "웅덩이 반등 확인 v2 (시가총액 상위 10) — 가상 1억, 실제 돈 아님"
HEADER = ["날짜", "평가금액(원)", "QQQ 보유(원)", "v3.2(원)", "QQQ 대비(%p)", "v3.2 대비(%p)", "무작위 중앙값 대비(%p)",
          "QQQ 비중", "보유 종목(비중)", "오늘 매매", "반등 확인 대기 종목", "올해 실현 이익(원)"]
WrongTabError = _base.WrongTabError


def write_rows(client, sheet_id: str, rows: list[list]) -> int:
    """사용자 시트의 「웅덩이 반등 확인 v2 (가상·시총)」 탭에만 쓴다(첫 기록 때 옛 탭을 이 이름으로 바꾸고 비움)."""
    ws, _ = _base.open_tab(client.open_by_key(sheet_id), TAB, OLD_TAB, len(HEADER))
    return _base.upsert(ws, rows, TAB, TITLE, HEADER)
