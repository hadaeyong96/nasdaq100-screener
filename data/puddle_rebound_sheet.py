"""사용자 시트(GOOGLE_SHEETS_ID, 「전략공장」)의 탭 「웅덩이 반등 확인 (가상)」 하나에만 쓴다 (2026-10-24, 사용자 승인).

다른 탭은 열지도 쓰지도 않는다(_assert_tab). 1행 제목, 2행 머리글, 3행부터 기준일마다 한 줄. 같은 기준일을 다시 쓰면 그 줄을 덮어쓴다.
쓰는 방식은 「웅덩이 v3.1 (가상)」 탭(data/puddle_v31_sheet.py)과 같다.
"""

from __future__ import annotations

from data import puddle_v31_sheet as _base

TAB = "웅덩이 반등 확인 (가상)"
TITLE = "웅덩이 반등 확인 — 가상 1억, 실제 돈 아님"
HEADER = ["날짜", "평가금액(원)", "QQQ 보유(원)", "v3.1(원)", "QQQ 대비(%p)", "v3.1 대비(%p)", "무작위 중앙값 대비(%p)",
          "QQQ 비중", "보유 종목(비중)", "오늘 매매", "반등 확인 대기 종목", "올해 실현 이익(원)"]
WrongTabError = _base.WrongTabError


def _assert_tab(ws) -> None:
    if getattr(ws, "title", None) != TAB:
        raise WrongTabError(f"「{TAB}」 탭에만 쓸 수 있습니다(받은 탭: {getattr(ws, 'title', None)})")


def _worksheet(spreadsheet):
    """탭을 열고, 없으면 만든다."""
    try:
        ws = spreadsheet.worksheet(TAB)
    except Exception:  # gspread.WorksheetNotFound
        ws = spreadsheet.add_worksheet(title=TAB, rows=400, cols=len(HEADER))
    _assert_tab(ws)
    return ws


def upsert_rows(ws, rows: list[list]) -> int:
    """제목·머리글을 맞추고 rows(첫 칸 = 날짜)를 날짜 줄에 덮어쓰거나 끝에 붙인다. 출력: 쓴 줄 수."""
    _assert_tab(ws)
    values = ws.get_all_values()
    data = values[2:] if len(values) > 2 else []
    where = {r[0]: i + 3 for i, r in enumerate(data) if r}
    nxt = len(values) + 1 if len(values) >= 2 else 3
    updates = [{"range": "A1", "values": [[TITLE]]}, {"range": "A2", "values": [HEADER]}]
    for row in rows:
        r = where.get(str(row[0]))
        if r is None:
            r = nxt
            nxt += 1
            where[str(row[0])] = r
        updates.append({"range": f"A{r}", "values": [row]})
    need = max(nxt, 3)
    if getattr(ws, "row_count", need) < need:
        ws.resize(rows=need + 50)
    ws.batch_update(updates, value_input_option="RAW")
    return len(rows)


def write_rows(client, sheet_id: str, rows: list[list]) -> int:
    """사용자 시트의 「웅덩이 반등 확인 (가상)」 탭에만 쓴다."""
    return upsert_rows(_worksheet(client.open_by_key(sheet_id)), rows)
