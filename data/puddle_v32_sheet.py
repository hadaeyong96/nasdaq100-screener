"""사용자 시트(GOOGLE_SHEETS_ID, 「전략공장」)의 탭 「웅덩이 v3.2 (가상·시총)」 하나에만 쓴다 (2026-10-27, 사용자 승인).

탭 교체(사전 등록 v3.2 2-8-3): 새 탭이 없고 옛 「웅덩이 v3.1 (가상)」 탭이 있으면 그 탭 이름을 새 이름으로 바꾸고
1행 제목·2행 머리글만 남기고 비운 뒤 새로 기록한다(옛 기록은 DB 보관본으로만). 둘 다 없으면 새로 만든다.
그 밖의 탭은 열지도 쓰지도 않는다(_assert_tab). 1행 제목, 2행 머리글, 3행부터 기준일마다 한 줄(같은 기준일은 덮어쓰기).
반등 확인 v2 탭(data/puddle_rebound_v2_sheet.py)도 이 모듈의 open_tab·upsert를 쓴다.
"""

from __future__ import annotations

TAB = "웅덩이 v3.2 (가상·시총)"
OLD_TAB = "웅덩이 v3.1 (가상)"
TITLE = "상승장 대형주 웅덩이 v3.2 (시가총액 상위 10) — 가상 1억, 실제 돈 아님"
HEADER = ["날짜", "v3.2 평가금액(원)", "QQQ 보유(원)", "필터 없는 판(원)", "QQQ 대비(%p)", "무작위 중앙값 대비(%p)", "QQQ 비중",
          "보유 종목(비중)", "오늘 매매", "올해 실현 이익(원)"]


class WrongTabError(RuntimeError):
    """이 탭이 아닌 워크시트에 쓰려 할 때."""


def _assert(ws, tab: str) -> None:
    if getattr(ws, "title", None) != tab:
        raise WrongTabError(f"「{tab}」 탭에만 쓸 수 있습니다(받은 탭: {getattr(ws, 'title', None)})")


def open_tab(spreadsheet, tab: str, old_tab: str, cols: int):
    """새 탭 열기 → 없으면 옛 탭을 새 이름으로 바꾸고 3행부터 비움 → 그것도 없으면 새로 만든다. 출력: (워크시트, 교체했는가)."""
    try:
        ws = spreadsheet.worksheet(tab)
        migrated = False
    except Exception:  # gspread.WorksheetNotFound
        try:
            ws = spreadsheet.worksheet(old_tab)
        except Exception:  # 옛 탭도 없음
            ws = spreadsheet.add_worksheet(title=tab, rows=400, cols=cols)
            migrated = False
        else:
            if getattr(ws, "title", None) != old_tab:
                raise WrongTabError(f"옛 탭 「{old_tab}」이 아닌 탭을 받았습니다: {getattr(ws, 'title', None)}")
            ws.update_title(tab)
            ws.batch_clear([f"A1:ZZ{max(int(getattr(ws, 'row_count', 400) or 400), 3)}"])  # 1·2행은 upsert가 새 제목·머리글로 다시 쓴다
            migrated = True
    _assert(ws, tab)
    return ws, migrated


def upsert(ws, rows: list[list], tab: str, title: str, header: list[str]) -> int:
    """제목·머리글을 맞추고 rows(첫 칸 = 날짜 문자열)를 날짜 줄에 덮어쓰거나 끝에 붙인다. 출력: 쓴 줄 수."""
    _assert(ws, tab)
    values = ws.get_all_values()
    data = values[2:] if len(values) > 2 else []
    where = {r[0]: i + 3 for i, r in enumerate(data) if r and r[0]}
    nxt = max([3] + [i + 1 for i in where.values()])
    updates = [{"range": "A1", "values": [[title]]}, {"range": "A2", "values": [header]}]
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
    """사용자 시트의 「웅덩이 v3.2 (가상·시총)」 탭에만 쓴다(첫 기록 때 옛 v3.1 탭을 이 이름으로 바꾸고 비움)."""
    ws, _ = open_tab(client.open_by_key(sheet_id), TAB, OLD_TAB, len(HEADER))
    return upsert(ws, rows, TAB, TITLE, HEADER)
