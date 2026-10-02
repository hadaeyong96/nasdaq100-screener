"""scripts/sheet_setup.py 테스트 — 가짜 스프레드시트, 네트워크 없이."""

from __future__ import annotations

import copy
import re

import pytest

from scripts import sheet_setup as ss

# ── 가짜 스프레드시트 ───────────────────────────────────────────────────────


class FakeWs:
    def __init__(self, title, sid, grid):
        self.title = title
        self.id = sid
        self.grid = [list(r) for r in grid]

    def get_all_values(self, value_render_option=None):
        rows = copy.deepcopy(self.grid)
        while rows and not any(str(x).strip() for x in rows[-1]):
            rows.pop()
        width = max((len(r) for r in rows), default=0)
        return [r + [""] * (width - len(r)) for r in rows]

    def set(self, row, col, value):
        while len(self.grid) < row:
            self.grid.append([])
        line = self.grid[row - 1]
        while len(line) < col:
            line.append("")
        line[col - 1] = value


class FakeSpreadsheet:
    def __init__(self, tabs: dict[str, list[list]]):
        self.tabs = {t: FakeWs(t, i + 100, g) for i, (t, g) in enumerate(tabs.items())}
        self.props = {t: {"index": i} for i, t in enumerate(tabs)}
        self.protected: list[dict] = []
        self.cf: dict[int, list[dict]] = {}
        self.dev_meta: list[dict] = []
        self.write_calls = 0
        self.time_zone = "America/Los_Angeles"
        self.hidden_cols: dict[int, set[int]] = {}  # sheetId -> 숨긴 열(0부터)

    # gspread 인터페이스
    def worksheets(self):
        return sorted(self.tabs.values(), key=lambda w: self.props[w.title]["index"])

    def add_worksheet(self, title, rows, cols):
        self.write_calls += 1
        ws = FakeWs(title, 100 + len(self.tabs), [])
        self.tabs[title] = ws
        self.props[title] = {"index": len(self.props)}
        return ws

    def fetch_sheet_metadata(self, params=None):
        sheets = []
        for t, ws in self.tabs.items():
            sheets.append({
                "properties": {"title": t, "sheetId": ws.id, **self.props[t]},
                "protectedRanges": [p for p in self.protected if p["sheet"] == ws.id],
                "conditionalFormats": self.cf.get(ws.id, []),
                "data": [{"columnMetadata": [{"hiddenByUser": True} if c in self.hidden_cols.get(ws.id, set()) else {}
                                             for c in range(80)]}],
            })
        return {"properties": {"locale": "ko_KR", "timeZone": self.time_zone}, "developerMetadata": list(self.dev_meta),
                "sheets": sheets}

    def _ws_by_id(self, sid):
        return next(w for w in self.tabs.values() if w.id == sid)

    def batch_update(self, body):
        self.write_calls += 1
        for req in body["requests"]:
            (kind, spec), = req.items()
            if kind == "moveDimension":
                ws = self._ws_by_id(spec["source"]["sheetId"])
                ws.grid = ss.apply_column_ops(ws.grid, [{"op": "move", "from": spec["source"]["startIndex"], "to": spec["destinationIndex"]}])
            elif kind == "insertDimension":
                ws = self._ws_by_id(spec["range"]["sheetId"])
                ws.grid = ss.apply_column_ops(ws.grid, [{"op": "insert", "at": spec["range"]["startIndex"]}])
            elif kind == "addProtectedRange":
                pr = spec["protectedRange"]
                self.protected.append({"sheet": pr["range"]["sheetId"], "description": pr["description"], "warningOnly": pr["warningOnly"]})
            elif kind == "addConditionalFormatRule":
                rule = spec["rule"]
                self.cf.setdefault(rule["ranges"][0]["sheetId"], []).append(rule)
            elif kind == "createDeveloperMetadata":
                m = dict(spec["developerMetadata"])
                m["metadataId"] = len(self.dev_meta) + 1
                self.dev_meta.append(m)
            elif kind == "deleteDeveloperMetadata":
                mid = spec["dataFilter"]["developerMetadataLookup"]["metadataId"]
                self.dev_meta = [m for m in self.dev_meta if m["metadataId"] != mid]
            elif kind == "deleteConditionalFormatRule":
                del self.cf[spec["sheetId"]][spec["index"]]
            elif kind == "updateCells" and "rows" not in spec and "userEnteredValue" in spec["fields"]:
                rng = spec["range"]
                ws = self._ws_by_id(rng["sheetId"])
                r_end = rng.get("endRowIndex", len(ws.grid))
                for r in range(rng["startRowIndex"], min(r_end, len(ws.grid))):
                    for c in range(rng["startColumnIndex"], min(rng["endColumnIndex"], len(ws.grid[r]))):
                        ws.grid[r][c] = ""
            elif kind == "updateSpreadsheetProperties":
                self.time_zone = spec["properties"]["timeZone"]
            elif kind == "updateDimensionProperties" and spec["properties"].get("hiddenByUser"):
                rng = spec["range"]
                self.hidden_cols.setdefault(rng["sheetId"], set()).update(range(rng["startIndex"], rng["endIndex"]))
            elif kind == "updateSheetProperties":
                p = spec["properties"]
                title = self._ws_by_id(p["sheetId"]).title
                for f in ("index", "hidden"):
                    if f in p:
                        self.props[title][f] = p[f]

    def values_batch_update(self, body):
        self.write_calls += 1
        for item in body["data"]:
            tab, rng = item["range"].split("!")
            ws = self.tabs[tab.strip("'")]
            r0, c0 = ss.parse_a1(rng.split(":")[0])
            for i, row in enumerate(item["values"]):
                for j, v in enumerate(row):
                    ws.set(r0 + i, c0 + j, v)


class FakeClient:
    def __init__(self, sh):
        self.sh = sh

    def open_by_key(self, key):
        return self.sh


_PLAN_HEADER = ["티커", "계획금액", "등록일", "기준가", "환율", "1차", "2차", "3차", "합계", "보유주", "남은주", "메모"]
_PLAN_ROW = ["OLED", "9000000", "46293", "178.41", '=IF(A2="","",GOOGLEFINANCE("CURRENCY:USDKRW"))',
             "=IF(I2=\"\",\"\",MAX(1,ROUND(I2/9)))", "=x", "=x", "=x", "=x", "=x", ""]
_FILLS_OLD = [
    ["날짜", "종목", "구분", "수량", "체결가", "환율", "수수료", "차수", "메모"],
    ["46294", "ODFL", "매수", "4", "178.41", "", "", "1차", ""],
    ["46295", "AAPL", "매수", "2", "230.5", "1390", "0.3", "", "메모 남김"],
]


def _sheet(fills=_FILLS_OLD, plan_a2="OLED", extra_tabs=None):
    plan_row = [plan_a2] + _PLAN_ROW[1:]
    tabs = {"계획": [_PLAN_HEADER, plan_row], "체결": fills}
    tabs.update(extra_tabs or {})
    return FakeSpreadsheet(copy.deepcopy(tabs))


def _list_rows(fills_tickers):
    return ss.build_list_rows(["ODFL", "AAPL", "MSFT"], {"AAPL": "애플", "ODFL": "올드 도미니언"}, {"AAPL": "Technology"}, fills_tickers)


def _run(sh, apply=True, tmp_path=None):
    lines = []
    setup = ss.SheetSetup(FakeClient(sh), "sheet-xyz-1234", _list_rows, apply=apply, out=lines.append,
                          backup_dir=tmp_path)
    return setup.run(), lines


# ── 보기 모드·멱등 ──────────────────────────────────────────────────────────


def test_view_mode_writes_nothing(tmp_path):
    sh = _sheet()
    before = {t: copy.deepcopy(w.grid) for t, w in sh.tabs.items()}
    result, lines = _run(sh, apply=False, tmp_path=tmp_path)
    assert sh.write_calls == 0
    assert {t: w.grid for t, w in sh.tabs.items()} == before
    assert result["changes"]  # 바꿀 내용은 출력
    assert not list(tmp_path.iterdir())  # 보기 모드는 백업도 안 만든다
    assert any("…1234" in line for line in lines) and not any("sheet-xyz" in line for line in lines)


def test_apply_twice_second_run_changes_nothing(tmp_path):
    sh = _sheet()
    first, _ = _run(sh, tmp_path=tmp_path)
    assert first["verify"]["problems"] == []
    grids = {t: copy.deepcopy(w.grid) for t, w in sh.tabs.items()}
    calls = sh.write_calls
    second, _ = _run(sh, tmp_path=tmp_path)
    assert second["changes"] == []
    assert sh.write_calls == calls  # 두 번째는 쓰기 0번
    assert {t: w.grid for t, w in sh.tabs.items()} == grids
    assert len(sh.protected) == len({p["description"] for p in sh.protected})  # 보호 중복 없음
    assert all(len(rules) == len({r["booleanRule"]["condition"]["values"][0]["userEnteredValue"] for r in rules}) for rules in sh.cf.values())


def test_backup_written_before_apply(tmp_path):
    sh = _sheet()
    result, _ = _run(sh, tmp_path=tmp_path)
    path = result["backup"]
    assert path.exists() and path.name.startswith("sheet_backup_")
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["tabs"]["체결"]["formulas"][1][1] == "ODFL"
    assert data["sheet_id_tail"] == "1234"


# ── 체결 열 이동 ─────────────────────────────────────────────────────────────


def test_fill_columns_reordered_and_data_unchanged(tmp_path):
    sh = _sheet()
    before = ss.rows_by_header(copy.deepcopy(_FILLS_OLD))
    _run(sh, tmp_path=tmp_path)
    grid = sh.tabs["체결"].get_all_values()
    assert grid[0][:13] == ss.FILLS_HEADERS
    after = ss.rows_by_header(grid)
    for b, a in zip(before, after):
        assert {k: a[k] for k in b} == b
    assert grid[1][ss.FILLS_HEADERS.index("차수")] == "1차"
    assert grid[2][ss.FILLS_HEADERS.index("환율(원/$)")] == "1390"
    assert grid[2][ss.FILLS_HEADERS.index("메모")] == "메모 남김"
    assert grid[1][6].startswith("=ARRAYFORMULA(")


def test_plan_fill_columns_noop_when_already_ordered():
    ops, header = ss.plan_fill_columns(ss.FILLS_HEADERS)
    assert ops == [] and header == ss.FILLS_HEADERS


def test_plan_fill_columns_moves_are_leftward_and_keep_unknown_columns():
    ops, header = ss.plan_fill_columns(["날짜", "종목", "구분", "수량", "체결가", "내 열", "차수"])
    assert all(op["to"] < op["from"] for op in ops if op["op"] == "move")
    assert header[:13] == ss.FILLS_HEADERS and header[13:] == ["내 열"]


# ── 투자현황 B2·B3 ──────────────────────────────────────────────────────────


def test_portfolio_keeps_existing_b2_b3(tmp_path):
    existing = [["항목", "값"], ["총 투자금(원)", "77,000,000"], ["종목당 계획금액(원)", "6000000"]]
    sh = _sheet(extra_tabs={"투자현황": existing})
    _run(sh, tmp_path=tmp_path)
    g = sh.tabs["투자현황"].grid
    assert g[1][1] == "77,000,000" and g[2][1] == "6000000"


def test_portfolio_fills_defaults_when_empty(tmp_path):
    sh = _sheet()
    _run(sh, tmp_path=tmp_path)
    g = sh.tabs["투자현황"].grid
    assert (g[1][0], g[1][1]) == ("총 투자금(원)", "40000000")
    assert (g[2][0], g[2][1]) == ("종목당 계획금액(원)", "5000000")


def test_portfolio_labels_match_program_reader():
    from data import sheets

    values, _ = ss.portfolio_values("", "")
    grid = [values["A1:B1"][0], [values["A2"][0][0], values["B2"][0][0]], [values["A3"][0][0], values["B3"][0][0]]]
    assert sheets.parse_portfolio_values(grid) == {"total_krw": 40_000_000.0, "per_ticker_budget_krw": 5_000_000.0}


# ── 계획 A2 ─────────────────────────────────────────────────────────────────


def test_plan_a2_oled_fixed_only_that_cell(tmp_path):
    sh = _sheet()
    result, _ = _run(sh, tmp_path=tmp_path)
    g = sh.tabs["계획"].grid
    assert g[1][0] == "ODFL" and g[1][1:] == _PLAN_ROW[1:]
    assert result["verify"]["problems"] == []


def test_plan_a2_not_oled_left_alone_and_reported(tmp_path):
    sh = _sheet(plan_a2="NVDA")
    result, _ = _run(sh, tmp_path=tmp_path)
    assert sh.tabs["계획"].grid[1][0] == "NVDA"
    assert any("계획 A2" in o for o in result["oddities"])


# ── 보호·탭 순서 ─────────────────────────────────────────────────────────────


def test_protections_warning_only_and_tab_order(tmp_path):
    sh = _sheet()
    _run(sh, tmp_path=tmp_path)
    assert sh.protected and all(p["warningOnly"] for p in sh.protected)
    assert [w.title for w in sh.worksheets()] == ss.TAB_ORDER
    assert sh.props["목록"].get("hidden") is True
    plan_cols = {p["description"] for p in sh.protected if "계획 수식 열" in p["description"]}
    assert plan_cols == {f"{ss.PROTECT_PREFIX}계획 수식 열 {c}" for c in "EFGHIJK"}


def test_oddities_unknown_ticker_and_text_number(tmp_path):
    fills = copy.deepcopy(_FILLS_OLD) + [["46296", "ZZZZ", "매수", "abc", "10", "", "", "", ""]]
    sh = _sheet(fills=fills)
    result, _ = _run(sh, apply=False, tmp_path=tmp_path)
    assert any("수량: 숫자 아님" in o for o in result["oddities"])
    rows = _list_rows(["ZZZZ"])
    assert ["ZZZZ", "", ""] in rows  # 체결 티커는 목록에 넣는다(빈 이름)


# ── 매매일지 짝짓기 (수식 평가기로 확인) ─────────────────────────────────────
# 시트 수식을 그대로 돌릴 수는 없으므로, 보조 열 수식이 담은 규칙을 같은 순서로 파이썬에 옮긴
# 참조 구현(_journal_reference)과, 생성된 수식 문자열이 그 규칙의 핵심 조건을 담고 있는지 둘 다 본다.


def _journal_reference(fills):
    """보조 열 규칙(key·held·cycle·idx·unit·첫 매도 짝짓기)을 그대로 옮긴 파이썬 판정."""
    rows = []
    for r, f in enumerate(fills, start=2):
        rows.append({**f, "key": f["date"] * 1000 + r, "cash": f.get("unit") == "대기자금"})
    for x in rows:
        x["sq"] = 0 if x["cash"] else (x["qty"] if x["side"] == "매수" else -x["qty"])
    for x in rows:
        x["held"] = sum(y["sq"] for y in rows if y["tk"] == x["tk"] and y["key"] < x["key"])
        x["isbuy"] = x["side"] == "매수" and not x["cash"]
        x["cstart"] = x["isbuy"] and x["held"] <= 0
    out = []
    for x in rows:
        if x["side"] == "매도":
            x["u"] = x.get("unit", "")
    for x in rows:
        if not x["isbuy"]:
            continue
        ckey = max(y["key"] for y in rows if y["tk"] == x["tk"] and y["cstart"] and y["key"] <= x["key"])
        idx = sum(1 for y in rows if y["tk"] == x["tk"] and y["isbuy"] and ckey <= y["key"] <= x["key"])
        x["u"] = x.get("unit") or ["1차", "2차", "3차"][min(idx, 3) - 1]
    for x in sorted((x for x in rows if x["isbuy"]), key=lambda x: x["key"]):
        sells = [y for y in rows if y["tk"] == x["tk"] and y["side"] == "매도" and y.get("u") == x["u"] and y["key"] > x["key"]]
        s = min(sells, key=lambda y: y["key"]) if sells else None
        status = "보유 중" if s is None else ("매도 완료" if s["qty"] >= x["qty"] else f"일부 매도 (남은 {x['qty'] - s['qty']}주)")
        out.append((x["tk"], x["u"], status))
    return out


def _f(date, tk, side, qty, unit=""):
    return {"date": date, "tk": tk, "side": side, "qty": qty, "unit": unit}


def test_journal_pairing_complete_partial_holding_and_cash_excluded():
    fills = [
        _f(100, "ODFL", "매수", 4),             # 1차 → 1차 매도 4주: 완료
        _f(101, "ODFL", "매수", 8),             # 2차 → 2차 매도 3주: 일부
        _f(102, "QQQM", "매수", 10, "대기자금"),  # 대기자금: 매매일지 제외
        _f(103, "AAPL", "매수", 2),             # 1차, 매도 없음: 보유 중
        _f(104, "ODFL", "매도", 4, "1차"),
        _f(105, "ODFL", "매도", 3, "2차"),
    ]
    assert _journal_reference(fills) == [
        ("ODFL", "1차", "매도 완료"),
        ("ODFL", "2차", "일부 매도 (남은 5주)"),
        ("AAPL", "1차", "보유 중"),
    ]


def test_journal_cycle_restarts_after_full_exit_like_program():
    # 전량 매도로 0주가 되면 다음 매수는 다시 1차 (data/fills.py _assign_units와 같은 규칙)
    fills = [_f(100, "NVDA", "매수", 1), _f(101, "NVDA", "매도", 1, "1차"), _f(102, "NVDA", "매수", 2)]
    assert _journal_reference(fills) == [("NVDA", "1차", "매도 완료"), ("NVDA", "1차", "보유 중")]
    # 프로그램 자동 배정과 같은지 직접 확인
    from data.fills import parse_fill_records

    recs = [{"날짜": f"2026-10-0{i + 1}", "티커": "NVDA", "구분": s, "수량": q, "체결가": 10, "차수": u}
            for i, (s, q, u) in enumerate([("매수", 1, ""), ("매도", 1, "1차"), ("매수", 2, "")])]
    df = parse_fill_records(recs).df
    assert list(df[df["side"] == "buy"]["unit"]) == ["1", "1"]


def test_journal_auto_units_match_program_assignment():
    from data.fills import parse_fill_records

    fills = [_f(100, "AMD", "매수", 1), _f(101, "AMD", "매수", 2), _f(102, "AMD", "매수", 6)]
    units = [u for _, u, _ in _journal_reference(fills)]
    recs = [{"날짜": f"2026-10-0{i + 1}", "티커": "AMD", "구분": "매수", "수량": q, "체결가": 10, "차수": ""}
            for i, q in enumerate([1, 2, 6])]
    code = {"1": "1차", "2": "2차", "6": "3차"}
    assert units == [code[u] for u in parse_fill_records(recs).df["unit"]] == ["1차", "2차", "3차"]


def test_journal_formulas_encode_rules():
    vals = ss.journal_values()
    helper_range = next(k for k in vals if k.startswith(ss.HELPER_COLS["key"]) and ":" in k and k.endswith(str(ss.J_LAST)))
    row = dict(zip([n for n, _ in ss._HELPERS], vals[helper_range][0]))
    assert '="대기자금"' in row["cash"]  # 대기자금 제외
    assert '"매수"' in row["isbuy"] and "NOT(" in row["isbuy"]
    assert 'MINIFS(' in row["skey"] and '"매도"' in row["skey"] and '">"&' in row["skey"]  # 매수 뒤 첫 매도, 같은 차수
    assert "일부 매도" in row["v_status"] and "매도 완료" in row["v_status"] and "보유 중" in row["v_status"]
    assert "체결'!A2" in row["key"]  # 첫 보조 줄 = 체결 2행
    assert vals["A4:U4"][0] == ss.JOURNAL_HEADERS
    assert len(vals[helper_range]) == ss.FILL_ROWS
    # 보이는 열 21개가 머리글 21개와 같은 순서
    v_names = [n for n, _ in ss._HELPERS if n.startswith("v_")]
    assert len(v_names) == len(ss.JOURNAL_HEADERS)
    # 수식 안에 남은 미치환 자리표시자가 없다
    for formula in vals[helper_range][0] + vals[helper_range][-1]:
        assert not re.search(r"\{[a-z_]+(:all)?\}", formula), formula


# ── 마무리(시간대·계획 형식·QQQM 업종·보조 열 숨김) ─────────────────────────────


def test_time_zone_set_to_seoul_and_kept(tmp_path):
    sh = _sheet()
    _run(sh, tmp_path=tmp_path)
    assert sh.time_zone == "Asia/Seoul"
    second, _ = _run(sh, tmp_path=tmp_path)
    assert second["changes"] == []


def test_plan_number_formats_b_and_d():
    ids = {t: i for i, t in enumerate(ss.TAB_ORDER)}
    reqs = [r["repeatCell"] for r in ss.format_requests(ids) if "repeatCell" in r and r["repeatCell"]["range"]["sheetId"] == ids["계획"]]
    by_col = {r["range"]["startColumnIndex"]: r["cell"]["userEnteredFormat"]["numberFormat"]["pattern"] for r in reqs}
    assert by_col == {1: "#,##0", 3: "0.00"}  # B열, D열(기준가)


def test_qqqm_sector_is_etf():
    rows = ss.build_list_rows(["AAPL"], {}, {"AAPL": "Technology"}, [])
    assert ["QQQM", "", "ETF"] in rows and ["AAPL", "", "Technology"] in rows


def test_journal_helper_columns_rehidden_if_shown(tmp_path):
    sh = _sheet()
    _run(sh, tmp_path=tmp_path)
    jid = sh.tabs["매매일지"].id
    assert set(range(ss.J_HIDE_FROM - 1, ss.J_HELPER_START + len(ss._HELPERS) - 1)) <= sh.hidden_cols[jid]
    sh.hidden_cols[jid].discard(30)  # 사용자가 보조 열 하나를 다시 펼침
    result, _ = _run(sh, tmp_path=tmp_path)
    assert any("매매일지 보조 열 숨김" in c for c in result["changes"])
    assert 30 in sh.hidden_cols[jid]
    assert any(p["description"] == ss.PROTECT_PREFIX + "매매일지 전체" for p in sh.protected)


def test_name_kr_csv_has_no_blank_names():
    import pandas as pd

    kr = pd.read_csv(ss.ROOT / "data" / "name_kr.csv", dtype=str).fillna("")
    assert (kr["name_kr"].str.strip() == "").sum() == 0
    assert dict(zip(kr["ticker"], kr["name_kr"]))["ODFL"] == "올드 도미니언 프레이트 라인"


# ── v3.2: 투자현황 2열 + 보유현황 ────────────────────────────────────────────


def _old_portfolio_layout():
    """v3.1 배치: 3열 코어·위성 표, 업종 표(C열 비중), 종목별 현황(A40~, K·L 보조 열)."""
    g = [["항목", "값"], ["총 투자금(원)", "77,000,000"], ["종목당 계획금액(원)", "6,000,000"], [""],
         ["[자산 요약]"], ["환율(원/$)", "=GOOGLEFINANCE(\"CURRENCY:USDKRW\")"]]
    g += [[""]] * 6
    g += [["[코어·위성]", "평가금액(원)", "비중"], ["코어 (대기자금 QQQM)", "=x", "=y"]]
    g += [[""]] * 25
    g += [["[종목별 현황]"], ["종목", "종목명", "업종", "보유주", "평균단가($)", "현재가($)", "손익률", "투자 원금(원)",
                           "평가금액(원)", "비중", "코어(보조)", "업종(보조)"], ["=IFERROR(SORT(...))", "=b", "=c"]]
    return g


def test_portfolio_two_columns_only(tmp_path):
    sh = _sheet()
    _run(sh, tmp_path=tmp_path)
    g = sh.tabs["투자현황"].get_all_values()
    assert all(not str(v).strip() for row in g for v in row[2:])  # C열 이후 비어 있음
    labels = [row[0] for row in g]
    for label in ["[자산]", "환율(원/$)", "투자 중 원금(원)", "평가금액(원)", "남은 현금(원)", "투자 비중", "전체 손익률",
                  "[코어·위성]", "코어(QQQM) 평가금액(원)", "코어 비중", "위성 평가금액(원)", "위성 비중",
                  "[세금 (근사값, 실제는 증권사 기준)]", "올해 실현손익(원)", "250만 원 공제 남은 금액(원)", "예상 양도세(원)",
                  "[업종별 비중 (위성 대비)]"]:
        assert label in labels, label
    assert g[1][0] == "총 투자금(원)" and g[2][0] == "종목당 계획금액(원)"
    sector = g[ss.P_SECTOR_FIRST - 1][0]
    assert "QUERY(" in sector and "'위성'" in sector and "desc" in sector and "$B$16" in sector


def test_portfolio_old_layout_reset_keeps_b2_b3_and_program_reads_them(tmp_path):
    from data import sheets

    p_old = _old_portfolio_layout()
    sh = _sheet(extra_tabs={"투자현황": p_old})
    pid = sh.tabs["투자현황"].id
    sh.cf[pid] = [{"ranges": [{"sheetId": pid}], "booleanRule": {"condition": {"values": [{"userEnteredValue": "=AND(ISNUMBER($C23),$C23>0.3)"}]}}}]
    result, _ = _run(sh, tmp_path=tmp_path)
    assert any("예전 배치 지우기" in c for c in result["changes"])
    g = sh.tabs["투자현황"].get_all_values()
    assert g[1][1] == "77,000,000" and g[2][1] == "6,000,000"  # 입력값 보존
    assert all(not str(v).strip() for row in g for v in row[2:])
    assert "종목" not in [row[0] for row in g[3:]]  # 예전 종목별 현황 머리글 없음
    assert "=AND(ISNUMBER($C23),$C23>0.3)" not in [r["booleanRule"]["condition"]["values"][0]["userEnteredValue"] for r in sh.cf[pid]]
    assert sheets.parse_portfolio_values(g) == {"total_krw": 77_000_000.0, "per_ticker_budget_krw": 6_000_000.0}
    second, _ = _run(sh, tmp_path=tmp_path)
    assert second["changes"] == []


def test_holdings_tab_columns_and_protection(tmp_path):
    sh = _sheet()
    _run(sh, tmp_path=tmp_path)
    g = sh.tabs["보유현황"].get_all_values()
    assert g[0][:11] == ["종목", "종목명", "업종", "구분", "보유주", "평균단가($)", "현재가($)", "손익률", "투자 원금(원)",
                         "평가금액(원)", "비중"]
    assert g[1][0].startswith("=IFERROR(SORT(UNIQUE(FILTER(UPPER('체결'!$B$2")
    assert '"코어","위성"' in g[1][3] and '"대기자금"' in g[1][3] and '"ETF"' in g[1][3]
    assert any(p["description"] == ss.PROTECT_PREFIX + "보유현황 전체" and p["warningOnly"] for p in sh.protected)
    assert ss.H_FX_COL - 1 in sh.hidden_cols[sh.tabs["보유현황"].id]


def test_tab_order_v32(tmp_path):
    sh = _sheet()
    _run(sh, tmp_path=tmp_path)
    assert [w.title for w in sh.worksheets()] == ["투자현황", "보유현황", "매매일지", "체결", "계획", "관심", "목록"]
    assert sh.props["목록"]["hidden"] is True


def test_portfolio_formulas_reference_holdings_not_hidden_columns():
    vals, _ = ss.portfolio_values("1", "1")
    formulas = " ".join(v for rng in vals.values() for row in rng for v in row)
    assert "'보유현황'!$J$" in formulas and "'보유현황'!$I$" in formulas
    assert all(re.fullmatch(r"[AB]\d+(:[AB]\d+)?", k) for k in vals)  # A·B 열만 쓴다
