"""구글 시트 정리 — 시트 재설계안 v3.1 (docs/design/sheet_layout.md).

사용법:
    python scripts/sheet_setup.py [시트ID]            보기 모드 — 바꿀 내용만 출력 (쓰기 0번)
    python scripts/sheet_setup.py [시트ID] --apply    적용 (백업 → 적용 → 비교 → read_sheets 점검)

시트ID를 생략하면 .env의 GOOGLE_SHEETS_ID(원본)를 쓴다. 인증은 data/sheets.py의 get_client를
그대로 쓴다(spreadsheets 쓰기 권한). 인증 정보는 출력하지 않고, 시트 ID는 끝 4자리만 보여 준다.

여러 번 돌려도 결과가 같다:
- 값·수식: 지금 시트의 값(수식 그대로 읽은 것)과 같으면 쓰지 않는다
- 체결 열 순서: 이미 목표 순서면 열 이동 0번
- 보호 범위·조건부 서식: 같은 것이 있으면 건너뛴다
- 검사·색·형식·메모·너비: 요청 묶음의 해시를 시트 개발자 메타데이터에 남기고, 같으면 건너뛴다

데이터 안전:
- 체결 데이터는 값을 읽어 다시 쓰지 않고 열 이동(moveDimension)·빈 열 삽입으로만 옮긴다
- 탭·데이터 줄은 지우지 않는다. 투자현황 B2·B3에 값이 있으면 덮어쓰지 않는다
- 계획 탭은 A2가 "OLED"일 때 그 한 칸만 "ODFL"로 고친다
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "outputs"

# ── 탭 이름 ─────────────────────────────────────────────────────────────────
TAB_PORTFOLIO = "투자현황"
TAB_HOLDINGS = "보유현황"
TAB_JOURNAL = "매매일지"
TAB_FILLS = "체결"
TAB_PLAN = "계획"
TAB_WATCH = "관심"
TAB_LIST = "목록"
TAB_ORDER = [TAB_PORTFOLIO, TAB_HOLDINGS, TAB_JOURNAL, TAB_FILLS, TAB_PLAN, TAB_WATCH, TAB_LIST]
NEW_TABS = {TAB_PORTFOLIO: (200, 14), TAB_HOLDINGS: (220, 14), TAB_JOURNAL: (410, 70), TAB_WATCH: (60, 8),
            TAB_LIST: (300, 3)}

FORMAT_HASH_KEY = "sheet_setup_format_hash"
TIME_ZONE = "Asia/Seoul"  # TODAY()(보유일수·올해 실현손익)를 한국 날짜로
SPECIAL_SECTORS = {"QQQM": "ETF"}  # yfinance 업종이 없는 종목
PROTECT_PREFIX = "sheet_setup: "

# ── 체결 탭 ─────────────────────────────────────────────────────────────────
FILLS_HEADERS = [
    "날짜", "티커", "구분", "수량", "체결가($)", "차수", "종목명", "손절가($)", "매도이유", "규칙대로",
    "환율(원/$)", "수수료($)", "메모",
]
# 체결 열 문자 — 수식·검사·색은 모두 이 표에서 열을 찾는다(열 순서를 바꿔도 FILLS_HEADERS만 고치면 된다)
_FILL_KEYS = {
    "date": "날짜", "tk": "티커", "side": "구분", "qty": "수량", "price": "체결가($)", "unit": "차수", "name": "종목명",
    "stop": "손절가($)", "reason": "매도이유", "rule": "규칙대로", "fx": "환율(원/$)", "fee": "수수료($)", "memo": "메모",
}


def _fill_col(key: str) -> int:
    """체결 열 번호(1부터)."""
    return FILLS_HEADERS.index(_FILL_KEYS[key]) + 1


FC = {k: chr(64 + FILLS_HEADERS.index(h) + 1) for k, h in _FILL_KEYS.items()}  # {"fx": "K", ...} (M열까지라 한 글자)
# 지금 시트 머리글 → 새 머리글 (괄호·공백을 뺀 이름으로 비교한다)
FILLS_RENAME = {"종목": "티커", "체결가": "체결가($)", "환율": "환율(원/$)", "수수료": "수수료($)", "손절가": "손절가($)"}
FILL_ROWS = 400  # 매매일지가 따라가는 체결 줄 수(체결 2~401행 — 매수 줄 200개 + 매도 줄 여유)
FILL_LAST = FILL_ROWS + 1
UNITS = ["1차", "2차", "3차", "재진입", "대기자금"]
SELL_REASONS = ["데드크로스", "RSI 50 이탈", "구름 이탈", "손절", "1차 만료", "기타"]
BUY_REASON = {"1차": "RSI 30 탈출", "2차": "골든크로스", "3차": "구름 돌파", "재진입": "추세 재진입"}

COLOR_INPUT = {"red": 1.0, "green": 0.949, "blue": 0.8}  # 연노랑 #FFF2CC
COLOR_OPTIONAL = {"red": 1.0, "green": 0.988, "blue": 0.918}  # 아주 연한 노랑 #FFFCEA
COLOR_AUTO = {"red": 0.937, "green": 0.937, "blue": 0.937}  # 회색 #EFEFEF
COLOR_RED = {"red": 0.957, "green": 0.8, "blue": 0.8}  # 연빨강 #F4CCCC
COLOR_RED_TEXT = {"red": 0.8, "green": 0.0, "blue": 0.0}
COLOR_GRAY_TEXT = {"red": 0.6, "green": 0.6, "blue": 0.6}

# ── 매매일지 탭 ──────────────────────────────────────────────────────────────
JOURNAL_HEADERS = [
    "종목명", "티커", "업종", "차수", "매수이유", "매수일", "매수가($)", "수량", "투자금액(원)", "손절가($)", "목표가 2R($)",
    "상태", "현재가($)", "수익률", "매도일", "매도가($)", "매도이유", "실현손익(원)", "R 배수", "보유일수", "규칙대로",
]
JOURNAL_SUMMARY_LABELS = ["거래 수", "완료 수", "승률", "평균 R", "실현손익 합계(원)", '규칙대로 "예" 비율']
J_HEADER_ROW = 4
J_FIRST = 5  # 데이터 첫 줄(보조 열도 같은 줄부터 체결 2행에 대응)
J_LAST = J_FIRST + FILL_ROWS - 1
J_FX_CELL = "$W$2"  # 숨김 — 현재 환율
J_HELPER_START = 27  # AA열부터 숨김 보조 열
J_HIDE_FROM = 23  # W열부터 숨김

# ── 투자현황 탭 ──────────────────────────────────────────────────────────────
P_DEFAULT_TOTAL = 40_000_000
P_DEFAULT_BUDGET = 5_000_000
# 투자현황은 A·B 두 열만 쓴다(항목 | 값). 줄 번호는 PORTFOLIO_LAYOUT에서 정해진다.
P_SECTOR_HEADER = 24  # [업종별 비중 (위성 대비)]
P_SECTOR_FIRST = 25  # 업종 | 비중 펼침 시작 — 이 아래는 펼침 영역이라 비워 둔다
P_SECTOR_LAST = 60
P_CLEAR_COLS = 26  # 재배치 때 지우는 열 수(A~Z)

# ── 보유현황 탭 ──────────────────────────────────────────────────────────────
HOLDINGS_HEADERS = [
    "종목", "종목명", "업종", "구분", "보유주", "평균단가($)", "현재가($)", "손익률", "투자 원금(원)", "평가금액(원)", "비중",
]
H_FIRST, H_LAST = 2, 201  # 체결 B열 고유 티커 200개까지
H_FX_COL = 13  # M열(숨김) — 현재 환율

WATCH_ROWS = (2, 51)


# ── 작은 도구 ────────────────────────────────────────────────────────────────


def col_letter(col: int) -> str:
    """1부터 시작하는 열 번호 → 열 문자 (1→A, 27→AA)."""
    out = ""
    while col > 0:
        col, rem = divmod(col - 1, 26)
        out = chr(65 + rem) + out
    return out


def col_index(letters: str) -> int:
    """열 문자 → 1부터 시작하는 열 번호."""
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch.upper()) - 64)
    return n


def normalize_header(name) -> str:
    """머리글 비교용: 괄호 설명과 공백을 뺀다 (data/fills.py _normalize_header와 같은 규칙)."""
    text = re.sub(r"[\(（].*?[\)）]", "", str(name))
    return re.sub(r"\s+", "", text)


def mask_id(sheet_id: str) -> str:
    return f"…{sheet_id[-4:]}" if sheet_id else "(없음)"


def q(tab: str) -> str:
    """수식·범위용 탭 이름 따옴표."""
    return f"'{tab}'"


def _str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


# ── 1. 목록 탭 ───────────────────────────────────────────────────────────────


def build_list_rows(universe_tickers, name_map: dict, sector_map: dict, fills_tickers) -> list[list[str]]:
    """목록 탭 값(머리글 포함). 나스닥100 + QQQM + 체결 티커, 티커 오름차순.

    입력: universe_tickers, name_map({티커: 한글 이름}), sector_map({티커: 업종}), fills_tickers
    출력: [["티커","종목명","업종"], [티커, 이름 또는 "", 업종 또는 ""], ...] — 업종이 없으면 SPECIAL_SECTORS(QQQM "ETF")
    """
    tickers = {str(t).strip().upper() for t in list(universe_tickers) + ["QQQM"] + list(fills_tickers) if str(t).strip()}
    rows = [["티커", "종목명", "업종"]]
    for t in sorted(tickers):
        rows.append([t, name_map.get(t, "") or "", sector_map.get(t, "") or SPECIAL_SECTORS.get(t, "")])
    return rows


# ── 2. 체결 열 이동 계획 ──────────────────────────────────────────────────────


def canonical_fill_header(name: str) -> str:
    """지금 머리글 → 새 머리글 이름 (모르는 머리글은 그대로)."""
    key = normalize_header(name)
    for target in FILLS_HEADERS:
        if normalize_header(target) == key:
            return target
    return FILLS_RENAME.get(key, str(name).strip())


def plan_fill_columns(header: list[str]) -> tuple[list[dict], list[str]]:
    """체결 열을 목표 순서로 만드는 열 작업 목록 (순수 함수).

    입력: header(지금 1행 값 — 뒤쪽 빈 칸 포함 가능)
    출력: (ops [{"op": "move", "from": i, "to": j} | {"op": "insert", "at": i}] — 0부터, 앞에서부터 차례로
          적용, 이동은 항상 왼쪽으로), 적용 후 머리글(새 이름))
    목표에 없는 머리글이 있는 열은 M열 뒤로 밀려 그대로 남는다(지우지 않음).
    """
    cur = [canonical_fill_header(h) if str(h).strip() else "" for h in header]
    ops: list[dict] = []
    for i, target in enumerate(FILLS_HEADERS):
        j = next((k for k in range(i, len(cur)) if cur[k] == target), None)
        if j == i:
            continue
        if j is None:
            ops.append({"op": "insert", "at": i})
            cur.insert(i, target)
        else:
            ops.append({"op": "move", "from": j, "to": i})
            cur.insert(i, cur.pop(j))
    while cur and cur[-1] == "":
        cur.pop()
    return ops, cur


def apply_column_ops(grid: list[list], ops: list[dict]) -> list[list]:
    """plan_fill_columns의 열 작업을 값 격자에 적용한 결과 (시뮬레이션·가짜 시트용, 순수 함수)."""
    width = max((len(r) for r in grid), default=0)
    out = [list(r) + [""] * (width - len(r)) for r in grid]
    for op in ops:
        for r in out:
            if op["op"] == "insert":
                r.insert(op["at"], "")
            else:
                r.insert(op["to"], r.pop(op["from"]))
    return out


def column_op_requests(sheet_id: int, ops: list[dict]) -> list[dict]:
    """열 작업 → Sheets API batchUpdate 요청. 이동은 왼쪽으로만이라 destinationIndex = to."""
    reqs = []
    for op in ops:
        if op["op"] == "insert":
            reqs.append({"insertDimension": {
                "range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": op["at"], "endIndex": op["at"] + 1},
                "inheritFromBefore": False}})
        else:
            reqs.append({"moveDimension": {
                "source": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": op["from"], "endIndex": op["from"] + 1},
                "destinationIndex": op["to"]}})
    return reqs


def rows_by_header(grid: list[list], rename: bool = True) -> list[dict]:
    """데이터 줄(2행부터)을 {새 머리글 이름: 값}으로 — 열 이동 전후 비교용. 빈 줄·빈 머리글 열은 뺀다."""
    if not grid:
        return []
    header = [canonical_fill_header(h) if rename else str(h) for h in grid[0]]
    out = []
    for row in grid[1:]:
        rec = {h: _str(row[i]) if i < len(row) else "" for i, h in enumerate(header) if str(h).strip()}
        if any(v.strip() for v in rec.values()):
            out.append(rec)
    return out


# ── 3. 매매일지 수식 ─────────────────────────────────────────────────────────

_F = q(TAB_FILLS)


def _fcol(c: str) -> str:
    """체결 열 전체 범위(1~FILL_LAST행) — INDEX용."""
    return f"{_F}!${c}$1:${c}${FILL_LAST}"


# 보조 열 정의: (이름, 수식 템플릿). {r}=체결 행, {h}=이 줄, {이름}=같은 줄의 보조 열 셀,
# {이름:all}=보조 열 전체 범위. 순서대로 AA열부터 배치한다.
_HELPERS: list[tuple[str, str]] = [
    ("key", f'=IF({_F}!{FC["date"]}{{r}}="","",IFERROR(({_F}!{FC["date"]}{{r}}+0)*1000+{{r}},""))'),
    ("tk", f'=UPPER(TRIM({_F}!{FC["tk"]}{{r}}))'),
    ("side", f'={_F}!{FC["side"]}{{r}}&""'),
    ("cash", f'={_F}!{FC["unit"]}{{r}}="대기자금"'),
    ("sq", '=IF(OR({key}="",{tk}="",{cash}),0,IF({side}="매수",1,IF({side}="매도",-1,0))*N(' + _F + '!' + FC['qty'] + '{r}))'),
    ("held", '=IF({key}="","",SUMIFS({sq:all},{tk:all},{tk},{key:all},"<"&{key}))'),
    ("isbuy", '=AND({key}<>"",{tk}<>"",{side}="매수",NOT({cash}))'),
    ("cstart", '=AND({isbuy},N({held})<=0)'),
    ("ckey", '=IF({isbuy},MAXIFS({key:all},{tk:all},{tk},{cstart:all},TRUE,{key:all},"<="&{key}),"")'),
    ("idx", '=IF({isbuy},COUNTIFS({tk:all},{tk},{isbuy:all},TRUE,{key:all},">="&{ckey},{key:all},"<="&{key}),"")'),
    ("unit", '=IF({isbuy},IF(' + _F + '!' + FC['unit'] + '{r}<>"",' + _F + '!' + FC['unit'] + '{r},CHOOSE(MIN({idx},3),"1차","2차","3차")),IF({side}="매도",' + _F + '!' + FC['unit'] + '{r}&"",""))'),
    ("skey", '=IF({isbuy},IFERROR(1/(1/MINIFS({key:all},{tk:all},{tk},{side:all},"매도",{unit:all},{unit},{key:all},">"&{key})),""),"")'),
    ("srow", '=IF({skey}="","",MOD({skey},1000))'),
    ("sqty", '=IF({srow}="","",INDEX(' + _fcol(FC['qty']) + ',{srow}))'),
    ("sfx", '=IF({srow}="","",IF(INDEX(' + _fcol(FC['fx']) + ',{srow})<>"",INDEX(' + _fcol(FC['fx']) + ',{srow}),' + J_FX_CELL + '))'),
    # ── 보이는 열과 같은 순서(종목명~규칙대로) ──
    ("v_name", '=IF({isbuy},IFERROR(VLOOKUP({tk},' + q(TAB_LIST) + '!$A:$B,2,FALSE),""),"")'),
    ("v_tk", '=IF({isbuy},{tk},"")'),
    ("v_sector", '=IF({isbuy},IFERROR(VLOOKUP({tk},' + q(TAB_LIST) + '!$A:$C,3,FALSE),""),"")'),
    ("v_unit", '=IF({isbuy},{unit},"")'),
    ("v_reason", '=IF({isbuy},IFERROR(VLOOKUP({unit},{"1차","RSI 30 탈출";"2차","골든크로스";"3차","구름 돌파";"재진입","추세 재진입"},2,FALSE),""),"")'),
    ("v_bdate", '=IF({isbuy},' + _F + '!' + FC['date'] + '{r},"")'),
    ("v_bprice", '=IF({isbuy},' + _F + '!' + FC['price'] + '{r},"")'),
    ("v_qty", '=IF({isbuy},' + _F + '!' + FC['qty'] + '{r},"")'),
    ("v_amount", '=IF({isbuy},{v_bprice}*{v_qty}*IF(' + _F + '!' + FC['fx'] + '{r}<>"",' + _F + '!' + FC['fx'] + '{r},' + J_FX_CELL + '),"")'),
    ("v_stop", '=IF(AND({isbuy},' + _F + '!' + FC['stop'] + '{r}<>""),' + _F + '!' + FC['stop'] + '{r},"")'),
    ("v_target", '=IF({v_stop}="","",{v_bprice}+2*({v_bprice}-{v_stop}))'),
    ("v_status", '=IF({isbuy},IF({srow}="","보유 중",IF({sqty}<{v_qty},"일부 매도 (남은 "&({v_qty}-{sqty})&"주)","매도 완료")),"")'),
    ("v_now", '=IF(AND({isbuy},OR({srow}="",N({sqty})<{v_qty})),IFERROR(GOOGLEFINANCE({tk},"price"),""),"")'),
    ("v_ret", '=IF({isbuy},IF({srow}="",IF({v_now}="","",{v_now}/{v_bprice}-1),{v_sprice}/{v_bprice}-1),"")'),
    ("v_sdate", '=IF({srow}="","",INDEX(' + _fcol(FC['date']) + ',{srow}))'),
    ("v_sprice", '=IF({srow}="","",INDEX(' + _fcol(FC['price']) + ',{srow}))'),
    ("v_sreason", '=IF({srow}="","",INDEX(' + _fcol(FC['reason']) + ',{srow})&"")'),
    ("v_pnl", '=IF({srow}="","",({v_sprice}-{v_bprice})*{sqty}*{sfx}-(N(' + _F + '!' + FC['fee'] + '{r})+N(INDEX(' + _fcol(FC['fee']) + ',{srow})))*{sfx})'),
    ("v_r", '=IF(OR({srow}="",{v_stop}="",{v_bprice}={v_stop}),"",({v_sprice}-{v_bprice})/({v_bprice}-{v_stop}))'),
    ("v_days", '=IF({isbuy},IF({srow}="",TODAY(),{v_sdate})-{v_bdate},"")'),
    # 규칙대로: 매도 줄 값이 있으면 그것, 없으면 매수 줄 값 (AND 안의 INDEX는 매도 없을 때 오류라 IF를 겹친다)
    ("v_rule", '=IF({isbuy},IF({srow}="",' + _F + '!' + FC['rule'] + '{r}&"",IF(INDEX(' + _fcol(FC['rule']) + ',{srow})&""<>"",INDEX(' + _fcol(FC['rule']) + ',{srow})&"",' + _F + '!' + FC['rule'] + '{r}&"")),"")'),
]
HELPER_COLS = {name: col_letter(J_HELPER_START + i) for i, (name, _) in enumerate(_HELPERS)}


def _render_helper(template: str, h: int) -> str:
    r = h - J_FIRST + 2

    def sub(m):
        name, _, kind = m.group(1).partition(":")
        if name not in HELPER_COLS:
            return m.group(0)
        c = HELPER_COLS[name]
        return f"${c}${J_FIRST}:${c}${J_LAST}" if kind == "all" else f"{c}{h}"

    text = re.sub(r"\{([a-z_]+(?::all)?)\}", sub, template)
    return text.replace("{r}", str(r)).replace("{h}", str(h))


def journal_values() -> dict[str, list[list[str]]]:
    """매매일지 탭에 쓸 값·수식 {A1 범위: 2차원 값}."""
    first_v, last_v = HELPER_COLS["v_name"], HELPER_COLS["v_rule"]
    isbuy, key = HELPER_COLS["isbuy"], HELPER_COLS["key"]
    rng = lambda c: f"{c}{J_FIRST}:{c}{J_LAST}"  # noqa: E731
    vis = lambda c: f"{c}{J_FIRST}:{c}{J_LAST}"  # noqa: E731  — 보이는 열(A~U)
    summary = [
        f'=COUNTIF({vis("B")},"?*")',
        f'=COUNTIF({vis("L")},"매도 완료")',
        f'=IFERROR(COUNTIF({vis("R")},">0")/COUNT({vis("R")}),"")',
        f'=IFERROR(AVERAGE({vis("S")}),"")',
        f'=SUM({vis("R")})',
        f'=IFERROR(COUNTIF({vis("U")},"예")/COUNTIF({vis("U")},"?*"),"")',
    ]
    out = {
        "A1:F1": [JOURNAL_SUMMARY_LABELS],
        "A2:F2": [summary],
        f"A{J_HEADER_ROW}:U{J_HEADER_ROW}": [JOURNAL_HEADERS],
        f"A{J_FIRST}": [[
            f"=IFERROR(SORT(FILTER({first_v}{J_FIRST}:{last_v}{J_LAST},{rng(isbuy)}=TRUE),"
            f"FILTER({rng(key)},{rng(isbuy)}=TRUE),TRUE),\"\")"
        ]],
        "W1:W2": [["환율(보조)"], ['=GOOGLEFINANCE("CURRENCY:USDKRW")']],
    }
    names = [n for n, _ in _HELPERS]
    first_c, last_c = HELPER_COLS[names[0]], HELPER_COLS[names[-1]]
    out[f"{first_c}{J_HEADER_ROW}:{last_c}{J_HEADER_ROW}"] = [names]
    out[f"{first_c}{J_FIRST}:{last_c}{J_LAST}"] = [
        [_render_helper(t, h) for _, t in _HELPERS] for h in range(J_FIRST, J_LAST + 1)
    ]
    return out


# ── 4. 투자현황·보유현황 수식 ──────────────────────────────────────────────────

_H = q(TAB_HOLDINGS)


def _h(c: str) -> str:
    """보유현황 열 범위(데이터 줄)."""
    return f"{_H}!${c}${H_FIRST}:${c}${H_LAST}"


def portfolio_layout() -> list[tuple[int, str, str, str]]:
    """투자현황 줄 배치 (행, A열 항목, B열 값·수식, 형식 종류 "krw"|"fx"|"pct"|"section"|"").

    A2·A3(총 투자금·종목당 계획금액)은 프로그램(data/sheets.read_portfolio)이 A열 이름으로 찾는 칸이라
    여기 넣지 않고 portfolio_values가 따로 다룬다(B2·B3 기존 값 보존).
    """
    J = q(TAB_JOURNAL)
    year_pnl = (
        f'=SUMIFS({J}!$R${J_FIRST}:$R${J_LAST},{J}!$O${J_FIRST}:$O${J_LAST},">="&DATE(YEAR(TODAY()),1,1),'
        f'{J}!$O${J_FIRST}:$O${J_LAST},"<="&DATE(YEAR(TODAY()),12,31))'
    )
    return [
        (5, "[자산]", "", "section"),
        (6, "환율(원/$)", '=GOOGLEFINANCE("CURRENCY:USDKRW")', "fx"),
        (7, "투자 중 원금(원)", f"=SUM({_h('I')})", "krw"),
        (8, "평가금액(원)", f"=SUM({_h('J')})", "krw"),
        (9, "남은 현금(원)", "=B2-B7", "krw"),
        (10, "투자 비중", '=IFERROR(B7/B2,"")', "pct"),
        (11, "전체 손익률", '=IF(N(B7)=0,"",B8/B7-1)', "pct"),  # 평가금액 ÷ 투자 중 원금 − 1, 원금 0이면 빈칸
        (13, "[코어·위성]", "", "section"),
        (14, "코어(QQQM) 평가금액(원)", f'=SUMIFS({_h("J")},{_h("D")},"코어")', "krw"),
        (15, "코어 비중", '=IFERROR(B14/B8,"")', "pct"),
        (16, "위성 평가금액(원)", "=B8-B14", "krw"),
        (17, "위성 비중", '=IFERROR(B16/B8,"")', "pct"),
        (19, "[세금 (근사값, 실제는 증권사 기준)]", "", "section"),
        (20, "올해 실현손익(원)", year_pnl, "krw"),
        (21, "250만 원 공제 남은 금액(원)", "=MAX(0,2500000-B20)", "krw"),
        (22, "예상 양도세(원)", "=MAX(0,B20-2500000)*22%", "krw"),
        (P_SECTOR_HEADER, "[업종별 비중 (위성 대비)]", "", "section"),
    ]


def sector_formula() -> str:
    """업종 | 비중(위성 평가금액 대비) 2열을 비중 큰 순서로 펼친다. 코어(QQQM·ETF)와 보유 0주는 뺀다."""
    query = (
        f"QUERY({_H}!$C${H_FIRST}:$J${H_LAST},"
        "\"select C, sum(J) where D = '위성' and E > 0 group by C order by sum(J) desc label sum(J) ''\",0)"
    )
    return f'=IFERROR(LET(q,{query},HSTACK(INDEX(q,0,1),INDEX(q,0,2)/$B$16)),"")'


def portfolio_values(b2_current: str, b3_current: str) -> tuple[dict[str, list[list[str]]], list[str]]:
    """투자현황 탭 값·수식(A·B 두 열). B2·B3은 비었을 때만 기본값을 넣는다(값이 있으면 그 칸은 결과에서 뺀다).

    입력: 지금 B2·B3 값(문자열)
    출력: ({A1 범위: 값}, 메모 목록)
    """
    notes = []
    out: dict[str, list[list[str]]] = {
        "A1:B1": [["항목", "값"]],
        "A2": [["총 투자금(원)"]],
        "A3": [["종목당 계획금액(원)"]],
    }
    for row, label, value, _ in portfolio_layout():
        out[f"A{row}:B{row}"] = [[label, value]]
    out[f"A{P_SECTOR_FIRST}"] = [[sector_formula()]]
    if not str(b2_current).strip():
        out["B2"] = [[str(P_DEFAULT_TOTAL)]]
        notes.append(f"투자현황 B2 비어 있음 → {P_DEFAULT_TOTAL:,}")
    else:
        notes.append(f"투자현황 B2 기존 값 유지 ({b2_current})")
    if not str(b3_current).strip():
        out["B3"] = [[str(P_DEFAULT_BUDGET)]]
        notes.append(f"투자현황 B3 비어 있음 → {P_DEFAULT_BUDGET:,}")
    else:
        notes.append(f"투자현황 B3 기존 값 유지 ({b3_current})")
    return out, notes


def portfolio_needs_reset(grid: list[list]) -> bool:
    """예전 배치(C열 이후에 값이 있음 — v3.1의 3열 표·종목별 현황)면 True. 새 배치는 A·B만 쓴다."""
    return any(str(v).strip() for row in grid for v in row[2:])


def holdings_values() -> dict[str, list[list[str]]]:
    """보유현황 탭: 체결 B열 고유 티커마다 한 줄. M열(숨김)은 현재 환율."""
    L = q(TAB_LIST)
    fb, fc, fd, fe, ff, fl = (f"{_F}!${FC[k]}$2:${FC[k]}${FILL_LAST}" for k in ("tk", "side", "qty", "price", "unit", "fx"))
    fx = f"${col_letter(H_FX_COL)}$2"
    fx_arr = f'IF({fl}="",{fx},{fl})'
    rows = []
    for r in range(H_FIRST, H_LAST + 1):
        a = f"$A{r}"
        rows.append([
            f'=IF({a}="","",IFERROR(VLOOKUP({a},{L}!$A:$B,2,FALSE),""))',
            f'=IF({a}="","",IFERROR(IF(VLOOKUP({a},{L}!$A:$C,3,FALSE)="","(미분류)",VLOOKUP({a},{L}!$A:$C,3,FALSE)),"(미분류)"))',
            f'=IF({a}="","",IF(OR(COUNTIFS({fb},{a},{ff},"대기자금")>0,C{r}="ETF"),"코어","위성"))',
            f'=IF({a}="","",SUMIFS({fd},{fb},{a},{fc},"매수")-SUMIFS({fd},{fb},{a},{fc},"매도"))',
            f'=IF({a}="","",IFERROR(SUMPRODUCT(({fb}={a})*({fc}="매수"),{fd},{fe})/SUMIFS({fd},{fb},{a},{fc},"매수"),""))',
            f'=IF({a}="","",IFERROR(GOOGLEFINANCE({a},"price"),""))',
            f'=IF(OR({a}="",F{r}="",G{r}=""),"",G{r}/F{r}-1)',
            f'=IF({a}="","",SUMPRODUCT(({fb}={a})*({fc}="매수"),{fd},{fe},{fx_arr})-SUMPRODUCT(({fb}={a})*({fc}="매도"),{fd},{fe},{fx_arr}))',
            f'=IF({a}="","",E{r}*N(G{r})*{fx})',
            f'=IF({a}="","",IFERROR(J{r}/SUM($J${H_FIRST}:$J${H_LAST}),""))',
        ])
    fxc = col_letter(H_FX_COL)
    return {
        "A1:K1": [HOLDINGS_HEADERS],
        f"A{H_FIRST}": [[f'=IFERROR(SORT(UNIQUE(FILTER(UPPER({fb}),{fb}<>""))),"")']],
        f"B{H_FIRST}:K{H_LAST}": rows,
        f"{fxc}1:{fxc}2": [["환율(보조)"], ['=GOOGLEFINANCE("CURRENCY:USDKRW")']],
    }


# ── 5. 관심 탭 ───────────────────────────────────────────────────────────────


def watch_values() -> dict[str, list[list[str]]]:
    """관심 탭 머리글·수식(B~G). A열(티커)은 사용자 입력이라 쓰지 않는다."""
    L = q(TAB_LIST)
    first, last = WATCH_ROWS
    rows = []
    for r in range(first, last + 1):
        a = f"$A{r}"
        rows.append([
            f'=IF({a}="","",IFERROR(VLOOKUP({a},{L}!$A:$B,2,FALSE),""))',
            f'=IF({a}="","",IFERROR(GOOGLEFINANCE({a},"price"),""))',
            f'=IF({a}="","",IFERROR(GOOGLEFINANCE({a},"changepct")/100,""))',
            f'=IF({a}="","",IFERROR(GOOGLEFINANCE({a},"price")/GOOGLEFINANCE({a},"high52")-1,""))',
            f'=IF({a}="","",IFERROR(C{r}/INDEX(GOOGLEFINANCE({a},"close",TODAY()-30),2,2)-1,""))',
            f'=IF({a}="","",IFERROR(C{r}/INDEX(GOOGLEFINANCE({a},"close",TODAY()-91),2,2)-1,""))',
        ])
    return {
        "A1:G1": [["티커", "종목명", "현재가($)", "오늘 등락률", "52주 최고가 대비", "1개월 수익률", "3개월 수익률"]],
        f"B{first}:G{last}": rows,
    }


# ── 6. 서식·검사·보호 요청 ────────────────────────────────────────────────────


def _grid(sheet_id: int, r1: int, r2: int | None, c1: int, c2: int) -> dict:
    """1부터 시작하는 행·열(끝 포함) → GridRange. r2=None이면 시트 끝까지."""
    g = {"sheetId": sheet_id, "startRowIndex": r1 - 1, "startColumnIndex": c1 - 1, "endColumnIndex": c2}
    if r2 is not None:
        g["endRowIndex"] = r2
    return g


def _validation(sheet_id, col, rule, strict=True, rows=1000):
    return {"setDataValidation": {"range": _grid(sheet_id, 2, rows, col, col), "rule": {**rule, "strict": strict, "showCustomUi": True}}}


def _list_rule(values):
    return {"condition": {"type": "ONE_OF_LIST", "values": [{"userEnteredValue": v} for v in values]}}


def _custom_rule(formula):
    return {"condition": {"type": "CUSTOM_FORMULA", "values": [{"userEnteredValue": formula}]}}


def _repeat(sheet_id, r1, r2, c1, c2, fmt: dict, fields: str):
    return {"repeatCell": {"range": _grid(sheet_id, r1, r2, c1, c2), "cell": {"userEnteredFormat": fmt}, "fields": fields}}


def _number(pattern, kind="NUMBER"):
    return {"numberFormat": {"type": kind, "pattern": pattern}}


def _center(sheet_id, r1, r2, c1, c2):
    """가로·세로 가운데 정렬 — 정렬 칸만 바꾸고 숫자 형식 등 다른 서식은 그대로 둔다."""
    return _repeat(sheet_id, r1, r2, c1, c2, {"horizontalAlignment": "CENTER", "verticalAlignment": "MIDDLE"},
                   "userEnteredFormat.horizontalAlignment,userEnteredFormat.verticalAlignment")


# 탭별 가운데 정렬 범위(1행부터 마지막 행, 1열부터 마지막 열) — 사용 중인 범위 전체(체결은 새로 적을 줄까지)
def center_ranges(fill_rows: int = 1000) -> dict[str, tuple[int, int]]:
    return {
        TAB_PORTFOLIO: (P_SECTOR_LAST, 2),
        TAB_HOLDINGS: (H_LAST, len(HOLDINGS_HEADERS)),
        TAB_JOURNAL: (J_LAST, len(JOURNAL_HEADERS)),
        TAB_FILLS: (fill_rows, len(FILLS_HEADERS)),
        TAB_PLAN: (1000, 12),
        TAB_WATCH: (WATCH_ROWS[1], 7),
        TAB_LIST: (300, 3),
    }


def _note(sheet_id, row, col, text):
    return {"updateCells": {"range": _grid(sheet_id, row, row, col, col), "rows": [{"values": [{"note": text}]}], "fields": "note"}}


def _freeze(sheet_id, rows=1):
    return {"updateSheetProperties": {"properties": {"sheetId": sheet_id, "gridProperties": {"frozenRowCount": rows}},
                                      "fields": "gridProperties.frozenRowCount"}}


def _width(sheet_id, c1, c2, px):
    return {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": c1 - 1, "endIndex": c2},
                                          "properties": {"pixelSize": px}, "fields": "pixelSize"}}


def _hide_cols(sheet_id, c1, c2):
    return {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": c1 - 1, "endIndex": c2},
                                          "properties": {"hiddenByUser": True}, "fields": "hiddenByUser"}}


def format_requests(ids: dict[str, int], fill_rows: int = 1000) -> list[dict]:
    """검사·색·형식·메모·너비·고정 요청(덮어써도 결과가 같은 것만). 해시로 건너뛰기를 판단한다."""
    f, j, p, w, pl = ids[TAB_FILLS], ids[TAB_JOURNAL], ids[TAB_PORTFOLIO], ids[TAB_WATCH], ids[TAB_PLAN]
    c = {h: i + 1 for i, h in enumerate(FILLS_HEADERS)}
    reqs: list[dict] = []
    # 체결 검사
    reqs += [
        _validation(f, c["날짜"], {"condition": {"type": "DATE_IS_VALID"}}, rows=fill_rows),
        _validation(f, c["티커"], {"condition": {"type": "ONE_OF_RANGE", "values": [{"userEnteredValue": f"={q(TAB_LIST)}!$A$2:$A"}]}}, rows=fill_rows),
        _validation(f, c["구분"], _list_rule(["매수", "매도"]), rows=fill_rows),
        _validation(f, c["수량"], _custom_rule(f"=AND(ISNUMBER({FC['qty']}2),{FC['qty']}2>0,{FC['qty']}2=INT({FC['qty']}2))"), rows=fill_rows),
        _validation(f, c["체결가($)"], {"condition": {"type": "NUMBER_GREATER", "values": [{"userEnteredValue": "0"}]}}, rows=fill_rows),
        _validation(f, c["차수"], _list_rule(UNITS), rows=fill_rows),
        _validation(f, c["손절가($)"], _custom_rule(f"=ISNUMBER({FC['stop']}2)"), rows=fill_rows),
        _validation(f, c["매도이유"], _list_rule(SELL_REASONS), rows=fill_rows),
        _validation(f, c["규칙대로"], _list_rule(["예", "아니오"]), rows=fill_rows),
        _validation(f, c["환율(원/$)"], _custom_rule(f"=ISNUMBER({FC['fx']}2)"), rows=fill_rows),
        _validation(f, c["수수료($)"], _custom_rule(f"=ISNUMBER({FC['fee']}2)"), rows=fill_rows),
    ]
    # 체결 색·형식
    reqs += [
        # 색: 필수 입력 연노랑 / 선택 입력 아주 연한 노랑 / 자동·선택(비우면 자동) 회색
        *[_repeat(f, 1, fill_rows, _fill_col(k), _fill_col(k), {"backgroundColor": color}, "userEnteredFormat.backgroundColor")
          for k, color in [("date", COLOR_INPUT), ("tk", COLOR_INPUT), ("side", COLOR_INPUT), ("qty", COLOR_INPUT),
                           ("price", COLOR_INPUT), ("unit", COLOR_OPTIONAL), ("name", COLOR_AUTO), ("stop", COLOR_OPTIONAL),
                           ("reason", COLOR_OPTIONAL), ("rule", COLOR_OPTIONAL), ("fx", COLOR_AUTO), ("fee", COLOR_AUTO),
                           ("memo", COLOR_OPTIONAL)]],
        _repeat(f, 2, fill_rows, _fill_col("date"), _fill_col("date"), _number("yyyy-mm-dd", "DATE"), "userEnteredFormat.numberFormat"),
        _repeat(f, 2, fill_rows, _fill_col("price"), _fill_col("price"), _number('"$"0.00'), "userEnteredFormat.numberFormat"),
        _repeat(f, 2, fill_rows, _fill_col("stop"), _fill_col("stop"), _number('"$"0.00'), "userEnteredFormat.numberFormat"),
        _repeat(f, 2, fill_rows, _fill_col("fx"), _fill_col("fx"), _number("#,##0.00"), "userEnteredFormat.numberFormat"),
        _repeat(f, 1, 1, 1, 13, {"textFormat": {"bold": True}}, "userEnteredFormat.textFormat.bold"),
        _note(f, 1, c["차수"], "살 때는 비우면 매수 순서대로 자동. 팔 때는 필수. 손절로 여러 차수를 팔면 차수별로 한 줄씩. QQQM 코어는 대기자금"),
        _note(f, 1, c["손절가($)"], "살 때 보고서의 손절가"),
        _note(f, 1, c["규칙대로"], "추천·손절 규칙대로 했나"),
        _note(f, 1, c["환율(원/$)"], "비워 두면 자동"),
        _note(f, 1, c["수수료($)"], "비워 두면 자동"),
        _freeze(f), _width(f, 1, 1, 90), _width(f, 2, 5, 70), _width(f, 6, 6, 60),
    ]
    # 매매일지
    jr = J_LAST
    reqs += [
        _freeze(j, J_HEADER_ROW),
        _repeat(j, 1, 1, 1, 6, {"textFormat": {"bold": True}}, "userEnteredFormat.textFormat.bold"),
        _repeat(j, J_HEADER_ROW, J_HEADER_ROW, 1, 21, {"textFormat": {"bold": True}, "backgroundColor": COLOR_AUTO},
                "userEnteredFormat.textFormat.bold,userEnteredFormat.backgroundColor"),
        _repeat(j, 2, 2, 3, 3, _number("0.0%", "PERCENT"), "userEnteredFormat.numberFormat"),
        _repeat(j, 2, 2, 4, 4, _number("0.00"), "userEnteredFormat.numberFormat"),
        _repeat(j, 2, 2, 5, 5, _number("#,##0"), "userEnteredFormat.numberFormat"),
        _repeat(j, 2, 2, 6, 6, _number("0.0%", "PERCENT"), "userEnteredFormat.numberFormat"),
        _repeat(j, J_FIRST, jr, 6, 6, _number("yyyy-mm-dd", "DATE"), "userEnteredFormat.numberFormat"),
        _repeat(j, J_FIRST, jr, 15, 15, _number("yyyy-mm-dd", "DATE"), "userEnteredFormat.numberFormat"),
        *[_repeat(j, J_FIRST, jr, col, col, _number('"$"0.00'), "userEnteredFormat.numberFormat") for col in (7, 10, 11, 13, 16)],
        *[_repeat(j, J_FIRST, jr, col, col, _number("#,##0"), "userEnteredFormat.numberFormat") for col in (9, 18)],
        _repeat(j, J_FIRST, jr, 14, 14, _number("0.0%", "PERCENT"), "userEnteredFormat.numberFormat"),
        _repeat(j, J_FIRST, jr, 19, 19, _number("0.00"), "userEnteredFormat.numberFormat"),
        _note(j, J_HEADER_ROW, 18, "근사값, 실제는 증권사 기준"),
        _note(j, J_HEADER_ROW, 12, "같은 티커·같은 차수, 매수일 이후 첫 매도 줄과 짝짓는다"),
        _hide_cols(j, J_HIDE_FROM, J_HELPER_START + len(_HELPERS) - 1),
    ]
    # 투자현황 — A·B 두 열만. 형식은 portfolio_layout의 종류대로
    kinds = {"krw": _number("#,##0"), "fx": _number("#,##0.00"), "pct": _number("0.0%", "PERCENT")}
    reqs += [
        _freeze(p),
        _repeat(p, 1, 1, 1, 2, {"textFormat": {"bold": True}, "backgroundColor": COLOR_AUTO},
                "userEnteredFormat.textFormat.bold,userEnteredFormat.backgroundColor"),
        _repeat(p, 2, 3, 2, 2, {"backgroundColor": COLOR_INPUT}, "userEnteredFormat.backgroundColor"),
        _repeat(p, 2, 3, 2, 2, kinds["krw"], "userEnteredFormat.numberFormat"),
        _repeat(p, P_SECTOR_FIRST, P_SECTOR_LAST, 2, 2, kinds["pct"], "userEnteredFormat.numberFormat"),
        _note(p, 2, 1, "프로그램(계획 자동 기록 v2)이 A열 이름으로 이 칸을 읽는다 — 이름 바꾸지 말 것"),
        _width(p, 1, 1, 260), _width(p, 2, 2, 140),
        {"updateDimensionProperties": {"range": {"sheetId": p, "dimension": "COLUMNS", "startIndex": 2, "endIndex": P_CLEAR_COLS},
                                       "properties": {"hiddenByUser": False}, "fields": "hiddenByUser"}},
    ]
    for row, _, _, kind in portfolio_layout():
        if kind == "section":
            reqs.append(_repeat(p, row, row, 1, 2, {"textFormat": {"bold": True}, "backgroundColor": COLOR_AUTO},
                                "userEnteredFormat.textFormat.bold,userEnteredFormat.backgroundColor"))
        elif kind:
            reqs.append(_repeat(p, row, row, 2, 2, kinds[kind], "userEnteredFormat.numberFormat"))
    # 보유현황
    h = ids[TAB_HOLDINGS]
    reqs += [
        _freeze(h),
        _repeat(h, 1, 1, 1, len(HOLDINGS_HEADERS), {"textFormat": {"bold": True}, "backgroundColor": COLOR_AUTO},
                "userEnteredFormat.textFormat.bold,userEnteredFormat.backgroundColor"),
        *[_repeat(h, H_FIRST, H_LAST, col, col, _number('"$"0.00'), "userEnteredFormat.numberFormat") for col in (6, 7)],
        *[_repeat(h, H_FIRST, H_LAST, col, col, _number("0.0%", "PERCENT"), "userEnteredFormat.numberFormat") for col in (8, 11)],
        _repeat(h, H_FIRST, H_LAST, 9, 10, _number("#,##0"), "userEnteredFormat.numberFormat"),
        _note(h, 1, 4, "코어 = 체결 차수 \"대기자금\"이 있거나 목록 업종이 ETF인 종목(QQQM). 나머지는 위성"),
        _hide_cols(h, H_FX_COL, H_FX_COL), _width(h, 2, 2, 200),
    ]
    # 관심·계획
    reqs += [
        _freeze(w),
        _validation(w, 1, {"condition": {"type": "ONE_OF_RANGE", "values": [{"userEnteredValue": f"={q(TAB_LIST)}!$A$2:$A"}]}}, rows=WATCH_ROWS[1]),
        _repeat(w, 2, WATCH_ROWS[1], 1, 1, {"backgroundColor": COLOR_INPUT}, "userEnteredFormat.backgroundColor"),
        _repeat(w, 2, WATCH_ROWS[1], 3, 3, _number('"$"0.00'), "userEnteredFormat.numberFormat"),
        _repeat(w, 2, WATCH_ROWS[1], 4, 7, _number("0.0%", "PERCENT"), "userEnteredFormat.numberFormat"),
        _repeat(w, 1, 1, 1, 7, {"textFormat": {"bold": True}}, "userEnteredFormat.textFormat.bold"),
        _freeze(pl),
        *[_center(ids[tab], 1, last_row, 1, last_col) for tab, (last_row, last_col) in center_ranges(fill_rows).items()],
        _repeat(pl, 2, 1000, 2, 2, _number("#,##0"), "userEnteredFormat.numberFormat"),
        # 기준가는 소수 둘째 자리까지 보이게 — read_sheets가 보이는 값을 읽으므로 178.41이 178로 읽히지 않게 한다
        _repeat(pl, 2, 1000, 4, 4, _number("0.00"), "userEnteredFormat.numberFormat"),
    ]
    return reqs


def conditional_rules(ids: dict[str, int]) -> list[tuple[int, str, dict]]:
    """(sheetId, 식별용 수식, addConditionalFormatRule 요청). 같은 수식 규칙이 있으면 건너뛰고,
    이 스크립트가 관리하는 탭(MANAGED_CF_TABS)에서 목록에 없는 규칙(예전 배치의 규칙)은 지운다."""
    f, p, h = ids[TAB_FILLS], ids[TAB_PORTFOLIO], ids[TAB_HOLDINGS]
    rules = [
        (f, '=AND($C2="매도",$F2="")', _grid(f, 2, 1000, 6, 6), {"backgroundColor": COLOR_RED}),
        (p, f"=AND(ISNUMBER($B{P_SECTOR_FIRST}),$B{P_SECTOR_FIRST}>0.3)", _grid(p, P_SECTOR_FIRST, P_SECTOR_LAST, 2, 2),
         {"textFormat": {"foregroundColor": COLOR_RED_TEXT, "bold": True}}),
        (h, f'=AND($A{H_FIRST}<>"",$E{H_FIRST}=0)', _grid(h, H_FIRST, H_LAST, 1, len(HOLDINGS_HEADERS)),
         {"textFormat": {"foregroundColor": COLOR_GRAY_TEXT}}),
    ]
    out = []
    for sid, formula, rng, fmt in rules:
        out.append((sid, formula, {"addConditionalFormatRule": {"index": 0, "rule": {
            "ranges": [rng], "booleanRule": {"condition": _custom_rule(formula)["condition"], "format": fmt}}}}))
    return out


MANAGED_CF_TABS = (TAB_FILLS, TAB_PORTFOLIO, TAB_HOLDINGS)


def protection_specs(ids: dict[str, int], plan_formula_cols: list[int]) -> list[tuple[str, dict]]:
    """(설명, protectedRange) — 모두 "경고만 표시"(warningOnly). 설명이 같은 보호가 있으면 건너뛴다."""
    specs = [
        ("목록 전체", {"range": {"sheetId": ids[TAB_LIST]}}),
        ("매매일지 전체", {"range": {"sheetId": ids[TAB_JOURNAL]}}),
        ("보유현황 전체", {"range": {"sheetId": ids[TAB_HOLDINGS]}}),
        ("투자현황 수식 칸", {"range": {"sheetId": ids[TAB_PORTFOLIO]},
                         "unprotectedRanges": [_grid(ids[TAB_PORTFOLIO], 2, 3, 2, 2)]}),
        ("체결 1행", {"range": _grid(ids[TAB_FILLS], 1, 1, 1, 26)}),
        ("체결 G열 종목명", {"range": _grid(ids[TAB_FILLS], 2, None, 7, 7)}),
        ("계획 1행", {"range": _grid(ids[TAB_PLAN], 1, 1, 1, 26)}),
        ("관심 1행", {"range": _grid(ids[TAB_WATCH], 1, 1, 1, 26)}),
    ]
    for col in plan_formula_cols:
        specs.append((f"계획 수식 열 {col_letter(col)}", {"range": _grid(ids[TAB_PLAN], 2, None, col, col)}))
    return [(PROTECT_PREFIX + name, {**spec, "description": PROTECT_PREFIX + name, "warningOnly": True}) for name, spec in specs]


def requests_hash(reqs: list[dict]) -> str:
    return hashlib.sha256(json.dumps(reqs, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


# ── 7. 값 비교 ───────────────────────────────────────────────────────────────


def parse_a1(a1: str) -> tuple[int, int]:
    """"B12" → (행 12, 열 2)."""
    m = re.fullmatch(r"([A-Z]+)(\d+)", a1)
    if not m:
        raise ValueError(a1)
    return int(m.group(2)), col_index(m.group(1))


def changed_ranges(current: list[list], wanted: dict[str, list[list[str]]]) -> dict[str, list[list[str]]]:
    """wanted 중 지금 값(수식 그대로 읽은 격자)과 다른 범위만 돌려준다."""
    out = {}
    for rng, values in wanted.items():
        r0, c0 = parse_a1(rng.split(":")[0])
        same = True
        for i, row in enumerate(values):
            for j, v in enumerate(row):
                r, c = r0 + i, c0 + j
                have = current[r - 1][c - 1] if r - 1 < len(current) and c - 1 < len(current[r - 1]) else ""
                if _str(have) != _str(v):
                    same = False
                    break
            if not same:
                break
        if not same:
            out[rng] = values
    return out


# ── 8. 실행 ─────────────────────────────────────────────────────────────────


class SheetSetup:
    """보기·적용 흐름. client는 gspread.Client와 같은 인터페이스(테스트는 가짜)."""

    def __init__(self, client, sheet_id: str, list_rows_provider, apply: bool, out=print, backup_dir: Path = OUTPUT_DIR):
        self.client = client
        self.sheet_id = sheet_id
        self.list_rows_provider = list_rows_provider  # (fills_tickers) -> 목록 탭 값
        self.apply = apply
        self.out = out
        self.backup_dir = backup_dir
        self.changes: list[str] = []
        self.oddities: list[str] = []
        self.backup_path: Path | None = None

    # ── 읽기 ──
    def _meta(self):
        return self.sh.fetch_sheet_metadata(params={
            "fields": "properties(locale,timeZone),developerMetadata,"
                      "sheets(properties,protectedRanges,conditionalFormats,data.columnMetadata.hiddenByUser)"})

    def _tabs(self) -> dict[str, object]:
        return {ws.title: ws for ws in self.sh.worksheets()}

    def _formula_grid(self, ws) -> list[list]:
        return ws.get_all_values(value_render_option="FORMULA")

    # ── 쓰기(보기 모드에서는 기록만) ──
    def _batch(self, reqs: list[dict], what: str):
        if not reqs:
            return
        self.changes.append(f"{what} ({len(reqs)}건)")
        if self.apply:
            self.sh.batch_update({"requests": reqs})

    def _values(self, tab: str, ranges: dict[str, list[list[str]]], what: str):
        if not ranges:
            return
        cells = sum(len(r) * max((len(x) for x in r), default=0) for r in ranges.values())
        self.changes.append(f"{what}: {', '.join(list(ranges)[:6])}{' …' if len(ranges) > 6 else ''} ({cells}칸)")
        if self.apply:
            self.sh.values_batch_update({"valueInputOption": "USER_ENTERED", "data": [
                {"range": f"{q(tab)}!{rng}", "values": vals} for rng, vals in ranges.items()]})

    def backup(self) -> Path:
        """모든 탭 값(보이는 값)·수식을 JSON으로 저장."""
        data = {"sheet_id_tail": self.sheet_id[-4:], "saved_at": datetime.now().isoformat(timespec="seconds"), "tabs": {}}
        for title, ws in self._tabs().items():
            data["tabs"][title] = {"values": ws.get_all_values(), "formulas": self._formula_grid(ws)}
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        path = self.backup_dir / f"sheet_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    def run(self) -> dict:
        self.sh = self.client.open_by_key(self.sheet_id)
        self.out(f"[시트] {mask_id(self.sheet_id)} · {'적용(--apply)' if self.apply else '보기 모드 — 쓰기 없음'}")
        tabs = self._tabs()
        if TAB_FILLS not in tabs or TAB_PLAN not in tabs:
            raise RuntimeError("체결·계획 탭이 없습니다 — 원래 시트가 맞는지 확인하세요.")
        before_fills = self._formula_grid(tabs[TAB_FILLS])
        before_plan = self._formula_grid(tabs[TAB_PLAN])
        if self.apply:
            self.backup_path = self.backup()
            self.out(f"[백업] {self.backup_path}")

        # 1) 없는 탭 만들기
        for title, (rows, cols) in NEW_TABS.items():
            if title not in tabs:
                self.changes.append(f"탭 만들기: {title}")
                if self.apply:
                    self.sh.add_worksheet(title=title, rows=rows, cols=cols)
        tabs = self._tabs() if self.apply else tabs
        ids = {t: ws.id for t, ws in tabs.items()}
        for t in NEW_TABS:  # 보기 모드: 아직 없는 탭은 가상의 id
            ids.setdefault(t, -1)

        # 2) 체결 열 이동·머리글
        header = before_fills[0] if before_fills else []
        ops, new_header = plan_fill_columns(header)
        self._batch(column_op_requests(ids[TAB_FILLS], ops), "체결 열 이동·삽입")
        fills_after = apply_column_ops(before_fills, ops) if before_fills else [[]]
        self._values(TAB_FILLS, changed_ranges(fills_after, {"A1:M1": [FILLS_HEADERS]}), "체결 머리글")
        # 탭 이름은 따옴표로 — 시트가 저장할 때 '목록'으로 바꿔 두므로 같은 글자여야 다시 쓰지 않는다
        g2 = f'=ARRAYFORMULA(IF(B2:B="","",IFERROR(VLOOKUP(B2:B,{q(TAB_LIST)}!A:B,2,FALSE),"?")))'
        self._values(TAB_FILLS, changed_ranges(fills_after, {"G2": [[g2]]}), "체결 G2 종목명 수식")

        # 3) 목록
        fills_tickers = [str(r[1]).strip().upper() for r in fills_after[1:] if len(r) > 1 and str(r[1]).strip()]
        list_rows = self.list_rows_provider(fills_tickers)
        list_now = self._formula_grid(tabs[TAB_LIST]) if TAB_LIST in tabs else []
        if [[_str(x) for x in r[:3]] for r in list_now if any(str(x).strip() for x in r)] != list_rows:
            self._values(TAB_LIST, {f"A1:C{len(list_rows)}": list_rows}, f"목록 새로 쓰기({len(list_rows) - 1}종목)")
            stale = len(list_now) - len(list_rows)
            if stale > 0:  # 예전 목록이 더 길면 남은 줄을 비운다(목록 탭은 프로그램 소유)
                self._values(TAB_LIST, {f"A{len(list_rows) + 1}:C{len(list_now)}": [["", "", ""]] * stale}, "목록 남은 줄 비우기")
        known = {r[0] for r in list_rows[1:]}
        for i, r in enumerate(fills_after[1:], start=2):
            t = str(r[1]).strip().upper() if len(r) > 1 else ""
            if t and t not in known:
                self.oddities.append(f"체결 {i}행 티커 {t}: 목록에 없음")
        self._check_numbers(fills_after)

        # 4) 매매일지
        journal_now = self._formula_grid(tabs[TAB_JOURNAL]) if TAB_JOURNAL in tabs else []
        self._values(TAB_JOURNAL, changed_ranges(journal_now, journal_values()), "매매일지 수식")

        # 5) 보유현황
        hold_now = self._formula_grid(tabs[TAB_HOLDINGS]) if TAB_HOLDINGS in tabs else []
        self._values(TAB_HOLDINGS, changed_ranges(hold_now, holdings_values()), "보유현황 수식")

        # 6) 투자현황 (A·B 두 열, B2·B3은 비었을 때만). 예전 배치(C열 이후 사용)면 1~3행 A·B를 뺀 나머지를
        #    한 번 지우고(값·서식·메모) 새로 쓴다 — 탭은 지우지 않고 B2·B3 입력값은 그대로 둔다.
        port_now = self._formula_grid(tabs[TAB_PORTFOLIO]) if TAB_PORTFOLIO in tabs else []
        cell = lambda g, r, c: _str(g[r - 1][c - 1]) if r - 1 < len(g) and c - 1 < len(g[r - 1]) else ""  # noqa: E731
        if portfolio_needs_reset(port_now):
            pid = ids[TAB_PORTFOLIO]
            clear = "userEnteredValue,userEnteredFormat,note"
            self._batch([
                {"updateCells": {"range": _grid(pid, 4, None, 1, P_CLEAR_COLS), "fields": clear}},
                {"updateCells": {"range": _grid(pid, 1, 3, 3, P_CLEAR_COLS), "fields": clear}},
            ], "투자현황 예전 배치 지우기(4행 이하·C열 이후, B2·B3 보존)")
            port_now = [list(r[:2]) for r in port_now[:3]]
        port_vals, port_notes = portfolio_values(cell(port_now, 2, 2), cell(port_now, 3, 2))
        self._values(TAB_PORTFOLIO, changed_ranges(port_now, port_vals), "투자현황 수식")
        self.out("\n".join(f"  · {n}" for n in port_notes))

        # 7) 계획 A2 오타
        a2 = cell(before_plan, 2, 1)
        if a2 == "OLED":
            self._values(TAB_PLAN, {"A2": [["ODFL"]]}, "계획 A2 OLED → ODFL")
        elif a2 != "ODFL":
            self.oddities.append(f"계획 A2가 {a2!r} — OLED가 아니라 고치지 않음")

        # 8) 관심
        watch_now = self._formula_grid(tabs[TAB_WATCH]) if TAB_WATCH in tabs else []
        self._values(TAB_WATCH, changed_ranges(watch_now, watch_values()), "관심 수식")

        # 9) 서식·조건부 서식·보호·탭 순서
        meta = self._meta()
        self._format(meta, ids, len(before_fills))
        self._conditional(meta, ids)
        plan_cols = sorted({j + 1 for row in before_plan[1:] for j, v in enumerate(row) if str(v).startswith("=")})
        self._protect(meta, ids, plan_cols)
        self._tab_order(meta, ids)
        self._time_zone(meta)
        self._journal_hidden(meta, ids)

        result = {"changes": self.changes, "oddities": self.oddities, "backup": self.backup_path}
        self._print_summary()
        if self.apply:
            result["verify"] = self.verify(before_fills, before_plan)
        return result

    def _check_numbers(self, grid):
        c = {h: i for i, h in enumerate(FILLS_HEADERS)}
        for i, row in enumerate(grid[1:], start=2):
            for name in ("수량", "체결가($)", "손절가($)", "환율(원/$)", "수수료($)"):
                k = c[name]
                v = row[k] if k < len(row) else ""
                if str(v).strip() == "":
                    continue
                try:
                    float(str(v).replace(",", ""))
                except ValueError:
                    self.oddities.append(f"체결 {i}행 {name}: 숫자 아님 ({v!r})")

    def _format(self, meta, ids, n_fill_rows):
        reqs = format_requests(ids)
        digest = requests_hash(reqs)
        found = [m for m in meta.get("developerMetadata", []) if m.get("metadataKey") == FORMAT_HASH_KEY]
        if found and found[0].get("metadataValue") == digest:
            return
        reqs = list(reqs)
        for m in found:
            reqs.append({"deleteDeveloperMetadata": {"dataFilter": {"developerMetadataLookup": {"metadataId": m["metadataId"]}}}})
        reqs.append({"createDeveloperMetadata": {"developerMetadata": {
            "metadataKey": FORMAT_HASH_KEY, "metadataValue": digest, "location": {"spreadsheet": True}, "visibility": "DOCUMENT"}}})
        self._batch(reqs, "검사·색·형식·메모·너비·고정")

    def _conditional(self, meta, ids):
        wanted = conditional_rules(ids)
        wanted_by_sheet: dict[int, set] = {}
        for sid, formula, _ in wanted:
            wanted_by_sheet.setdefault(sid, set()).add(formula)
        managed = {ids[t] for t in MANAGED_CF_TABS if ids.get(t, -1) >= 0}
        existing: dict[int, set] = {}
        deletes = []
        for s in meta.get("sheets", []):
            sid = s["properties"]["sheetId"]
            for i, rule in enumerate(s.get("conditionalFormats", [])):
                vals = (((rule.get("booleanRule") or {}).get("condition") or {}).get("values") or [{}])
                formula = vals[0].get("userEnteredValue")
                if sid in managed and (formula not in wanted_by_sheet.get(sid, set()) or formula in existing.get(sid, set())):
                    deletes.append({"deleteConditionalFormatRule": {"sheetId": sid, "index": i}})
                    continue
                existing.setdefault(sid, set()).add(formula)
        # 같은 탭에서 뒤 번호부터 지워야 앞 번호가 밀리지 않는다
        deletes.sort(key=lambda r: (r["deleteConditionalFormatRule"]["sheetId"], -r["deleteConditionalFormatRule"]["index"]))
        self._batch(deletes, "예전 조건부 서식 지우기")
        reqs = [req for sid, formula, req in wanted if formula not in existing.get(sid, set())]
        self._batch(reqs, "조건부 서식")

    def _protect(self, meta, ids, plan_cols):
        have = {p.get("description") for s in meta.get("sheets", []) for p in s.get("protectedRanges", [])}
        reqs = [{"addProtectedRange": {"protectedRange": spec}} for desc, spec in protection_specs(ids, plan_cols) if desc not in have]
        self._batch(reqs, "보호(경고만 표시)")

    def _tab_order(self, meta, ids):
        props = {s["properties"]["title"]: s["properties"] for s in meta.get("sheets", [])}
        reqs = []
        for idx, title in enumerate(TAB_ORDER):
            p = props.get(title, {})
            if p.get("index") != idx and ids.get(title, -1) >= 0:
                reqs.append({"updateSheetProperties": {"properties": {"sheetId": ids[title], "index": idx}, "fields": "index"}})
            elif title not in props and not self.apply:
                self.changes.append(f"탭 순서: {title} → {idx + 1}번째")
        if ids.get(TAB_LIST, -1) >= 0 and not props.get(TAB_LIST, {}).get("hidden"):
            reqs.append({"updateSheetProperties": {"properties": {"sheetId": ids[TAB_LIST], "hidden": True}, "fields": "hidden"}})
        self._batch(reqs, "탭 순서·목록 숨김")

    def _time_zone(self, meta):
        if (meta.get("properties") or {}).get("timeZone") != TIME_ZONE:
            self._batch([{"updateSpreadsheetProperties": {"properties": {"timeZone": TIME_ZONE}, "fields": "timeZone"}}],
                        f"스프레드시트 시간대 → {TIME_ZONE}")

    def _journal_hidden(self, meta, ids):
        """매매일지 보조 열(W~)이 실제로 숨겨져 있는지 시트에서 확인하고, 아니면 숨긴다(해시와 무관)."""
        sheet = next((s for s in meta.get("sheets", []) if s["properties"].get("title") == TAB_JOURNAL), None)
        if sheet is None or ids.get(TAB_JOURNAL, -1) < 0:
            return
        cols = ((sheet.get("data") or [{}])[0].get("columnMetadata")) or []
        last = J_HELPER_START + len(_HELPERS) - 1
        shown = [col_letter(c) for c in range(J_HIDE_FROM, last + 1)
                 if c - 1 >= len(cols) or not cols[c - 1].get("hiddenByUser")]
        if shown:
            self._batch([_hide_cols(ids[TAB_JOURNAL], J_HIDE_FROM, last)], f"매매일지 보조 열 숨김 ({shown[0]}~{shown[-1]} 중 {len(shown)}열)")

    def _print_summary(self):
        self.out("[바꿀 내용]" if not self.apply else "[바꾼 내용]")
        self.out("\n".join(f"  - {c}" for c in self.changes) if self.changes else "  (없음 — 이미 정리된 상태)")
        if self.oddities:
            self.out("[이상한 값]")
            self.out("\n".join(f"  - {o}" for o in self.oddities))

    def verify(self, before_fills, before_plan) -> dict:
        """적용 후 다시 읽어 체결 데이터(머리글 이름 기준)·계획 데이터(A2 제외)가 그대로인지 확인한다."""
        tabs = self._tabs()
        after_fills = self._formula_grid(tabs[TAB_FILLS])
        after_plan = self._formula_grid(tabs[TAB_PLAN])
        problems = []
        b_rows, a_rows = rows_by_header(before_fills), rows_by_header(after_fills)
        for i, b in enumerate(b_rows):
            a = a_rows[i] if i < len(a_rows) else {}
            for k, v in b.items():
                if k == _FILL_KEYS["name"]:  # 종목명(G2)은 이 스크립트가 쓰는 수식 열 — 사용자 데이터 아님
                    continue
                if _str(a.get(k, "")) != v:
                    problems.append(f"체결 데이터 {i + 2}행 {k}: {v!r} → {a.get(k)!r}")
        if len(a_rows) != len(b_rows):
            problems.append(f"체결 데이터 줄 수 {len(b_rows)} → {len(a_rows)}")
        for r in range(max(len(before_plan), len(after_plan))):
            rb = before_plan[r] if r < len(before_plan) else []
            ra = after_plan[r] if r < len(after_plan) else []
            for c in range(max(len(rb), len(ra))):
                vb, va = (_str(rb[c]) if c < len(rb) else ""), (_str(ra[c]) if c < len(ra) else "")
                if vb != va and not (r == 1 and c == 0 and vb == "OLED" and va == "ODFL"):
                    problems.append(f"계획 {col_letter(c + 1)}{r + 1}: {vb!r} → {va!r}")
        if problems:
            self.out("[비교] *** 적용 전과 다른 데이터가 있습니다 ***")
            self.out("\n".join(f"  - {p}" for p in problems[:30]))
            self.out(f"  백업: {self.backup_path}")
        else:
            self.out(f"[비교] 체결 데이터 {len(b_rows)}줄 그대로, 계획은 A2 한 칸 외 변경 없음")
        return {"problems": problems, "fill_rows": len(b_rows)}


# ── 9. 실제 실행용 입력 ───────────────────────────────────────────────────────


def real_list_rows(fills_tickers) -> list[list[str]]:
    """나스닥100 구성 종목(data/universe) + 한글 이름(name_kr.csv) + 업종(data/sectors 캐시)."""
    from data.sectors import get_sectors
    from data.universe import get_universe

    import pandas as pd

    universe = get_universe()
    # name_kr.csv 전체(구성 종목이 아닌 QQQM 등 포함) + 구성 종목 조인 결과
    kr = pd.read_csv(ROOT / "data" / "name_kr.csv", dtype=str).fillna("")
    name_map = {t.strip(): n.strip() for t, n in zip(kr["ticker"], kr["name_kr"])}
    name_map.update({t: n for t, n in zip(universe["ticker"], universe["name_kr"]) if n})
    tickers = sorted({*universe["ticker"], "QQQM", *fills_tickers})
    sector_map, failed = get_sectors(tickers)
    failed = [t for t in failed if t not in SPECIAL_SECTORS]  # QQQM은 "ETF"로 채운다
    if failed:
        print(f"[목록] 업종을 못 받은 종목 {len(failed)}개(빈칸): {', '.join(sorted(failed))}")
    return build_list_rows(universe["ticker"], name_map, sector_map, fills_tickers)


def read_sheets_check(sheet_id: str) -> None:
    """적용 후 프로그램 경로(data/sheets.read_sheets)로 시트를 읽어 파싱 결과·오류를 출력한다."""
    import os

    import yaml

    from data import sheets

    os.environ["GOOGLE_SHEETS_ID"] = sheet_id
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    result = sheets.read_sheets(cfg=cfg)
    print("[read_sheets] 계획:")
    print(result.plan_df.to_string(index=False) if not result.plan_df.empty else "  (없음)")
    print("[read_sheets] 체결:")
    df = result.fills.df
    print(df[["date", "ticker", "side", "qty", "price", "unit", "fx_rate", "fee_usd"]].to_string(index=False) if not df.empty else "  (없음)")
    errors = list(result.plan_errors) + list(result.fills.errors)
    print("[read_sheets] 오류·경고:" if errors else "[read_sheets] 오류·경고 없음")
    for e in errors:
        print(f"  - {e}")
    if hasattr(sheets, "read_portfolio"):
        portfolio, warnings = sheets.read_portfolio(result.client)
        print(f"[read_portfolio] 총 투자금 {portfolio['total_krw']}, 종목당 계획금액 {portfolio['per_ticker_budget_krw']}"
              + (f", 경고 {warnings}" if warnings else ""))


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="구글 시트 정리 (시트 재설계안 v3.1)")
    parser.add_argument("sheet_id", nargs="?", help="시트 ID (생략하면 .env의 GOOGLE_SHEETS_ID)")
    parser.add_argument("--apply", action="store_true", help="실제로 적용 (없으면 보기만)")
    args = parser.parse_args(argv)

    from data import sheets

    sheet_id = args.sheet_id or sheets._sheet_id()
    setup = SheetSetup(sheets.get_client(), sheet_id, real_list_rows, apply=args.apply)
    result = setup.run()
    if args.apply:
        read_sheets_check(sheet_id)
        if result["verify"]["problems"]:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
