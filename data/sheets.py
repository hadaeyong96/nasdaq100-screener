"""구글 스프레드시트 입력 (계획·체결) — 라이브 어드바이저 1단계
(docs/design/live_advisor.md 1번).

data/fills.xlsx를 대체한다. 서비스 계정으로 두 탭을 읽는다 (2026-09-30 주식 수
기준으로 개편한 시트 헤더):
  - "계획": 티커, 계획금액, 등록일, 기준가($), 환율, 1차(주), 2차(주), 3차(주),
    합계(주), 보유(주), 남은(주), 메모 (계획금액은 원화, "3,000,000"처럼 쉼표
    가능. 기준가는 선택 — 있으면 core.sizing.plan_tranche_qty가 그 값과 그날
    환율로 고정 총 주수를 계산해 시트의 1차~남은(주) 수식과 같은 결과를 낸다.
    환율·1차(주)~남은(주)는 시트 수식이라 읽지 않는다)
  - "체결": 날짜, 종목, 매수매도, 수량, 체결가, 차수, 메모
수식 때문에 1000행까지 생기는 빈 줄은 건너뛴다. 헤더의 괄호 설명("계획금액(원)" 등)은
무시하고 비교하며, 차수가 비면 data/fills.py가 매수 순서로 배정한다. 환율·수수료 열이
있으면 쓰고, 없거나 비면 체결일 원/달러 종가(data/fx)와 config 수수료율
(backtest.costs.commission_buy_pct/sell_pct)로 채운다. 행 검증은 data/fills.py의
parse_fill_records·parse_plan_records를 그대로 쓴다.

인증: 환경변수 GOOGLE_SERVICE_ACCOUNT_JSON — 값이 "{"로 시작하면 JSON
문자열 그대로(GitHub Actions Secret), 아니면 JSON 키 파일 경로로 취급한다
(로컬은 .env에 경로). 읽을 시트 ID는 환경변수 GOOGLE_SHEETS_ID.
인증 정보의 값은 어떤 경우에도 출력·로그하지 않는다 (CLAUDE.md 보안).

"투자현황" 탭(총 투자금·종목당 계획금액)은 계획 탭 자동 기록 v2(docs/design/auto_plan.md)가
read_portfolio로 읽기만 한다.

쓰기: 계획 탭 자동 기록(write_auto_plan, docs/design/auto_plan.md)만 — 계획 탭의 입력 칸 5개
(티커·계획금액·등록일·기준가·메모)만 셀 단위로 쓰고, 다른 탭 쓰기는 코드로 막는다.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from data.fills import FillsResult, _normalize_header, fill_missing_costs, parse_fill_records, parse_plan_records

ROOT = Path(__file__).resolve().parents[1]
# notify/telegram.py와 같은 패턴: 이 모듈만 단독으로 쓰는 스크립트(scripts/drive_sync.py
# 등, notify.telegram을 안 거치는 경로)도 .env를 확실히 읽도록 여기서 직접 불러온다.
# override=True: 이 프로세스에 같은 이름의 빈 환경변수가 이미 있어도 .env 값으로 덮는다.
load_dotenv(ROOT / ".env", override=True)

# 계획 탭 자동 기록(docs/design/auto_plan.md)으로 시트 쓰기 권한이 필요하다. 쓰기는
# write_auto_plan → 계획 탭만 허용(_plan_worksheet·_assert_plan_tab이 코드로 막는다).
_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

SHEET_PLAN = "계획"
SHEET_FILLS = "체결"
SHEET_PORTFOLIO = "투자현황"

_PLAN_COLUMNS = ["ticker", "budget_krw", "ref_price", "memo"]


class SheetsConfigError(RuntimeError):
    """인증 정보나 시트 ID 환경변수가 없거나 잘못됐을 때."""


@dataclass
class SheetsResult:
    """read_sheets()의 결과.

    plan_df: DataFrame(ticker, budget_krw, ref_price, memo) — 계획 탭의 유효한 행만.
    plan_errors: 계획 탭의 건너뛴 줄 사유 ("계획 기록 오류: N번째 줄 - ...").
    fills: 체결 탭을 data.fills.parse_fill_records로 파싱한 FillsResult.
    read_at_kst: 두 탭을 읽은 시각(KST, tz-aware Timestamp).
    client: 읽을 때 쓴 클라이언트(같은 실행에서 계획 자동 기록에 그대로 쓴다).
    """

    plan_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=_PLAN_COLUMNS))
    plan_errors: list[str] = field(default_factory=list)
    fills: FillsResult = field(default_factory=FillsResult)
    read_at_kst: pd.Timestamp | None = None
    client: object = None


def load_credentials_info() -> dict:
    """GOOGLE_SERVICE_ACCOUNT_JSON을 읽어 서비스 계정 정보 dict로 만든다.
    "{"로 시작하면 JSON 문자열(Actions Secret), 아니면 JSON 키 파일 경로(로컬)."""
    raw = (os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON") or "").strip()
    if not raw:
        raise SheetsConfigError("환경변수 GOOGLE_SERVICE_ACCOUNT_JSON이 없습니다.")
    if raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SheetsConfigError(f"GOOGLE_SERVICE_ACCOUNT_JSON이 올바른 JSON이 아닙니다: {exc}") from exc
    path = Path(raw)
    if not path.exists():
        raise SheetsConfigError(f"GOOGLE_SERVICE_ACCOUNT_JSON이 가리키는 키 파일이 없습니다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sheet_id() -> str:
    sheet_id = (os.environ.get("GOOGLE_SHEETS_ID") or "").strip()
    if not sheet_id:
        raise SheetsConfigError("환경변수 GOOGLE_SHEETS_ID가 없습니다.")
    return sheet_id


def get_client():
    """서비스 계정 인증으로 gspread 클라이언트를 만든다."""
    import gspread
    from google.oauth2.service_account import Credentials

    info = load_credentials_info()
    creds = Credentials.from_service_account_info(info, scopes=_SCOPES)
    return gspread.authorize(creds)


def _default_fx_provider(start, end) -> dict[str, float]:
    """[start, end] 원/달러 종가를 받는다 (실패 시 예외 — 호출부가 캐시로 대체)."""
    from data import fx

    return fx.fetch_usd_krw_range(start, end)


def _fx_history_for(fills_df: pd.DataFrame, fx_provider) -> tuple[dict[str, float], list[str]]:
    """빈 환율 칸이 있는 체결일 구간의 {날짜: 환율}을 구한다. 네트워크 실패면
    data/cache/fx_krw.json 캐시를 쓰고 경고를 남긴다(조용히 넘기지 않는다)."""
    from data import fx

    if fills_df.empty or not fills_df["fx_rate"].isna().any():
        return {}, []
    need = fills_df[fills_df["fx_rate"].isna()]
    start = (pd.Timestamp(need["date"].min()) - pd.Timedelta(days=10)).date()
    end = pd.Timestamp(need["date"].max()).date()
    history = fx._load_cache()
    try:
        history = {**history, **fx_provider(start, end)}
    except Exception as exc:
        return history, [f"체결 기록 경고: 체결일 환율을 받지 못해 캐시 값을 씁니다 ({exc})"]
    return history, []


def read_sheets(client=None, cfg: dict | None = None, fx_provider=None) -> SheetsResult:
    """계획·체결 두 탭을 읽어 SheetsResult로 돌려준다.

    client를 주면(테스트용) 그 클라이언트를 쓰고, 없으면 get_client()로 새로
    인증한다. client는 gspread.Client와 같은 인터페이스
    (open_by_key(sheet_id).worksheet(name).get_all_records())만 있으면 된다.
    cfg: 수수료율(backtest.costs) 조회용. 없으면 수수료 빈 칸은 0으로 채운다.
    fx_provider(start, end) -> {날짜: 환율}: 빈 환율 칸 채우기용(테스트 주입, 기본은 yfinance KRW=X).
    빈 줄은 건너뛰고, 잘못된 줄·헤더 오류는 plan_errors/fills.errors에 남는다.
    """
    if client is None:
        client = get_client()
    sheet_id = _sheet_id()
    spreadsheet = client.open_by_key(sheet_id)

    plan_records = spreadsheet.worksheet(SHEET_PLAN).get_all_records()
    plan_df, plan_errors = parse_plan_records(plan_records)

    fills_records = spreadsheet.worksheet(SHEET_FILLS).get_all_records()
    fills_result = parse_fill_records(fills_records)

    costs = ((cfg or {}).get("backtest") or {}).get("costs") or {}
    fx_history, fx_warnings = _fx_history_for(fills_result.df, fx_provider or _default_fx_provider)
    fills_df, cost_warnings = fill_missing_costs(
        fills_result.df,
        fx_history,
        float(costs.get("commission_buy_pct", 0.0)),
        float(costs.get("commission_sell_pct", 0.0)),
    )
    fills_result.df = fills_df
    fills_result.errors = list(fills_result.errors) + fx_warnings + cost_warnings

    read_at_kst = pd.Timestamp.now(tz="Asia/Seoul")

    return SheetsResult(
        plan_df=plan_df,
        plan_errors=plan_errors,
        fills=fills_result,
        read_at_kst=read_at_kst,
        client=client,
    )


# ── 투자현황 탭 읽기 (docs/design/auto_plan.md v2 규칙 A) ─────────────────────────

# A열 항목 이름(정규화 키) → 내부 이름. B열 값을 읽는다.
_PORTFOLIO_LABEL_MAP = {"총투자금": "total_krw", "종목당계획금액": "per_ticker_budget_krw"}
PORTFOLIO_BUDGET_MISSING_WARNING = "투자현황 탭에 종목당 계획금액이 없어 기본 수량을 계산하지 않았습니다"


def _parse_krw_cell(raw) -> float | None:
    """"3,000,000"·"₩3,000,000"·3000000 → 3000000.0. 비었거나 숫자가 아니면 None."""
    text = str(raw if raw is not None else "").replace(",", "").replace("₩", "").replace("원", "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_portfolio_values(values: list[list]) -> dict:
    """투자현황 탭 get_all_values 결과에서 A열 항목 이름으로 행을 찾아 B열 값을 읽는다(행 순서 무관).

    입력: values(행 목록 — 각 행은 셀 값 목록)
    출력: {"total_krw": float|None, "per_ticker_budget_krw": float|None}
         (항목 이름은 괄호 설명·공백 무시, 같은 항목이 여럿이면 첫 행)
    """
    out: dict = {"total_krw": None, "per_ticker_budget_krw": None}
    seen: set[str] = set()
    for line in values or []:
        if not line:
            continue
        key = _PORTFOLIO_LABEL_MAP.get(_normalize_header(line[0]))
        if key is None or key in seen:
            continue
        seen.add(key)
        out[key] = _parse_krw_cell(line[1] if len(line) > 1 else "")
    return out


def read_portfolio(client) -> tuple[dict, list[str]]:
    """투자현황 탭을 읽는다(읽기만). 탭이 없거나 읽지 못해도 멈추지 않고 None 값 + 경고를 돌려준다.

    입력: client(read_sheets가 쓴 gspread 클라이언트)
    출력: ({"total_krw","per_ticker_budget_krw"}, 경고 목록) — 종목당 계획금액이 없으면
         PORTFOLIO_BUDGET_MISSING_WARNING 한 줄
    """
    try:
        values = client.open_by_key(_sheet_id()).worksheet(SHEET_PORTFOLIO).get_all_values()
        portfolio = parse_portfolio_values(values)
    except Exception as exc:  # 탭 없음(WorksheetNotFound) 등 — 사유는 콘솔에만, 보고서는 경고 한 줄
        print(f"[sheets] 투자현황 탭을 읽지 못했습니다: {type(exc).__name__}: {exc}")
        portfolio = {"total_krw": None, "per_ticker_budget_krw": None}
    if portfolio["per_ticker_budget_krw"] is not None and portfolio["per_ticker_budget_krw"] <= 0:
        portfolio["per_ticker_budget_krw"] = None
    warnings = [] if portfolio["per_ticker_budget_krw"] is not None else [PORTFOLIO_BUDGET_MISSING_WARNING]
    return portfolio, warnings


# ── 계획 탭 자동 기록 (docs/design/auto_plan.md) ────────────────────────────────

# 자동 기록이 쓰는 입력 칸 5개(머리글 정규화 키 → 내부 이름). 이 밖의 열(환율, 1차~남은(주) 등
# 시트 수식)은 절대 쓰지 않는다.
_PLAN_WRITE_COLUMN_MAP = {
    "티커": "ticker", "종목": "ticker", "계획금액": "budget_krw", "등록일": "reg_date",
    "기준가": "ref_price", "메모": "memo",
}
PLAN_WRITE_FIELDS = ("ticker", "budget_krw", "reg_date", "ref_price", "memo")


class SheetsWriteForbiddenError(RuntimeError):
    """계획 탭이 아닌 탭에 쓰려 할 때."""


class PlanHeaderError(RuntimeError):
    """계획 탭 머리글에서 입력 칸 5개를 찾지 못했을 때."""


@dataclass
class AutoPlanWriteResult:
    """write_auto_plan()의 결과. written: 실제로 쓴 줄({"ticker",...,"row"}),
    skipped: 쓰지 않은 줄과 사유, already_present: 시트에 이미 있어 안 쓴 티커."""

    written: list[dict] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    already_present: list[str] = field(default_factory=list)


def _plan_worksheet(spreadsheet, name: str = SHEET_PLAN):
    """쓰기용 워크시트를 연다 — 계획 탭이 아니면 SheetsWriteForbiddenError."""
    if name != SHEET_PLAN:
        raise SheetsWriteForbiddenError(f"계획 탭 외 탭에는 쓸 수 없습니다: {name!r}")
    ws = spreadsheet.worksheet(name)
    _assert_plan_tab(ws)
    return ws


def _assert_plan_tab(ws) -> None:
    title = getattr(ws, "title", None)
    if title != SHEET_PLAN:
        raise SheetsWriteForbiddenError(f"계획 탭 외 탭에는 쓸 수 없습니다: {title!r}")


def _col_letter(col: int) -> str:
    """1부터 시작하는 열 번호 → A1 표기 열 문자 (1→A, 27→AA)."""
    letters = ""
    while col > 0:
        col, rem = divmod(col - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def find_plan_write_columns(header_row: list) -> dict[str, int]:
    """계획 탭 머리글 줄에서 입력 칸 5개의 열 번호(1부터)를 찾는다 (열 순서와 무관).

    입력: header_row(머리글 셀 값 목록 — "계획금액(원)"처럼 괄호 설명이 있어도 됨)
    출력: {"ticker","budget_krw","reg_date","ref_price","memo": 열 번호}
    예외: 하나라도 없으면 PlanHeaderError
    """
    cols: dict[str, int] = {}
    for i, name in enumerate(header_row, start=1):
        key = _PLAN_WRITE_COLUMN_MAP.get(_normalize_header(name))
        if key and key not in cols:
            cols[key] = i
    missing = [f for f in PLAN_WRITE_FIELDS if f not in cols]
    if missing:
        raise PlanHeaderError(f"계획 탭 머리글에서 입력 칸을 찾지 못했습니다: {missing}")
    return cols


def _cell(values: list[list], row: int, col: int) -> str:
    """get_all_values 결과에서 (행, 열)(1부터) 값을 문자열로 — 줄·칸이 없으면 빈 문자열."""
    if row - 1 >= len(values):
        return ""
    line = values[row - 1]
    return str(line[col - 1]).strip() if col - 1 < len(line) else ""


def _sheet_value(field_name: str, value):
    if field_name == "budget_krw":
        return int(round(float(value)))
    if field_name == "ref_price":
        return round(float(value), 2)
    return value


def write_plan_rows(ws, rows: list[dict], max_rows: int = 1000) -> AutoPlanWriteResult:
    """계획 탭에 새 줄을 쓴다 — 티커 칸이 빈 첫 줄의 입력 칸 5개만 셀 단위로 업데이트한다.

    append_row는 쓰지 않는다(1000행까지 수식이 미리 채워져 있어 맨 아래에 붙어 버림). 입력 칸
    5개가 모두 빈 줄만 쓴다(사용자 값이 남은 줄은 덮어쓰지 않음). 시트 원본에 이미 있는
    티커는 다시 쓰지 않는다(계획금액 오류 등으로 계획 DataFrame에서 빠진 줄 포함).

    입력: ws(계획 탭 워크시트 — title, get_all_values(), batch_update(data, value_input_option)),
         rows(core.auto_plan.select_first_buy_plan_rows 결과), max_rows(빈 줄을 찾을 마지막 행)
    출력: AutoPlanWriteResult(written에 "row" 포함)
    """
    _assert_plan_tab(ws)
    result = AutoPlanWriteResult()
    if not rows:
        return result
    values = ws.get_all_values()
    if not values:
        raise PlanHeaderError("계획 탭이 비어 있습니다(머리글 없음)")
    cols = find_plan_write_columns(values[0])
    present = {_cell(values, r, cols["ticker"]).upper() for r in range(2, len(values) + 1)} - {""}
    last_row = max(max_rows, len(values))

    def _row_is_empty(r: int) -> bool:
        return all(_cell(values, r, cols[f]) == "" for f in PLAN_WRITE_FIELDS)

    updates = []
    next_row = 2
    for row in rows:
        ticker = str(row["ticker"]).upper()
        if ticker in present:
            result.skipped.append(f"{ticker}: 계획 탭에 이미 있음")
            result.already_present.append(ticker)
            continue
        while next_row <= last_row and not _row_is_empty(next_row):
            next_row += 1
        if next_row > last_row:
            result.skipped.append(f"{ticker}: {last_row}행까지 빈 줄이 없음")
            continue
        for f in PLAN_WRITE_FIELDS:
            updates.append({"range": f"{_col_letter(cols[f])}{next_row}", "values": [[_sheet_value(f, row[f])]]})
        result.written.append({**row, "row": next_row})
        present.add(ticker)
        next_row += 1
    if updates:
        ws.batch_update(updates, value_input_option="USER_ENTERED")
    return result


def write_auto_plan(client, rows: list[dict]) -> AutoPlanWriteResult:
    """계획 탭 자동 기록 진입점 — 첫 매수 종목의 새 계획 줄 쓰기.

    입력: client(read_sheets가 쓴 gspread 클라이언트), rows(core.auto_plan.select_first_buy_plan_rows 결과)
    출력: AutoPlanWriteResult. 실패하면 예외를 그대로 낸다(호출부가 경고로 바꾼다).
    """
    ws = _plan_worksheet(client.open_by_key(_sheet_id()), SHEET_PLAN)
    return write_plan_rows(ws, rows)
