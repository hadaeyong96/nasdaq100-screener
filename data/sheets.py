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
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from data.fills import FillsResult, fill_missing_costs, parse_fill_records, parse_plan_records

ROOT = Path(__file__).resolve().parents[1]
# notify/telegram.py와 같은 패턴: 이 모듈만 단독으로 쓰는 스크립트(scripts/drive_sync.py
# 등, notify.telegram을 안 거치는 경로)도 .env를 확실히 읽도록 여기서 직접 불러온다.
# override=True: 이 프로세스에 같은 이름의 빈 환경변수가 이미 있어도 .env 값으로 덮는다.
load_dotenv(ROOT / ".env", override=True)

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

SHEET_PLAN = "계획"
SHEET_FILLS = "체결"

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
    """

    plan_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=_PLAN_COLUMNS))
    plan_errors: list[str] = field(default_factory=list)
    fills: FillsResult = field(default_factory=FillsResult)
    read_at_kst: pd.Timestamp | None = None


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
    )
