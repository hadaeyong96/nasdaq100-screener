"""구글 스프레드시트 입력 (계획·체결) — 라이브 어드바이저 1단계
(docs/design/live_advisor.md 1번).

data/fills.xlsx를 대체한다. "계획" 탭(티커, 계획금액, 메모)과 "체결" 탭
(data/fills.xlsx의 "체결기록" 시트와 같은 열: 날짜, 종목, 차수, 매수매도,
체결가, 수량)을 서비스 계정으로 읽는다. 체결 탭의 행 검증은
data/fills.py의 기존 파서(parse_fill_records)를 그대로 재사용한다 — 새
파서를 만들지 않는다.

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

from data.fills import FillsResult, parse_fill_records, parse_plan_records

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

_PLAN_COLUMNS = ["ticker", "budget_krw", "memo"]


class SheetsConfigError(RuntimeError):
    """인증 정보나 시트 ID 환경변수가 없거나 잘못됐을 때."""


@dataclass
class SheetsResult:
    """read_sheets()의 결과.

    plan_df: DataFrame(ticker, budget_krw, memo) — 계획 탭의 유효한 행만.
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


def read_sheets(client=None) -> SheetsResult:
    """계획·체결 두 탭을 읽어 SheetsResult로 돌려준다.

    client를 주면(테스트용) 그 클라이언트를 쓰고, 없으면 get_client()로 새로
    인증한다. client는 gspread.Client와 같은 인터페이스
    (open_by_key(sheet_id).worksheet(name).get_all_records())만 있으면 된다.
    """
    if client is None:
        client = get_client()
    sheet_id = _sheet_id()
    spreadsheet = client.open_by_key(sheet_id)

    plan_records = spreadsheet.worksheet(SHEET_PLAN).get_all_records()
    plan_df, plan_errors = parse_plan_records(plan_records)

    fills_records = spreadsheet.worksheet(SHEET_FILLS).get_all_records()
    fills_result = parse_fill_records(fills_records)

    read_at_kst = pd.Timestamp.now(tz="Asia/Seoul")

    return SheetsResult(
        plan_df=plan_df,
        plan_errors=plan_errors,
        fills=fills_result,
        read_at_kst=read_at_kst,
    )
