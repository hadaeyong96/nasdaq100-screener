"""체결 기록 입력 (P3.4부터 data/fills.xlsx가 기본, data/fills.csv는 폴백).

data/fills.xlsx (기본, P3.4): "체결기록" 시트만 읽는다("사용법"·"작성예시" 시트는
읽지 않는다). 열: 날짜, 종목, 차수, 매수매도, 체결가, 수량, 금액($)(수식, 무시),
메모(무시). data/fills.xlsx가 있으면 그것을 쓰고, 없으면 data/fills.csv를 쓴다.
둘 다 있으면 xlsx 우선 + 경고.

data/fills.csv (폴백, P3): 한글 형식(날짜, 종목, 차수, 매수매도, 체결가, 수량)과
영어 형식(P2, date, ticker, unit, side, price, qty)을 모두 읽는다. 날짜는
2026-09-23 / 2026/9/23 / 2026.9.23 형식을 모두 받는다. 인코딩은 엑셀이 저장하는
UTF-8(BOM 포함)과 CP949를 모두 시도해서 읽는다.

두 형식 모두 공통:
  - 차수: 1차, 2차, 3차, 재진입 (내부 코드 1/2/6/9) 또는 대기자금(QQQM 쉬는 돈 —
    FillsResult.cash_rows로 따로 담고, 신호 판정에는 쓰지 않는다. QQQM 운용
    로직은 P4).
  - 매수매도: 매수, 매도
  - 잘못된 줄(모르는 종목·차수·매수매도, 음수 수량 등)은 건너뛰고 오류 메시지를
    FillsResult.errors에 "N번째 줄 ..." 형식으로 남긴다 (호출부가 보고서 경고
    탭·실행 로그에 쓴다). 빈 줄은 조용히 건너뛴다.
  - 사용자가 실제 체결가·수량을 직접 적어 두면, 다음 실행 때 추천 수량·진입가를
    덮어쓴다. qty=0인 buy는 "체결 안 됨"으로 core/state.py의 apply_fill이 처리한다.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
FILLS_XLSX = DATA_DIR / "fills.xlsx"
FILLS_CSV = DATA_DIR / "fills.csv"

XLSX_SHEET_FILLS = "체결기록"
_LOCKED_WARNING = "체결 기록 오류: 체결 기록 파일이 열려 있음, 저장 후 닫아 주세요"

_COLUMNS = ["date", "ticker", "unit", "side", "price", "qty"]

# 한글 열 이름 -> 영어 내부 열 이름
_KR_COLUMN_MAP = {
    "날짜": "date",
    "종목": "ticker",
    "차수": "unit",
    "매수매도": "side",
    "체결가": "price",
    "수량": "qty",
}
_EN_COLUMNS = {"date", "ticker", "unit", "side", "price", "qty"}

_UNIT_KR_TO_CODE = {"1차": "1", "2차": "2", "3차": "6", "재진입": "9"}
_UNIT_CODE_SET = {"1", "2", "6", "9"}
_CASH_UNIT_LABEL = "대기자금"
_SIDE_KR_TO_CODE = {"매수": "buy", "매도": "sell"}
_SIDE_CODE_SET = {"buy", "sell"}

_ENCODINGS = ["utf-8-sig", "cp949"]


@dataclass
class FillsResult:
    """load_fills()의 결과. df는 신호 판정에 쓰는 유효한 행만, cash_rows는
    차수=대기자금(QQQM) 행만, errors는 건너뛴 줄의 사유."""

    df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=_COLUMNS))
    errors: list[str] = field(default_factory=list)
    cash_rows: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=_COLUMNS))


def _read_csv_any_encoding(path: Path) -> pd.DataFrame:
    """UTF-8(BOM 포함)·CP949 인코딩을 순서대로 시도해 CSV를 읽는다."""
    last_exc: Exception | None = None
    for encoding in _ENCODINGS:
        try:
            return pd.read_csv(path, dtype=str, encoding=encoding, keep_default_na=False)
        except (UnicodeDecodeError, UnicodeError) as exc:
            last_exc = exc
            continue
    raise ValueError(f"{path}를 지원하는 인코딩(UTF-8, CP949)으로 읽지 못했습니다: {last_exc}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """한글 열 이름을 영어 내부 열 이름으로 바꾼다. 이미 영어 형식이면 그대로 둔다."""
    columns = {c.strip() for c in df.columns}
    if columns & set(_KR_COLUMN_MAP):
        rename = {c: _KR_COLUMN_MAP.get(c.strip(), c.strip()) for c in df.columns}
        df = df.rename(columns=rename)
    else:
        df = df.rename(columns={c: c.strip() for c in df.columns})
    missing = _EN_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"필요한 열이 없습니다: {sorted(missing)}")
    return df[list(_EN_COLUMNS)]


def _parse_date_cell(raw: str) -> pd.Timestamp | None:
    """2026-09-23 / 2026/9/23 / 2026.9.23 형식을 모두 받는다. 실패하면 None."""
    text = str(raw).strip().replace(".", "-").replace("/", "-")
    try:
        return pd.Timestamp(text)
    except (ValueError, TypeError):
        return None


def _parse_unit_cell(raw) -> str | None:
    """1차/2차/3차/재진입(또는 코드 1/2/6/9) -> 내부 코드. 대기자금(QQQM)은 "cash"."""
    text = _cell_to_str(raw)
    if text == _CASH_UNIT_LABEL:
        return "cash"
    if text in _UNIT_KR_TO_CODE:
        return _UNIT_KR_TO_CODE[text]
    if text in _UNIT_CODE_SET:
        return text
    return None


def _parse_side_cell(raw) -> str | None:
    text = _cell_to_str(raw)
    if text in _SIDE_KR_TO_CODE:
        return _SIDE_KR_TO_CODE[text]
    lowered = text.lower()
    if lowered in _SIDE_CODE_SET:
        return lowered
    return None


def _parse_qty_cell(raw) -> int | None:
    try:
        qty = int(float(_cell_to_str(raw)))
    except (ValueError, TypeError):
        return None
    return qty if qty >= 0 else None


def _parse_price_cell(raw) -> float | None:
    try:
        return float(_cell_to_str(raw))
    except (ValueError, TypeError):
        return None


def _is_na(raw) -> bool:
    if raw is None:
        return True
    try:
        return bool(pd.isna(raw))
    except (TypeError, ValueError):
        return False


def _cell_to_str(raw) -> str:
    """셀 값을 문자열로 만든다. None·NaN·NaT은 빈 문자열(엑셀 빈 칸·CSV 빈 칸 공통 처리)."""
    if _is_na(raw):
        return ""
    return str(raw).strip()


def _parse_date_value(raw) -> pd.Timestamp | None:
    """CSV의 문자열 날짜와 엑셀의 날짜 셀(datetime)을 모두 받는다."""
    if _is_na(raw):
        return None
    if isinstance(raw, pd.Timestamp):
        return raw.normalize()
    if isinstance(raw, _dt.datetime):
        return pd.Timestamp(raw).normalize()
    if isinstance(raw, _dt.date):
        return pd.Timestamp(raw)
    return _parse_date_cell(str(raw))


def _is_blank_record(record: dict) -> bool:
    """날짜·종목·차수·매수매도·체결가·수량이 모두 빈 줄인지 (건너뛸 빈 줄)."""
    return all(_cell_to_str(record.get(col)) == "" for col in _COLUMNS)


def _rows_from_records(records: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
    """정규화된 열(date/ticker/unit/side/price/qty)을 가진 레코드 목록을 검증해
    (신호용 행, 대기자금 행, 오류 메시지)로 나눈다. 빈 줄은 조용히 건너뛴다."""
    rows: list[dict] = []
    cash_rows: list[dict] = []
    errors: list[str] = []
    for i, record in enumerate(records, start=1):
        if _is_blank_record(record):
            continue

        problems: list[str] = []

        date = _parse_date_value(record["date"])
        if date is None:
            problems.append(f"날짜 형식을 알 수 없음({record['date']!r})")

        ticker = _cell_to_str(record["ticker"]).upper()
        if not ticker:
            problems.append("종목이 비어 있음")

        unit = _parse_unit_cell(record["unit"])
        if unit is None:
            problems.append(f"차수 값을 알 수 없음({record['unit']!r})")

        side = _parse_side_cell(record["side"])
        if side is None:
            problems.append(f"매수매도 값을 알 수 없음({record['side']!r})")

        price = _parse_price_cell(record["price"])
        if price is None or price < 0:
            problems.append(f"체결가가 올바르지 않음({record['price']!r})")

        qty = _parse_qty_cell(record["qty"])
        if qty is None:
            problems.append(f"수량이 올바르지 않음(음수 또는 숫자 아님: {record['qty']!r})")

        if problems:
            errors.append(f"체결 기록 오류: {i}번째 줄 - {', '.join(problems)}")
            continue

        row = {"date": date, "ticker": ticker, "unit": unit, "side": side, "price": price, "qty": qty}
        (cash_rows if unit == "cash" else rows).append(row)

    return rows, cash_rows, errors


def _to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=_COLUMNS) if rows else pd.DataFrame(columns=_COLUMNS)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df["unit"] = df["unit"].astype(str)
    return df


def _load_csv(path: Path) -> FillsResult:
    """fills.csv(폴백)를 읽는다. 파일이 없거나 헤더만 있으면 빈 DataFrame.

    출력: FillsResult(df, errors, cash_rows). df는 DataFrame(date, ticker, unit, side,
         price, qty). date는 Timestamp, unit은 문자열("1"/"2"/"6"/"9")로 정규화한다.
         잘못된 줄은 df에서 빠지고 errors에 "N번째 줄 ..."로 남는다 (N은 헤더 다음
         첫 데이터 줄을 1로 하는 1-base 줄 번호).
    """
    if not path.exists():
        return FillsResult(df=pd.DataFrame(columns=_COLUMNS))

    raw = _read_csv_any_encoding(path)
    if raw.empty:
        return FillsResult(df=pd.DataFrame(columns=_COLUMNS))

    try:
        raw = _normalize_columns(raw)
    except ValueError as exc:
        return FillsResult(df=pd.DataFrame(columns=_COLUMNS), errors=[f"체결 기록 오류: 헤더 오류 - {exc}"])

    rows, cash_rows, errors = _rows_from_records(raw.to_dict(orient="records"))
    return FillsResult(df=_to_frame(rows), errors=errors, cash_rows=_to_frame(cash_rows))


def _is_xlsx_locked(path: Path) -> bool:
    """엑셀이 파일을 열어 두면 같은 폴더에 임시 파일 ~$fills.xlsx가 생긴다."""
    return path.with_name("~$" + path.name).exists()


def _load_xlsx(path: Path) -> FillsResult:
    """fills.xlsx의 "체결기록" 시트만 읽는다("사용법"·"작성예시" 시트는 읽지 않는다).

    출력: FillsResult(df, errors, cash_rows). 파일이 열려 있어 잠겨 있으면(임시 파일
         존재 또는 읽기 오류) 실행을 멈추지 않고 경고 하나만 남긴다.
    """
    if _is_xlsx_locked(path):
        return FillsResult(errors=[_LOCKED_WARNING])

    try:
        raw = pd.read_excel(path, sheet_name=XLSX_SHEET_FILLS, engine="openpyxl")
    except Exception as exc:  # 엑셀이 파일을 잠그고 있거나 그 밖의 읽기 오류
        return FillsResult(errors=[f"{_LOCKED_WARNING} ({exc})"])

    if raw.empty:
        return FillsResult(df=pd.DataFrame(columns=_COLUMNS))

    try:
        raw = _normalize_columns(raw)
    except ValueError as exc:
        return FillsResult(df=pd.DataFrame(columns=_COLUMNS), errors=[f"체결 기록 오류: 헤더 오류 - {exc}"])

    rows, cash_rows, errors = _rows_from_records(raw.to_dict(orient="records"))
    return FillsResult(df=_to_frame(rows), errors=errors, cash_rows=_to_frame(cash_rows))


def load_fills(path: Path | None = None) -> FillsResult:
    """체결 기록을 읽는다. path를 주면 확장자로 형식을 정해 그 파일만 읽는다
    (테스트용). path가 없으면 data/fills.xlsx가 있으면 그것을, 없으면
    data/fills.csv를 쓴다. 둘 다 있으면 xlsx 우선 + 경고.
    """
    if path is not None:
        return _load_xlsx(path) if path.suffix.lower() == ".xlsx" else _load_csv(path)

    xlsx_exists = FILLS_XLSX.exists()
    csv_exists = FILLS_CSV.exists()
    if xlsx_exists:
        result = _load_xlsx(FILLS_XLSX)
        if csv_exists:
            result.errors = [
                "체결 기록 경고: data/fills.xlsx와 data/fills.csv가 모두 있어 fills.xlsx를 사용합니다."
            ] + result.errors
        return result
    if csv_exists:
        return _load_csv(FILLS_CSV)
    return FillsResult(df=pd.DataFrame(columns=_COLUMNS))


def summarize_cash_rows(cash_rows: pd.DataFrame) -> dict | None:
    """대기자금(QQQM) 체결 기록을 날짜 순으로 누적해 순보유수량·평균단가를 낸다
    (신호 판정에는 쓰지 않고 보고서 표시용, QQQM 운용 로직은 P4).

    출력: {"qty": int, "avg_price": float|None} 또는 기록이 없으면 None.
    """
    if cash_rows is None or cash_rows.empty:
        return None
    qty = 0
    cost = 0.0
    for _, r in cash_rows.sort_values("date").iterrows():
        if r["side"] == "buy":
            cost += r["price"] * r["qty"]
            qty += r["qty"]
        elif qty > 0:
            avg = cost / qty
            sell_qty = min(r["qty"], qty)
            cost -= avg * sell_qty
            qty -= sell_qty
    if qty <= 0:
        return {"qty": 0, "avg_price": None}
    return {"qty": qty, "avg_price": cost / qty}


def fills_for(df: pd.DataFrame, ticker: str, date) -> list[dict]:
    """특정 종목·날짜의 체결 기록을 행 dict 목록으로 반환한다."""
    if df.empty:
        return []
    match = df[(df["ticker"] == ticker) & (df["date"] == pd.Timestamp(date))]
    return match.to_dict(orient="records")
