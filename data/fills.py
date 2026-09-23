"""체결 기록(data/fills.csv) 입력.

한글 형식(기본, P3): 날짜, 종목, 차수, 매수매도, 체결가, 수량
  - 차수: 1차, 2차, 3차, 재진입
  - 매수매도: 매수, 매도
영어 형식(P2, 계속 읽음): date, ticker, unit, side, price, qty
  - unit: 1, 2, 6, 9 / side: buy, sell

날짜는 2026-09-23 / 2026/9/23 / 2026.9.23 형식을 모두 받는다. 인코딩은 엑셀이
저장하는 UTF-8(BOM 포함)과 CP949를 모두 시도해서 읽는다.

잘못된 줄(모르는 종목·차수·매수매도, 음수 수량 등)은 건너뛰고 오류 메시지를
FillsResult.errors에 "N번째 줄 ..." 형식으로 남긴다 (호출부가 보고서 경고 탭·
실행 로그에 쓴다). 사용자가 실제 체결가·수량을 직접 적어 두면, 다음 실행 때
추천 수량·진입가를 덮어쓴다. qty=0인 buy는 "체결 안 됨"으로 core/state.py의
apply_fill이 처리한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
FILLS_CSV = DATA_DIR / "fills.csv"

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
_SIDE_KR_TO_CODE = {"매수": "buy", "매도": "sell"}
_SIDE_CODE_SET = {"buy", "sell"}

_ENCODINGS = ["utf-8-sig", "cp949"]


@dataclass
class FillsResult:
    """load_fills()의 결과. df는 유효한 행만, errors는 건너뛴 줄의 사유."""

    df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=_COLUMNS))
    errors: list[str] = field(default_factory=list)


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


def _parse_unit_cell(raw: str) -> str | None:
    text = str(raw).strip()
    if text in _UNIT_KR_TO_CODE:
        return _UNIT_KR_TO_CODE[text]
    if text in _UNIT_CODE_SET:
        return text
    return None


def _parse_side_cell(raw: str) -> str | None:
    text = str(raw).strip()
    if text in _SIDE_KR_TO_CODE:
        return _SIDE_KR_TO_CODE[text]
    lowered = text.lower()
    if lowered in _SIDE_CODE_SET:
        return lowered
    return None


def _parse_qty_cell(raw: str) -> int | None:
    try:
        qty = int(float(str(raw).strip()))
    except (ValueError, TypeError):
        return None
    return qty if qty >= 0 else None


def _parse_price_cell(raw: str) -> float | None:
    try:
        return float(str(raw).strip())
    except (ValueError, TypeError):
        return None


def load_fills(path: Path = FILLS_CSV) -> FillsResult:
    """fills.csv를 읽는다. 파일이 없거나 헤더만 있으면 빈 DataFrame.

    출력: FillsResult(df, errors). df는 DataFrame(date, ticker, unit, side, price, qty).
         date는 Timestamp, unit은 문자열("1"/"2"/"6"/"9")로 정규화한다.
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

    rows: list[dict] = []
    errors: list[str] = []
    for i, record in enumerate(raw.to_dict(orient="records"), start=1):
        problems: list[str] = []

        date = _parse_date_cell(record["date"])
        if date is None:
            problems.append(f"날짜 형식을 알 수 없음({record['date']!r})")

        ticker = str(record["ticker"]).strip().upper()
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

        rows.append({"date": date, "ticker": ticker, "unit": unit, "side": side, "price": price, "qty": qty})

    df = pd.DataFrame(rows, columns=_COLUMNS) if rows else pd.DataFrame(columns=_COLUMNS)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df["unit"] = df["unit"].astype(str)
    return FillsResult(df=df, errors=errors)


def fills_for(df: pd.DataFrame, ticker: str, date) -> list[dict]:
    """특정 종목·날짜의 체결 기록을 행 dict 목록으로 반환한다."""
    if df.empty:
        return []
    match = df[(df["ticker"] == ticker) & (df["date"] == pd.Timestamp(date))]
    return match.to_dict(orient="records")
