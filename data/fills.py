"""체결 기록 입력 (P3.4부터 data/fills.xlsx가 기본, data/fills.csv는 폴백).

data/fills.xlsx (기본, P3.4): "체결기록" 시트만 읽는다("사용법"·"작성예시" 시트는
읽지 않는다). 열: 날짜, 종목, 차수, 매수매도, 체결가, 수량, 금액($)(수식, 무시),
메모(무시). data/fills.xlsx가 있으면 그것을 쓰고, 없으면 data/fills.csv를 쓴다.
둘 다 있으면 xlsx 우선 + 경고.

data/fills.csv (폴백, P3): 한글 형식(날짜, 종목, 차수, 매수매도, 체결가, 수량)과
영어 형식(P2, date, ticker, unit, side, price, qty)을 모두 읽는다. 날짜는
2026-09-23 / 2026/9/23 / 2026.9.23 형식을 모두 받는다. 인코딩은 엑셀이 저장하는
UTF-8(BOM 포함)과 CP949를 모두 시도해서 읽는다.

구글 시트 "체결" 탭(라이브, data/sheets.py): 날짜, 티커, 구분(매수/매도), 수량,
체결가($), 환율(원/$), 수수료($), 메모. 헤더의 괄호 설명·공백은 무시하고 비교한다.
차수 열이 없으면 종목별 매수 순서로 1차→2차→3차를 자동 배정한다(_assign_units).

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
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
FILLS_XLSX = DATA_DIR / "fills.xlsx"
FILLS_CSV = DATA_DIR / "fills.csv"

XLSX_SHEET_FILLS = "체결기록"
XLSX_SHEET_PLAN = "계획"  # 라이브 어드바이저 1단계(docs/design/live_advisor.md 1번) — 선택 사항, 없어도 오류 아님
_LOCKED_WARNING = "체결 기록 오류: 체결 기록 파일이 열려 있음, 저장 후 닫아 주세요"

# unit = 차수(묶음) 내부 코드: 1차="1", 2차="2", 3차="6"(1:2:6의 6), 재진입="9",
# 대기자금(QQQM)="cash". core/state.py가 묶음별 수량·진입가를 이 키로 관리한다.
# fx_rate(원/$)·fee_usd($)·amount_usd($, 이 줄의 실제 체결금액)는 신호 판정에는 안 쓰고
# 기록용이다. 구글 시트는 비어 있으면 data/sheets.py가 채운다(환율=체결일 종가, 수수료=config).
_COLUMNS = ["date", "ticker", "unit", "side", "price", "qty", "fx_rate", "fee_usd", "amount_usd"]

_PLAN_COLUMNS = ["ticker", "budget_krw", "memo"]
# 계획 탭 헤더(괄호 설명·공백 제거 후) -> 내부 열 이름. "등록일" 등 나머지 열은 무시한다.
_PLAN_KR_COLUMN_MAP = {"티커": "ticker", "종목": "ticker", "계획금액": "budget_krw", "메모": "memo"}

# 체결 헤더(괄호 설명·공백 제거 후) -> 영어 내부 열 이름. 구글 시트 "체결" 탭
# (날짜, 티커, 구분(매수/매도), 수량, 체결가($), 환율(원/$), 수수료($), 메모)과
# 기존 fills.xlsx/csv(날짜, 종목, 차수, 매수매도, 체결가, 수량)를 모두 받는다.
_KR_COLUMN_MAP = {
    "날짜": "date",
    "체결일": "date",
    "종목": "ticker",
    "티커": "ticker",
    "차수": "unit",
    "매수매도": "side",
    "구분": "side",
    "체결가": "price",
    "수량": "qty",
    "환율": "fx_rate",
    "수수료": "fee_usd",
    "메모": "memo",
}
_REQUIRED_COLUMNS = {"date", "ticker", "side", "price", "qty"}
# 없어도 되는 열: 차수가 없으면 종목별 매수 순서로 1차→2차→3차를 자동 배정한다
# (_assign_units). 환율·수수료가 없거나 비면 NaN으로 두고 호출부가 채운다.
_OPTIONAL_COLUMNS = ("unit", "fx_rate", "fee_usd")
_AUTO_UNIT_ORDER = ("1", "2", "6")

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


def _normalize_header(name) -> str:
    """헤더 셀을 비교용 키로 만든다: 괄호 설명("(원)", "($)", "(매수/매도)")과 공백을 지운다.
    예: "계획금액(원)" -> "계획금액", "구분(매수/매도)" -> "구분", " price " -> "price"."""
    text = re.sub(r"[\(（].*?[\)）]", "", str(name))
    return re.sub(r"\s+", "", text)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """한글·영어 열 이름을 영어 내부 열 이름으로 바꾼다.

    입력: 원본 DataFrame (헤더는 한글 또는 영어, 괄호 설명 포함 가능)
    출력: DataFrame(date, ticker, unit, side, price, qty, fx_rate, fee_usd).
         없는 선택 열(차수·환율·수수료)은 빈 문자열로 채운다.
    예외: 필수 열(날짜·티커·구분·체결가·수량)이 없으면 ValueError(한글 열 이름 안내).
    """
    rename = {}
    for c in df.columns:
        key = _normalize_header(c)
        rename[c] = _KR_COLUMN_MAP.get(key, key.lower())
    df = df.rename(columns=rename)
    df = df.loc[:, ~df.columns.duplicated()]
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        kr = {"date": "날짜", "ticker": "티커", "side": "구분(매수/매도)", "price": "체결가($)", "qty": "수량"}
        names = [f"{kr[m]}({m})" for m in sorted(missing)]
        raise ValueError(f"필요한 열이 없습니다: {names} — 지금 헤더: {list(rename)}")
    for col in _OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[["date", "ticker", "unit", "side", "price", "qty", "fx_rate", "fee_usd"]]


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
        qty = int(float(_cell_to_str(raw).replace(",", "")))
    except (ValueError, TypeError):
        return None
    return qty if qty >= 0 else None


def _parse_number_cell(raw) -> float | None:
    """숫자 셀. 쉼표·통화 기호($, ₩, 원)를 지우고 float로. 실패하면 None."""
    text = _cell_to_str(raw).replace(",", "").replace("$", "").replace("₩", "").replace("원", "").strip()
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def _parse_price_cell(raw) -> float | None:
    return _parse_number_cell(raw)


def _parse_optional_number(raw) -> tuple[bool, float | None]:
    """선택 숫자 칸(환율·수수료). 출력: (올바름 여부, 값). 빈 칸은 (True, None)."""
    if _cell_to_str(raw) == "":
        return True, None
    value = _parse_number_cell(raw)
    if value is None or value < 0:
        return False, None
    return True, value


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
    """날짜·종목·매수매도·체결가·수량이 모두 빈 줄인지 (건너뛸 빈 줄). 차수·환율·
    수수료·메모만 적힌 줄(시트에 미리 채워 둔 서식 등)도 빈 줄로 본다."""
    return all(_cell_to_str(record.get(col)) == "" for col in _REQUIRED_COLUMNS)


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

        unit_text = _cell_to_str(record.get("unit"))
        unit = _parse_unit_cell(unit_text) if unit_text else None  # 비면 매수 순서로 자동 배정
        if unit_text and unit is None:
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

        fx_ok, fx_rate = _parse_optional_number(record.get("fx_rate"))
        if not fx_ok:
            problems.append(f"환율이 올바르지 않음({record.get('fx_rate')!r})")

        fee_ok, fee_usd = _parse_optional_number(record.get("fee_usd"))
        if not fee_ok:
            problems.append(f"수수료가 올바르지 않음({record.get('fee_usd')!r})")

        if problems:
            errors.append(f"체결 기록 오류: {i}번째 줄 - {', '.join(problems)}")
            continue

        row = {
            "date": date, "ticker": ticker, "unit": unit, "side": side, "price": price, "qty": qty,
            "fx_rate": fx_rate, "fee_usd": fee_usd, "amount_usd": price * qty, "line": i,
        }
        (cash_rows if unit == "cash" else rows).append(row)

    return rows, cash_rows, errors


def _assign_units(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """차수가 빈 체결 줄에 차수(unit)를 매수 순서로 배정한다 (순수 함수).

    규칙(구글 시트 "체결" 탭에는 차수 열이 없다):
      - 종목별로 날짜 순(같은 날은 시트 줄 순서) 처리. 보유 0주에서 시작하는 매수가
        1차("1"), 그다음 매수가 2차("2"), 그다음이 3차("6").
      - 3차까지 다 쓴 뒤의 추가 매수는 3차 묶음에 합친다(수량 합산·가중평균 단가) —
        core.state.apply_fill은 같은 묶음 매수를 덮어쓰므로 누적값으로 적는다.
      - 차수 없는 매도는 나중 차수부터(3차→2차→1차, 재진입 포함) 줄인다. 한 줄이 여러
        묶음에 걸치면 묶음별 줄로 나눈다. 보유보다 많이 팔면 경고.
      - 전량 매도로 0주가 되면 다음 매수는 다시 1차부터.
      - 차수가 적힌 줄은 그대로 두고 보유 추적에만 반영한다.
    입력: _rows_from_records의 신호용 행 목록(unit이 None이면 자동 배정 대상)
    출력: (차수가 채워진 행 목록, 경고 메시지 목록)
    """
    out: list[dict] = []
    warnings: list[str] = []
    held: dict[str, dict[str, list[float]]] = {}  # ticker -> unit -> [qty, avg_price]
    used: dict[str, set[str]] = {}
    ordered = sorted(rows, key=lambda r: (r["ticker"], r["date"], r["line"]))
    for row in ordered:
        t = row["ticker"]
        pos = held.setdefault(t, {})
        used_t = used.setdefault(t, set())
        if sum(q for q, _ in pos.values()) <= 0:
            pos.clear()
            used_t.clear()

        if row["unit"] is not None:  # 사용자가 차수를 적었다 — apply_fill과 같은 의미로 추적만
            u = row["unit"]
            if row["side"] == "buy":
                pos[u] = [row["qty"], row["price"]]
                used_t.add(u)
            else:
                pos.setdefault(u, [0, row["price"]])
                pos[u][0] = max(pos[u][0] - row["qty"], 0)
            out.append(row)
            continue

        if row["side"] == "buy":
            free = [u for u in _AUTO_UNIT_ORDER if u not in used_t]
            u = free[0] if free else _AUTO_UNIT_ORDER[-1]
            prev_qty, prev_price = pos.get(u, [0, 0.0]) if not free else (0, 0.0)
            total_qty = prev_qty + row["qty"]
            avg = (prev_qty * prev_price + row["qty"] * row["price"]) / total_qty if total_qty > 0 else row["price"]
            pos[u] = [total_qty, avg]
            used_t.add(u)
            out.append({**row, "unit": u, "qty": total_qty, "price": avg})
            continue

        remaining = row["qty"]
        split: list[dict] = []
        for u in sorted(pos, key=lambda k: {"9": 3, "6": 2, "2": 1, "1": 0}.get(k, -1), reverse=True):
            if remaining <= 0:
                break
            take = min(pos[u][0], remaining)
            if take <= 0:
                continue
            pos[u][0] -= take
            remaining -= take
            split.append({**row, "unit": u, "qty": take})
        if remaining > 0:
            warnings.append(
                f"체결 기록 경고: {row['line']}번째 줄 - {t} 매도 수량이 보유보다 {remaining}주 많음 (보유분만 반영)"
            )
        for part in split:  # 체결금액·수수료는 나눈 수량 비율대로
            ratio = part["qty"] / row["qty"] if row["qty"] else 0
            part["amount_usd"] = row["amount_usd"] * ratio
            if row["fee_usd"] is not None:
                part["fee_usd"] = row["fee_usd"] * ratio
        out.extend(split)
    return out, warnings


def _to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=_COLUMNS) if rows else pd.DataFrame(columns=_COLUMNS)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df["unit"] = df["unit"].astype(str)
        df = df.sort_values("date", kind="stable").reset_index(drop=True)
    return df


def parse_fill_records(records: list[dict]) -> FillsResult:
    """행 dict 목록(구글 시트 get_all_records() 등)을 체결 기록으로 파싱한다.
    열 이름은 한글(날짜/종목/차수/매수매도/체결가/수량) 또는 영어(P2 레거시)
    모두 받는다. _load_csv/_load_xlsx와 완전히 같은 검증 로직을 쓴다
    (data/sheets.py의 "체결" 탭 읽기가 이 함수를 재사용한다).
    """
    if not records:
        return FillsResult(df=pd.DataFrame(columns=_COLUMNS))

    raw = pd.DataFrame(records)
    if raw.empty:
        return FillsResult(df=pd.DataFrame(columns=_COLUMNS))

    try:
        raw = _normalize_columns(raw)
    except ValueError as exc:
        return FillsResult(df=pd.DataFrame(columns=_COLUMNS), errors=[f"체결 기록 오류: 헤더 오류 - {exc}"])

    rows, cash_rows, errors = _rows_from_records(raw.to_dict(orient="records"))
    rows, unit_warnings = _assign_units(rows)
    return FillsResult(df=_to_frame(rows), errors=errors + unit_warnings, cash_rows=_to_frame(cash_rows))


def fill_missing_costs(
    df: pd.DataFrame, fx_by_date: dict[str, float], commission_buy_pct: float, commission_sell_pct: float
) -> tuple[pd.DataFrame, list[str]]:
    """빈 환율·수수료 칸을 채운다 (순수 함수).

    입력: df(FillsResult.df 형태), fx_by_date({"YYYY-MM-DD": 원/달러 종가} — 체결일 이하
         가장 최근 값을 쓴다), commission_buy_pct·commission_sell_pct(수수료율 %)
    출력: (채운 새 DataFrame, 경고 목록 — 환율을 못 구한 줄)
    - 환율: 체결일 원/달러 종가(그날 값이 없으면 직전 거래일 값).
    - 수수료: 체결금액(amount_usd) × 수수료율 / 100.
    """
    if df.empty:
        return df, []
    from data.fx import pick_rate_for_date

    out = df.copy()
    warnings: list[str] = []
    for idx, r in out.iterrows():
        if _is_na(r["fx_rate"]):
            picked = pick_rate_for_date(fx_by_date, pd.Timestamp(r["date"]).date().isoformat())
            if picked is None:
                warnings.append(f"체결 기록 경고: {r['ticker']} {pd.Timestamp(r['date']).date()} 환율을 구하지 못함 (빈 칸 유지)")
            else:
                out.at[idx, "fx_rate"] = picked[0]
        if _is_na(r["fee_usd"]):
            pct = commission_buy_pct if r["side"] == "buy" else commission_sell_pct
            out.at[idx, "fee_usd"] = round(float(r["amount_usd"]) * pct / 100, 4)
    return out, warnings


def _parse_budget_cell(raw) -> float | None:
    value = _parse_number_cell(raw)
    return value if value is not None and value >= 0 else None


def _is_blank_plan_record(normalized: dict) -> bool:
    """티커·계획금액이 모두 빈 줄(등록일·메모만 있는 줄 포함)은 건너뛸 빈 줄로 본다."""
    return _cell_to_str(normalized.get("ticker")) == "" and _cell_to_str(normalized.get("budget_krw")) == ""


def parse_plan_records(records: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    """계획 탭/시트의 행 dict 목록을 파싱한다 (라이브 어드바이저 1단계,
    docs/design/live_advisor.md 1번, 2번). 열 이름은 한글(티커/계획금액/메모)
    또는 이미 영어(ticker/budget_krw/memo)면 그대로 받는다. 빈 줄은 조용히
    건너뛴다. 잘못된 줄(티커 없음, 계획금액이 숫자가 아니거나 음수)은
    반환 DataFrame에서 빠지고 errors에 "N번째 줄 ..."로 남는다.

    data/sheets.py(구글 시트 "계획" 탭)와 이 파일의 load_plan()(fills.xlsx의
    "계획" 시트) 둘 다 이 함수 하나를 재사용한다 — 새 파서를 만들지 않는다.

    출력: (DataFrame(ticker, budget_krw, memo), 오류 메시지 목록).
    """
    rows: list[dict] = []
    errors: list[str] = []
    for i, record in enumerate(records, start=1):
        normalized = {}
        for k, v in record.items():
            key = _normalize_header(k)
            normalized.setdefault(_PLAN_KR_COLUMN_MAP.get(key, key.lower()), v)
        if _is_blank_plan_record(normalized):
            continue

        problems: list[str] = []

        ticker = _cell_to_str(normalized.get("ticker")).upper()
        if not ticker:
            problems.append("티커가 비어 있음")

        budget_raw = normalized.get("budget_krw", "")
        budget = _parse_budget_cell(budget_raw)
        if budget is None:
            problems.append(f"계획금액이 올바르지 않음({budget_raw!r})")

        memo = _cell_to_str(normalized.get("memo"))

        if problems:
            errors.append(f"계획 기록 오류: {i}번째 줄 - {', '.join(problems)}")
            continue

        rows.append({"ticker": ticker, "budget_krw": budget, "memo": memo})

    df = pd.DataFrame(rows, columns=_PLAN_COLUMNS) if rows else pd.DataFrame(columns=_PLAN_COLUMNS)
    return df, errors


def load_plan(path: Path | None = None) -> tuple[pd.DataFrame, list[str]]:
    """fills.xlsx의 "계획" 시트를 읽는다 (라이브 어드바이저 1단계 1e — 구글
    시트 연동(1g) 전까지 로컬 임시 입력처). 시트가 아예 없으면(기존
    fills.xlsx에는 없던, 새로 추가하는 선택 시트) 오류 없이 빈 결과를 낸다 —
    계획을 아직 안 올린 종목이 있을 뿐이므로.

    path를 주면 그 파일만 읽는다(테스트용). path가 없으면 data/fills.xlsx를
    쓰고, 그 파일 자체가 없으면 빈 결과.

    출력: (DataFrame(ticker, budget_krw, memo), 오류 메시지 목록)
    """
    target = path if path is not None else FILLS_XLSX
    if not target.exists():
        return pd.DataFrame(columns=_PLAN_COLUMNS), []
    if _is_xlsx_locked(target):
        return pd.DataFrame(columns=_PLAN_COLUMNS), [_LOCKED_WARNING.replace("체결 기록", "계획")]

    try:
        raw = pd.read_excel(target, sheet_name=XLSX_SHEET_PLAN, engine="openpyxl")
    except ValueError:  # 시트가 없음 — 선택 사항이므로 조용히 빈 결과
        return pd.DataFrame(columns=_PLAN_COLUMNS), []
    except Exception as exc:  # 엑셀이 파일을 잠그고 있거나 그 밖의 읽기 오류
        return pd.DataFrame(columns=_PLAN_COLUMNS), [f"계획 기록 오류: 파일을 읽을 수 없음 ({exc})"]

    if raw.empty:
        return pd.DataFrame(columns=_PLAN_COLUMNS), []

    return parse_plan_records(raw.to_dict(orient="records"))


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

    return parse_fill_records(raw.to_dict(orient="records"))


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

    return parse_fill_records(raw.to_dict(orient="records"))


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



def fill_key(rec: dict) -> tuple:
    """체결 한 줄의 식별 키 (날짜, 종목, 매수매도, 수량, 체결가). store fill_ledger와 시트를
    비교해 이미 반영한 체결을 다시 반영하지 않는 데 쓴다 (engine.daily)."""
    return (
        str(pd.Timestamp(rec["date"]).date()),
        str(rec["ticker"]).upper(),
        str(rec["side"]),
        int(rec["qty"]),
        round(float(rec["price"]), 4),
    )
