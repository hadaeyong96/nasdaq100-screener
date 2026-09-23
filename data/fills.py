"""체결 기록(data/fills.csv) 입력.

컬럼: date, ticker, unit(1/2/6/9), side(buy/sell), price, qty
사용자가 실제 체결가·수량을 직접 적어 두면, 다음 실행 때 추천 수량·진입가를
덮어쓴다. qty=0인 buy는 "체결 안 됨"으로 core/state.py의 apply_fill이 처리한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
FILLS_CSV = DATA_DIR / "fills.csv"

_COLUMNS = ["date", "ticker", "unit", "side", "price", "qty"]


def load_fills(path: Path = FILLS_CSV) -> pd.DataFrame:
    """fills.csv를 읽는다. 파일이 없거나 헤더만 있으면 빈 DataFrame.

    출력: DataFrame(date, ticker, unit, side, price, qty). date는 Timestamp,
         unit은 문자열("1"/"2"/"6"/"9")로 정규화한다.
    """
    if not path.exists():
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.read_csv(path, dtype={"unit": str})
    if df.empty:
        return pd.DataFrame(columns=_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    df["unit"] = df["unit"].astype(str)
    df["side"] = df["side"].str.strip().str.lower()
    return df


def fills_for(df: pd.DataFrame, ticker: str, date) -> list[dict]:
    """특정 종목·날짜의 체결 기록을 행 dict 목록으로 반환한다."""
    if df.empty:
        return []
    match = df[(df["ticker"] == ticker) & (df["date"] == pd.Timestamp(date))]
    return match.to_dict(orient="records")
