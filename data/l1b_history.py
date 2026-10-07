"""L1b 장기 과거 데이터 로더 (docs/l1b_plan.md 데이터 절).

- 케네스 프렌치 일별 요인(Mkt-RF, RF)과 배당수익률 포트폴리오(월별 VW Hi 30)를 zip으로 받아 data/cache/french/에 둔다.
- 나스닥 종합(^IXIC)은 yfinance auto_adjust=False Close를 받아 data/cache/l1b/에 둔다.
모든 결과는 1998-12-31까지 자르고 봉인 검사(core.l1b.cut_and_seal)를 거친다. 네트워크 실패는 예외로 멈춘다.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from core import l1b

CACHE = Path(__file__).resolve().parent / "cache"
FRENCH_DIR = CACHE / "french"
L1B_DIR = CACHE / "l1b"
FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
DAILY_ZIP = "F-F_Research_Data_Factors_daily_CSV.zip"
DP_ZIP = "Portfolios_Formed_on_D-P_CSV.zip"


def _french_text(zip_name: str) -> str:
    """zip을 캐시에서 읽거나 내려받아 첫 CSV 내용을 문자열로 준다. 실패하면 RuntimeError."""
    path = FRENCH_DIR / zip_name
    if not path.exists():
        import requests

        resp = requests.get(FRENCH_BASE + zip_name, timeout=60)
        if resp.status_code != 200 or not resp.content:
            raise RuntimeError(f"프렌치 데이터 내려받기 실패: {zip_name} (HTTP {resp.status_code})")
        FRENCH_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
    with zipfile.ZipFile(path) as z:
        return z.read(z.namelist()[0]).decode("latin1")


def parse_french_daily(text: str) -> pd.DataFrame:
    """프렌치 일별 요인 CSV 문자열 → DataFrame(mkt_rf, rf, 소수). 입력: 파일 내용 / 출력: 날짜 인덱스"""
    lines = text.splitlines()
    head = next(i for i, ln in enumerate(lines) if ln.replace(" ", "").startswith(",Mkt-RF"))
    rows = []
    for ln in lines[head + 1:]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 5 or not parts[0].isdigit() or len(parts[0]) != 8:
            if rows:
                break
            continue
        rows.append((parts[0], float(parts[1]), float(parts[4])))
    df = pd.DataFrame(rows, columns=["date", "mkt_rf", "rf"])
    df.index = pd.to_datetime(df.pop("date"), format="%Y%m%d")
    return df / 100.0


def parse_french_dp_hi30(text: str) -> pd.Series:
    """D/P 포트폴리오 CSV → "Value Weight Returns -- Monthly" Hi 30 월 수익(소수). 인덱스 = 그달 1일.

    결측 표시(−99.99, −999)는 NaN으로 바꾼다(판정 구간 검사는 호출부).
    """
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("Value Weight Returns -- Monthly"))
    header = [h.strip() for h in lines[start + 1].split(",")]
    col = header.index("Hi 30")
    vals = {}
    for ln in lines[start + 2:]:
        parts = [p.strip() for p in ln.split(",")]
        if not parts[0].isdigit() or len(parts[0]) != 6:
            break
        v = float(parts[col])
        vals[pd.Timestamp(f"{parts[0][:4]}-{parts[0][4:]}-01")] = None if v in (-99.99, -999.0) else v / 100.0
    return pd.Series(vals, dtype=float)


def load_french_daily() -> pd.DataFrame:
    """프렌치 일별 요인(1926-07~1998-12, 봉인 검사 통과)."""
    return l1b.cut_and_seal(parse_french_daily(_french_text(DAILY_ZIP)), "french_daily")


def load_hi30_monthly() -> pd.Series:
    """프렌치 D/P VW Hi 30 월 수익(~1998-12, 봉인 검사 통과)."""
    return l1b.cut_and_seal(parse_french_dp_hi30(_french_text(DP_ZIP)), "hi30_monthly")


def load_ixic() -> pd.Series:
    """^IXIC 일별 Close(auto_adjust=False), 야후 첫 행 ~ 1998-12-31. 캐시가 없으면 받는다. 실패하면 RuntimeError."""
    path = L1B_DIR / "IXIC_close.csv"
    if path.exists():
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        import yfinance as yf

        df = yf.download("^IXIC", start="1960-01-01", end="1999-01-01", auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            raise RuntimeError("^IXIC 내려받기 실패(빈 결과)")
        close = df["Close"]
        s = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
        s = s.dropna()
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        L1B_DIR.mkdir(parents=True, exist_ok=True)
        s.rename("close").to_csv(path)
    s.name = "close"
    return l1b.cut_and_seal(s.sort_index().astype(float), "ixic")
