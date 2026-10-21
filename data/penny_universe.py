"""10번 「동전주 거래량 수렴 전략」 입력 — Nasdaq Trader 공개 목록·yfinance 일봉 (docs/prereg/penny_volume_squeeze.md 2-1).

- 목록: nasdaqlisted.txt(나스닥), otherlisted.txt(NYSE·NYSE American만). 무료 공개 파일, 키 없음.
- 보통주 판별은 제외 방식(parse_lists — 순수 함수, 테스트 대상).
- 가격: 분기 선정일에만 목록 전체를 yfinance로 200종목씩 묶어 받는다(download_batch). 평일은 보유 종목만(history).
네트워크 실패는 조용히 넘기지 않는다 — 실패 종목·사유를 모아 돌려준다.
"""

from __future__ import annotations

import io
import re
import time
from datetime import date

import pandas as pd

LIST_URLS = {"nasdaq": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
             "other": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"}
OTHER_EXCHANGES = {"N", "A"}  # NYSE, NYSE American (P=Arca·Z=Cboe·V=IEX 등 제외)
NAME_EXCLUDE = re.compile(
    r"\b(?:warrants?|units?|rights?|preferred|preference|pfd|prd|notes?|debentures?|bonds?|fund|closed end|etns?|"
    r"certificates?|certs)\b|%|interest in a", re.I)
OTHER_SYMBOL = re.compile(r"[A-Z]+(\.[A-Z])?")
NASDAQ_SYMBOL = re.compile(r"[A-Z]+")


class PennyListError(RuntimeError):
    """목록을 받지 못했거나 비어 있을 때 — 그날은 선정하지 않는다."""


def _read(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), sep="|", dtype=str).fillna("")
    first = df.columns[0]
    return df[~df[first].str.startswith("File Creation Time")]


def parse_lists(nasdaq_text: str, other_text: str) -> list[str]:
    """두 목록 원문 → 대상 보통주 yfinance 기호 목록(정렬, 중복 없음).

    nasdaqlisted: Test Issue N·ETF N·NextShares≠Y / otherlisted: Exchange N·A, ETF N, Test Issue N, 기호 [A-Z]+(.[A-Z])?
    (점 뒤 U·W·R 제외). 이름에 워런트·유닛·권리·우선주·노트·펀드 등 낱말이 있으면 제외. 점은 '-'로(BRK.B → BRK-B).
    예외: 어느 한쪽이라도 형식이 틀리거나 남는 종목이 없으면 PennyListError
    """
    try:
        n, o = _read(nasdaq_text), _read(other_text)
        n = n[(n["Test Issue"] == "N") & (n["ETF"] == "N") & (n["NextShares"] != "Y")]
        o = o[(o["ETF"] == "N") & (o["Test Issue"] == "N") & (o["Exchange"].isin(OTHER_EXCHANGES))]
    except (KeyError, ValueError, pd.errors.ParserError) as exc:
        raise PennyListError(f"목록 형식이 다릅니다: {type(exc).__name__}: {exc}") from exc
    out = set()
    for sym, name in zip(n["Symbol"], n["Security Name"]):
        if NASDAQ_SYMBOL.fullmatch(sym) and not NAME_EXCLUDE.search(name):
            out.add(sym)
    for sym, name in zip(o["ACT Symbol"], o["Security Name"]):
        if not OTHER_SYMBOL.fullmatch(sym) or ("." in sym and sym[-1] in "UWR") or NAME_EXCLUDE.search(name):
            continue
        out.add(sym.replace(".", "-"))
    if not out:
        raise PennyListError("목록에 남는 보통주가 없습니다")
    return sorted(out)


def fetch_lists(timeout: int = 30, get=None) -> list[str]:
    """두 목록을 받아 parse_lists. 예외: PennyListError(받기 실패·빈 목록)."""
    if get is None:
        import requests

        get = requests.get
    texts = {}
    for k, url in LIST_URLS.items():
        try:
            r = get(url, timeout=timeout)
            r.raise_for_status()
            texts[k] = r.text
        except Exception as exc:  # noqa: BLE001 — 사유를 그대로 올린다
            raise PennyListError(f"{url.rsplit('/', 1)[-1]} 받기 실패: {type(exc).__name__}") from exc
    return parse_lists(texts["nasdaq"], texts["other"])


def download_batch(tickers: list[str], start: date, end_inclusive: date, chunk: int = 200, download=None,
                   pause: float = 1.0) -> tuple[dict[str, pd.DataFrame], dict[str, str], float]:
    """목록 전체 일봉(close·volume, auto_adjust=False)을 묶어 받는다.

    출력: ({티커: DataFrame(close, volume)}, {실패 티커: 사유}, 걸린 초)
    """
    if download is None:
        import yfinance as yf

        download = yf.download
    t0 = time.time()
    out, failed = {}, {}
    end = (pd.Timestamp(end_inclusive) + pd.Timedelta(days=1)).date().isoformat()
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        try:
            df = download(part, start=pd.Timestamp(start).date().isoformat(), end=end, auto_adjust=False, actions=False,
                          group_by="ticker", threads=True, progress=False)
        except Exception as exc:  # noqa: BLE001
            failed.update({t: f"{type(exc).__name__}" for t in part})
            continue
        for t in part:
            try:
                sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
                f = pd.DataFrame({"close": sub["Close"].astype(float), "volume": sub["Volume"].astype(float)})
            except KeyError:
                failed[t] = "가격 없음"
                continue
            idx = pd.DatetimeIndex(f.index)
            f.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
            f = f[f["close"].notna()]
            if f.empty:
                failed[t] = "가격 없음"
                continue
            out[t] = f[~f.index.duplicated(keep="last")]
        if pause and i + chunk < len(tickers):
            time.sleep(pause)
    return out, failed, time.time() - t0


def history(ticker: str, start: date) -> pd.DataFrame:
    """한 종목 일봉 + 분배금·분할 (yfinance Ticker.history, auto_adjust=False) → DataFrame(close, volume, dividend, split)."""
    import yfinance as yf

    raw = yf.Ticker(ticker).history(start=pd.Timestamp(start).date().isoformat(), auto_adjust=False, actions=True)
    if raw is None or raw.empty or "Close" not in raw.columns:
        raise ValueError("가격 없음")
    idx = pd.DatetimeIndex(raw.index)
    raw = raw.set_axis((idx.tz_localize(None) if idx.tz is not None else idx).normalize())
    raw = raw[~raw.index.duplicated(keep="last")]
    z = pd.Series(0.0, index=raw.index)
    return pd.DataFrame({"close": raw["Close"].astype(float), "volume": raw.get("Volume", z).astype(float).fillna(0.0),
                         "dividend": raw.get("Dividends", z).astype(float).fillna(0.0),
                         "split": raw.get("Stock Splits", z).astype(float).fillna(0.0)})
