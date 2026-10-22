"""실패 구간 분석기 3단계 — 원인 후보 지표 받기 (무료, 2021-12-31까지만 요청·저장). 값·키는 출력하지 않는다.

FRED: 기존 무료 키(.env FRED_API_KEY)로 API(observation_end=2021-12-31). yfinance: end=2022-01-01(그 전날까지).
AAII 강세−약세: 공개 xls를 한 번 시도(막히면 "못 구함").
출력: data/cache/failure/ind_raw/*.csv, data/cache/failure/ind_fetch.json(받은 범위·실패)
실행: python scripts/failure/fetch_indicators.py   (작업 폴더 = research/failure-analysis 루트)
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / "data" / "cache" / "failure" / "ind_raw"
OUT.mkdir(parents=True, exist_ok=True)
FRED = ["DGS10", "DGS2", "T10Y2Y", "FEDFUNDS", "BAA", "DTWEXM", "DTWEXBGS", "UNRATE", "CPIAUCSL", "USREC", "VIXCLS"]
ETF = ["SPY", "RSP", "QQQ", "QQQE", "IWM", "IWD", "IWF", "^GSPC", "^IXIC", "^NDX"]


def env_key() -> str:
    from dotenv import dotenv_values

    for p in (ROOT / ".env", Path(r"C:\projects\nasdaq100-screener\.env")):
        if p.exists():
            k = dotenv_values(p).get("FRED_API_KEY")
            if k:
                return k
    return ""


def main() -> None:
    key = env_key()
    rep = {"fred": {}, "yf": {}, "aaii": None, "not_available": {}}
    for sid in FRED:
        q = urllib.parse.urlencode({"series_id": sid, "api_key": key, "file_type": "json", "observation_end": "2021-12-31"})
        try:
            with urllib.request.urlopen("https://api.stlouisfed.org/fred/series/observations?" + q, timeout=120) as r:
                obs = json.load(r)["observations"]
            s = pd.Series({o["date"]: (None if o["value"] == "." else float(o["value"])) for o in obs}, dtype=float)
            s.index = pd.to_datetime(s.index)
            s = s[s.index <= "2021-12-31"]
            s.rename(sid).to_csv(OUT / f"fred_{sid}.csv", index_label="date")
            v = s.dropna()
            rep["fred"][sid] = f"{v.index[0].date()} ~ {v.index[-1].date()} ({len(v)})"
        except Exception as exc:  # noqa: BLE001 — 키가 들어간 URL은 내보내지 않는다
            rep["fred"][sid] = f"실패 {type(exc).__name__}"
    import yfinance as yf

    for tk in ETF:
        df = yf.download(tk, start="1900-01-01", end="2022-01-01", auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            rep["yf"][tk] = "없음"
            continue
        out = pd.DataFrame({c: (df[c].iloc[:, 0] if isinstance(df[c], pd.DataFrame) else df[c]) for c in ["Close", "Adj Close"]})
        out.index = pd.DatetimeIndex(out.index).tz_localize(None)
        out = out[out.index <= "2021-12-31"]
        out.to_csv(OUT / f"yf_{tk.replace('^', '')}.csv", index_label="date")
        rep["yf"][tk] = f"{out.index[0].date()} ~ {out.index[-1].date()} ({len(out)})"
    try:
        req = urllib.request.Request("https://www.aaii.com/files/surveys/sentiment.xls", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read()
        if body[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":  # xls 시그니처
            (OUT / "aaii_sentiment.xls").write_bytes(body)
            rep["aaii"] = "받음(xls)"
        else:
            rep["aaii"] = "못 구함(엑셀 파일이 아님 — 로그인·차단 화면)"
    except Exception as exc:  # noqa: BLE001
        rep["aaii"] = f"못 구함({type(exc).__name__})"
    rep["not_available"] = {
        "QQQ 상위 10종목 비중": "무료 과거 보유 비중 자료 없음(운용사 일별 보유 파일은 최근만) — 빈칸",
        "QQQE/QQQ": "QQQE 2012-03~ 만(그 전 빈칸)", "RSP/SPY": "RSP 2003-04~ 만", "IWM·IWD·IWF": "2000-05~ 만",
        "VIX": "1990~ 만(VIXCLS)", "달러 지수": "DTWEXM 1973~2019-12, DTWEXBGS 2006~ — 2020~21은 DTWEXBGS 변화율로 이음"}
    (ROOT / "data" / "cache" / "failure" / "ind_fetch.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
