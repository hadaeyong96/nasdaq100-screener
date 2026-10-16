"""전략공장 F1: 시장 국면 달력 만들기 (보고용 — 판정 계산에 쓰지 않는다).

    python scripts/factory_regime_calendar.py

QQQ 일별 종가(yfinance auto_adjust=False Close, 분할 반영·배당 미반영) 1999-03-10 ~ 마지막 확정 봉을 받아
factory.regimes.classify로 국면을 붙인다. 출력: docs/factory/regime_calendar.csv, docs/factory/regime_summary.md.
기존 백테스트 캐시(data/cache/backtest/QQQ.parquet, 1999-03-10 ~ 2021-12-31)와 겹치는 구간 종가 차이도 보고한다.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factory import regimes as rg  # noqa: E402

CACHE = ROOT / "data" / "cache" / "factory" / "QQQ_daily.csv"
BT_CACHE = ROOT / "data" / "cache" / "backtest" / "QQQ.parquet"
CAL = ROOT / "docs" / "factory" / "regime_calendar.csv"
SUMMARY = ROOT / "docs" / "factory" / "regime_summary.md"
SHORT_DAYS = 60
ORDER = [rg.UP, rg.DOWN, rg.CRASH, rg.UNKNOWN]


def fetch_qqq() -> pd.Series:
    """yfinance QQQ Close. 미국장이 열려 있거나 아직 안 닫힌 오늘(동부시간) 봉은 버린다(확정 봉만)."""
    import yfinance as yf

    h = yf.Ticker("QQQ").history(start="1999-03-01", auto_adjust=False)
    if h is None or h.empty:
        raise SystemExit("yfinance QQQ 받기 실패")
    s = h["Close"]
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    now_et = datetime.now(ZoneInfo("America/New_York"))
    today = pd.Timestamp(now_et.date())
    if s.index[-1] >= today and now_et.hour < 17:  # 16:00 마감 + 여유
        s = s[s.index < today]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    s.rename("close").to_csv(CACHE, index_label="date")
    return s


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    q = fetch_qqq()
    print(f"[regime] QQQ {q.index[0].date()} ~ {q.index[-1].date()} ({len(q)}일)")
    check = ""
    if BT_CACHE.exists():
        b = pd.read_parquet(BT_CACHE)["close"]
        j = pd.concat([q.rename("yf"), b.rename("bt")], axis=1, join="inner").dropna()
        diff = (j["yf"] / j["bt"] - 1).abs()
        check = (f"기존 백테스트 캐시와 겹치는 {len(j)}일({j.index[0].date()} ~ {j.index[-1].date()}) 종가 상대 차이: "
                 f"최대 {diff.max() * 100:.4f}%, 0.01% 넘는 날 {(diff > 1e-4).sum()}일")
        print(f"[regime] {check}")
    reg = rg.classify(q)
    pd.DataFrame({"date": reg.index.strftime("%Y-%m-%d"), "regime": reg.values}).to_csv(CAL, index=False, encoding="utf-8")

    yearly = pd.crosstab(reg.index.year, reg).reindex(columns=ORDER, fill_value=0)
    yearly.loc["합계"] = yearly.sum()
    seg = rg.segments(reg)
    L = ["# 시장 국면 달력 요약 (QQQ, 보고용)", "",
         f"- 만든 날: {datetime.now(ZoneInfo('Asia/Seoul')).date()} · `python scripts/factory_regime_calendar.py` · 규칙 `factory/regimes.py`",
         f"- 데이터: yfinance QQQ `Close`(auto_adjust=False, 분할 반영·배당 미반영) {q.index[0].date()} ~ {q.index[-1].date()} ({len(q)}거래일), "
         "확정 봉만",
         f"- {check}" if check else "- 기존 캐시 비교 없음",
         "- 규칙: 폭락장(252거래일 최고가 대비 −20% 이하 또는 20거래일 −10% 이하, 우선) > 하락장(종가 < 200일선) > 상승장. "
         "200거래일 전은 판단불가. 그날 종가까지만 사용",
         "- 이 달력은 보고용이다. 2022-01 이후(봉인 구간)는 판정 계산에 쓰지 않는다", "",
         "## 연도별·국면별 일수", "", "| 연도 | " + " | ".join(ORDER) + " | 합 |", "| --- | " + " | ".join(["---"] * (len(ORDER) + 1)) + " |"]
    for y, r in yearly.iterrows():
        L.append(f"| {y} | " + " | ".join(str(int(r[c])) for c in ORDER) + f" | {int(r.sum())} |")
    L += ["", f"## 국면 구간 ({len(seg)}개, ★ = {SHORT_DAYS}거래일 미만)", "",
          f"국면이 바뀐 날 = 각 구간의 시작일. {SHORT_DAYS}거래일 미만 구간 {int((seg['days'] < SHORT_DAYS).sum())}개.", "",
          "| # | 시작(바뀐 날) | 끝 | 국면 | 이전 국면 | 거래일 | |", "| --- | --- | --- | --- | --- | --- | --- |"]
    prev = None
    for i, r in seg.iterrows():
        L.append(f"| {i + 1} | {r['start'].date()} | {r['end'].date()} | {r['regime']} | {prev or '-'} | {r['days']} | "
                 f"{'★' if r['days'] < SHORT_DAYS else ''} |")
        prev = r["regime"]
    SUMMARY.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(yearly.to_string())
    print(f"[regime] 구간 {len(seg)}개, {SHORT_DAYS}거래일 미만 {int((seg['days'] < SHORT_DAYS).sum())}개 → {CAL.relative_to(ROOT)}, "
          f"{SUMMARY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
