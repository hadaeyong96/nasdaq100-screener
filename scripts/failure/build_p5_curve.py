"""실패 구간 분석기 1단계 — MACD 1:2:6 신호(P5, B0) 일간 곡선.

2012~2021로 다시 돌린 백테스트(engine.backtest --start 2012-01-01, outputs/backtest/2026-10-22_21eb8626)는
총자산이 음수가 되는 등(MDD −475%) 결과가 성립하지 않았다(2012 시작용으로 만든 엔진이 아님 — 가격도 2014년부터 받음).
그래서 기존 P5 실행(2015~2021, outputs/backtest/2026-09-25_92de90b0, 문서 B0 세후 23.57%)의 equity.csv를 그대로 쓴다.
기준 = QQQ 수정주가(분배금 재투자) × 원/달러(DEXKOUS) — 원화, 세전.
실행: python scripts/failure/build_p5_curve.py <출력 폴더>
"""
import json
import sys
from pathlib import Path

import pandas as pd

MAIN = Path(r"C:\projects\nasdaq100-screener")
OUT = Path(sys.argv[1])
eq = pd.read_csv(MAIN / "outputs/backtest/2026-09-25_92de90b0/equity.csv", index_col=0, parse_dates=True, encoding="utf-8-sig")
eq = eq[eq.index <= "2021-12-31"]
q = pd.read_csv(Path.cwd() / "data/cache/failure/ind_raw/yf_QQQ.csv", index_col=0, parse_dates=True)["Adj Close"]
fx = pd.read_csv(MAIN / "data/cache/shock_dip/fred_DEXKOUS.csv", index_col=0, parse_dates=True).iloc[:, 0]
fx = pd.to_numeric(fx, errors="coerce")
fx = fx[fx.index <= "2021-12-31"].ffill().reindex(eq.index, method="ffill")
bench = q.reindex(eq.index, method="ffill") * fx
pd.DataFrame({"strategy": eq["total_krw"], "benchmark": bench}, index=pd.DatetimeIndex(eq.index, name="date")).to_csv(OUT / "MACD126_P5__QQQ기준.csv")
yrs = (eq.index[-1] - eq.index[0]).days / 365.25
chk = {"전략": "MACD 1:2:6 P5 B0", "기존 실행 총자산 연복리(세후 b 근사)": (eq["total_krw"].iat[-1] / eq["total_krw"].iat[0]) ** (1 / yrs) - 1,
       "문서 값": "B0 세후(b) 23.57%, QQQ 23.62%", "기간": f"{eq.index[0].date()} ~ {eq.index[-1].date()}",
       "2012~2021 다시 돌리기": "실패 — 총자산 음수(MDD −475%), 가격 2014년부터 — 기존 2015~2021 실행 결과로 대체"}
(OUT / "p5_check.json").write_text(json.dumps(chk, ensure_ascii=False, indent=1), encoding="utf-8")
print(chk)
