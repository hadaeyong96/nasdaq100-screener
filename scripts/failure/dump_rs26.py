"""실패 구간 분석기 1단계 — 6개월 강세(RS26, 검증 ③) 일간 곡선 다시 뽑기 (research/rs26-zone3 코드 그대로).

research/rs26-zone3 작업 트리에서 실행: python <이 파일> <출력 폴더>
대상은 원래 시험 그대로(검증 ③: S&P 500 − (나스닥100 ∪ S&P 100), 2012~2021). 비용 0.1%.
출력: 6개월강세RS26__QQQ기준.csv (date, strategy, benchmark — 세금 전 일별 총자산), rs26_check.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = Path(sys.argv[1])

from scripts import rs26_zone3_run as run  # noqa: E402

sig_days = run.monthly_signals()
zone = run.Zone(sig_days, monthly=True)
qc, qd = run.load_qqq()
mat = run.Mat(zone, run.nyse_days())
strat, elig = run.build_signals("rs26", zone, qc, sig_days)
first = sig_days[0]
s = run.simulate(mat, strat, 0.1)
q = run.simulate_qqq(qc, qd, first, 0.1)
df = pd.DataFrame({"strategy": s["curve"], "benchmark": q["curve"]}).dropna()
df.index.name = "date"
df.to_csv(OUT / "6개월강세RS26__QQQ기준.csv")
chk = {"전략": "6개월 강세 RS26", "다시 뽑은 세후 연복리": run.cagr(run.INITIAL, s["final"], first, run.END),
       "QQQ 세후 연복리": run.cagr(run.INITIAL, q["final"], first, run.END)}
(OUT / "rs26_check.json").write_text(json.dumps(chk, ensure_ascii=False, indent=1), encoding="utf-8")
print("done", chk)
