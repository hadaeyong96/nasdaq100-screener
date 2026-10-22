"""실패 구간 분석기 1단계 — 강한 업종 MACD 눌림(탐색 B, B0) 일간 곡선 다시 뽑기 (research/sector-pullback 코드 그대로).

research/sector-pullback 작업 트리에서 실행: python <이 파일> <출력 폴더>
출력: 강한업종눌림B0__QQQ기준.csv (date, strategy, benchmark — 세금 전 일별 총자산, 비용 0.1%), sector_check.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
OUT = Path(sys.argv[1])

from core import sector_pullback as sp  # noqa: E402
from scripts import sector_pullback_run as run  # noqa: E402

qqq = run.load_qqq()
mk, comps = run.pit_market(qqq)
s, e = mk.days.get_loc(run.START), mk.days.get_loc(run.END)
c1 = sp.Costs(0.001)
q1 = sp.run_qqq(mk, s, e, c1)
vd = sp.variant_data(mk, "B0")
r1 = sp.run_book(mk, s, e, c1, events=sp.signal_events(mk, vd, s, e), exits=sp.exit_flags(mk, vd))
days = mk.days[s:e + 1]
pd.DataFrame({"strategy": r1["curve"], "benchmark": q1["curve"]}, index=pd.DatetimeIndex(days, name="date")).to_csv(OUT / "강한업종눌림B0__QQQ기준.csv")
chk = {"전략": "강한 업종 MACD 눌림 B0", "다시 뽑은 세후 연복리": sp.cagr(run.INITIAL, r1["final"], days[0], days[-1]),
       "QQQ 세후 연복리": sp.cagr(run.INITIAL, q1["final"], days[0], days[-1])}
(OUT / "sector_check.json").write_text(json.dumps(chk, ensure_ascii=False, indent=1), encoding="utf-8")
print("done", chk)
