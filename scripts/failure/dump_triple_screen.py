"""실패 구간 분석기 1단계 — 상승장 RSI 눌림(3단 거르기 A1) 일간 곡선 다시 뽑기 (research/triple-screen 코드 그대로).

research/triple-screen 작업 트리에서 실행: python <이 파일> <출력 폴더>
출력: 상승장RSI눌림__QQQ기준.csv (date, strategy, benchmark — 세후 규칙 장부의 세전 원화 평가액), triple_check.json(세후 XIRR 다시 뽑기)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
for p in (ROOT, ROOT / "scripts"):
    sys.path.insert(0, str(p))
OUT = Path(sys.argv[1])

import triple_screen_run as tr  # noqa: E402
from core import triple_screen as ts  # noqa: E402

mk = tr.load_market()
cal = tr.cal_of(mk)
s = ts.run_strategy(mk, ts.A1, ts.FULL, cal)
q = ts.run_qqq_hold(mk, ts.FULL, cal)
pd.DataFrame({"strategy": s["book"].curve_pre, "benchmark": q["book"].curve_pre},
             index=pd.DatetimeIndex(mk.days, name="date")).to_csv(OUT / "상승장RSI눌림__QQQ기준.csv")
chk = {"전략": "상승장 RSI 눌림(A1)", "다시 뽑은 세후 XIRR": ts.xirr(s["book"].flows), "QQQ 세후 XIRR": ts.xirr(q["book"].flows),
       "매매": [{**{k: (int(v) if hasattr(v, "item") and float(v).is_integer() else (float(v) if hasattr(v, "item") else v))
                   for k, v in t.items() if k in ("col", "fill", "exit", "pnl_krw", "why")}} for t in s["book"].trades][:0]}
(OUT / "triple_check.json").write_text(json.dumps(chk, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print("done", chk["다시 뽑은 세후 XIRR"], chk["QQQ 세후 XIRR"])
