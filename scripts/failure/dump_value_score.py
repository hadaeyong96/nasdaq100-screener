"""실패 구간 분석기 1단계 — 가치평가(value-score, 검증 ③) 일간 곡선 다시 뽑기 (research/value-score 코드 그대로).

research/value-score 작업 트리에서 실행: python <이 파일> <출력 폴더>
원래 시험 그대로(검증 ③ 대상, 2012~2021, 1억 + 매달 100만 원). 곡선은 장부의 단위 가격(세전, 납입 영향 없음).
출력: 가치평가__QQQ기준.csv (date, strategy, benchmark), value_check.json(세후 XIRR 다시 뽑기)
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

from scripts import value_score_zone3_run as run  # noqa: E402

vd = run.vd
data = vd.Data()
try:
    scores = vd.load_scores(data, rebuild=False)
except TypeError:
    scores = vd.load_scores(data)
m = run.Market(data)
board = run.Board(scores, scores["total"], data.sic2)
s = run.run_value(m, board, run.FULL, record=True)
q = run.run_qqq(m, run.FULL, record=True)
sp, _ = run.curves(s, m)
qp, _ = run.curves(q, m)
df = pd.DataFrame({"strategy": sp, "benchmark": qp})
df.index.name = "date"
df.to_csv(OUT / "가치평가__QQQ기준.csv")
chk = {"전략": "가치평가", "다시 뽑은 세후 XIRR": run.xirr(s["book"].flows), "QQQ 세후 XIRR": run.xirr(q["book"].flows)}
(OUT / "value_check.json").write_text(json.dumps(chk, ensure_ascii=False, indent=1), encoding="utf-8")
print("done", chk)
