"""실패 구간 분석기 1단계 — 지수 타이밍 2개·자산 배분 2개 일간 곡선 다시 뽑기 (research/vol-target 코드 그대로).

research/vol-target 작업 트리에서 실행한다(그 브랜치의 scripts/vol_target_run.py·two_more_run.py 함수를 그대로 부른다 — 규칙 변경 없음).
    python <이 파일> <출력 폴더>
출력: <폴더>/<전략>__<구간>.csv (date, strategy, benchmark — 세전 일별 가치), <폴더>/index_alloc_check.json(세후 포함 최종 연복리 다시 뽑기)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

from scripts import vol_target_run as vr  # noqa: E402  (sealed_guard.install()이 들어 있다)
from scripts import two_more_run as tr  # noqa: E402
from scripts.shock_dip_gate3 import asof_series  # noqa: E402
from scripts import sealed_guard  # noqa: E402
from core import vol_target as vt  # noqa: E402
from core import multi_book as mb  # noqa: E402
from core import two_more as tm  # noqa: E402


def save(name, period, d, strat, bench):
    pd.DataFrame({"strategy": strat, "benchmark": bench}, index=pd.DatetimeIndex(d, name="date")).to_csv(OUT / f"{name}__{period}.csv")


def main() -> None:
    dtb3 = sealed_guard.load_fred("DTB3")
    fx_all = sealed_guard.load_fred("DEXKOUS")
    check = []
    US = [("S&P500", "^GSPC", "1977-01-01", "1998-12-31"), ("나스닥 종합", "^IXIC", "1977-01-01", "1998-12-31"),
          ("QQQ(보정)", "QQQ", "2000-01-01", "2021-12-31")]
    for name, tk, start, end in US:
        d, close, prices, w0, _ = vr.market(name, tk, start, end, dtb3)
        fx = asof_series(fx_all, d).to_numpy() if tk == "QQQ" else None
        dd = d[w0:]
        base = vr.book(prices, d, w0, {}, lambda a, b: False, None, (0.0, 1.0, 0.0))[0].nav
        # 200일선 2배 전환(아래 현금)
        above = (close > close.rolling(200).mean()).to_numpy()
        dec = {p: (np.array([0.0, 0.0, 1.0]) if above[p] else np.array([1.0, 0.0, 0.0])) for p in range(w0, len(d))}
        sw_should = lambda a, b: np.abs(a - b).max() > 0.5  # noqa: E731
        sw_start = (0.0, 0.0, 1.0) if above[w0 - 1] else (1.0, 0.0, 0.0)
        sw_pre = vr.book(prices, d, w0, dec, sw_should, None, sw_start)[0].nav
        save("200일선2배전환", name, dd, sw_pre, base)
        check.append({"전략": "200일선 2배 전환", "구간": name, "다시 뽑은 값": vr.sma_switch(close, prices, d, w0, fx)["val"]})
        # 변동성 맞춤 V1
        pos = vt.week_end_positions(d, w0)
        es = vt.target_exposures(close, pos, 0.20, False)
        v1_pre = vr.book(prices, d, w0, {p: vt.weights_of(e) for p, e in zip(pos, es)}, vt.should_rebalance(0.2), None)[0].nav
        save("변동성맞춤V1", name, dd, v1_pre, base)
        check.append({"전략": "변동성 맞춤 V1", "구간": name, "다시 뽑은 값": vr.run_vt(close, prices, d, w0, fx, 0.20, False)["val"]})
    win = {"S&P500": ("1977-01-01", "1998-12-31"), "QQQ(보정)": ("2000-01-01", "2021-12-31")}
    for name, (s, e) in win.items():
        m = tr.build(name, dtb3)
        tax = name == "QQQ(보정)"
        base = tr.stock_hold(m, s, e, fx_all, False)
        sw = tr.book(m, s, e, tr.switch_decisions(m, s, 12), tr.changed, fx_all, tm.one_hot(tm.STOCK), False)
        bl = tr.balanced(m, s, e, fx_all, False)
        d = m["d"]
        w0 = int(np.searchsorted(d, pd.Timestamp(s)))
        w1 = int(np.searchsorted(d, pd.Timestamp(e), side="right"))
        dd = d[w0:w1]
        save("갈아타기", name, dd, sw["nav"], base["nav"])
        save("2배균형", name, dd, bl["nav"], base["nav"])
        check.append({"전략": "주식·채권·금 갈아타기" + ("" if tax else "(금 없이, 참고)"), "구간": name,
                      "다시 뽑은 값": tr.book(m, s, e, tr.switch_decisions(m, s, 12), tr.changed, fx_all, tm.one_hot(tm.STOCK), tax)["val"]})
        check.append({"전략": "주식·채권 2배 균형", "구간": name, "다시 뽑은 값": tr.balanced(m, s, e, fx_all, tax)["val"]})
    (OUT / "index_alloc_check.json").write_text(json.dumps(check, ensure_ascii=False, indent=1), encoding="utf-8")
    print("done", len(check))


if __name__ == "__main__":
    main()
