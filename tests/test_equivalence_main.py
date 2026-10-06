"""동등성: 스크리닝 화면·"왜?" 설명 작업(feat/why-explain)이 신호·수량·손절가를 바꾸지 않았는지.

tests/fixtures/equivalence_main.json은 main(94bea29)에서 이 파일의 _snapshot()으로 만든
값이다. 같은 고정 입력(합성 24종목 × 150거래일, paper 가상 체결 + live 계획 사이징)에서
매수·매도 이벤트, 가상 체결 수량, 손절가, 보고서 매수·매도 줄이 그대로인지 비교한다.

다시 만들기(신호 규칙을 일부러 바꿨을 때만): python -m tests.test_equivalence_main
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from core import state as st
from data.fills import FillsResult
from data.fx import FxRateResult
from engine.daily import build_report_summary, simulate_since
from tests._synthetic_universe import synthetic_indicator_map

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "equivalence_main.json"
_FX = 1400.0


def _cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def _r(v):
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else (round(float(v), 4) if isinstance(v, float) else v)


def _run(cfg, im, upto: int):
    """마지막 150거래일 중 upto번째 날까지 paper 가상 체결로 돌린다."""
    any_df = next(iter(im.values()))
    days = list(any_df.index[-150:])[: upto + 1]
    dates = {t: days for t in im}
    states = {t: st.init_state(t, t) for t in im}
    fx = {d.date().isoformat(): _FX for d in any_df.index}
    earnings = {t: None for t in im}
    earnings["T03"] = days[-1] + pd.Timedelta(days=1)  # 실적 필터가 실제로 걸리게
    sim = simulate_since(
        im, dates, states, cfg, earnings, {t: set() for t in im}, pd.DataFrame(),
        virtual_fill=True, max_concurrent=cfg["risk"]["max_concurrent_positions"], fx_rate_by_date=fx,
    )
    return sim, earnings


def _event_view(e: dict) -> dict:
    keep = ("date", "ticker", "kind", "stage", "unit", "price", "qty", "score", "grade", "reasons", "blocked_type", "stop_after", "state_after", "old_stop", "new_stop")
    return {k: (str(pd.Timestamp(e[k]).date()) if k == "date" else _r(e[k])) for k in keep if k in e}


def _summary_view(summary: dict) -> dict:
    buys = [
        {k: _r(r.get(k)) for k in ("ticker", "stage", "limit", "stop", "stop_pct", "qty", "amount_krw", "max_loss_krw", "decision", "score", "grade", "note")}
        for key in ("b1", "b2", "b3", "b9") for r in summary["buy_groups"][key]
    ]
    sells = [{k: _r(r.get(k)) for k in ("티커", "kind", "수량", "평균단가", "종가")} for r in summary["sell_rows"]]
    filtered = [{k: _r(r.get(k)) for k in ("티커", "단계", "유형", "사유", "점수")} for r in summary["filtered_rows"]]
    return {"buys": buys, "sells": sells, "filtered": filtered, "funnel": summary["funnel"]}


def _summary(mode, cfg, im, sim, earnings):
    names = {t: t for t in im}
    kwargs = {}
    if mode == "live":
        kwargs = {"plan_by_ticker": {"T01": 5_000_000.0, "T05": 3_000_000.0}, "default_budget_krw": 4_000_000.0}
    return build_report_summary(
        mode, cfg, im, names, earnings, sim["states"], sim["today_events"], sim["as_of_by_ticker"], [],
        FillsResult(), [], cfg["risk"]["max_concurrent_positions"],
        FxRateResult(_FX, "2026-01-01", False, None), **kwargs,
    )


def _snapshot() -> dict:
    cfg = _cfg()
    im = synthetic_indicator_map(cfg)
    sim, earnings = _run(cfg, im, 149)
    events = [_event_view(e) for e in sim["all_events"]]
    # 매수 이벤트가 많은 날 3곳에서 보고서 매수·매도 줄(수량·손절가 포함)을 비교한다.
    buy_days = pd.Series([e["date"] for e in events if e["kind"] in ("A1", "A2", "A3", "B")]).value_counts()
    any_df = next(iter(im.values()))
    days = list(any_df.index[-150:])
    reports = {}
    for day_str in sorted(buy_days.index[:3]):
        upto = days.index(pd.Timestamp(day_str))
        sim_d, earn_d = _run(cfg, im, upto)
        reports[day_str] = {mode: _summary_view(_summary(mode, cfg, im, sim_d, earn_d)) for mode in ("paper", "live")}
    final_states = {
        t: {"state": s["state"], "units": s["units"], "stop": _r(s.get("stop")), "a1_date": str(s["a1_date"]) if s.get("a1_date") is not None else None}
        for t, s in sim["states"].items()
    }
    return {"events": events, "reports": reports, "final_states": final_states}


def test_signals_qty_stops_match_main():
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    actual = json.loads(json.dumps(_snapshot(), ensure_ascii=False, default=str))
    assert actual["events"] == expected["events"]
    assert actual["final_states"] == expected["final_states"]
    assert actual["reports"] == expected["reports"]


if __name__ == "__main__":
    FIXTURE.parent.mkdir(exist_ok=True)
    FIXTURE.write_text(json.dumps(_snapshot(), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"저장: {FIXTURE}")
