"""B0.5 재현 테스트 3단계 (AI 펀드 F2, docs/design/fund_sim.md 5.8, 사용자 지시 2026-09-30로 수정).

engine.backtest.simulate_portfolio(기존 B0.5=B0+슬롯5, docs/p5_3_instructions.md 9번과
완전히 같은 설정)를 여러 방식으로 돌려 세 숫자(재현 오차·생존 편향·체결 가정 영향)를
낸다. 새 시뮬레이션 로직을 만들지 않는다 — engine/backtest.py에 F2에서 추가한 옵션
(force_universe_mode, stop_next_day_open, apply_slippage)만 켜고 끄며 같은 엔진을 그대로 쓴다.

미리 확인한 사실(직접 코드 확인, docs/audits/2026-09-29-sizing.md): engine.backtest.
prepare_data는 시점별 구성 종목 재구성이 "성공하면" 기본값으로 그것을 쓰고, 실패할
때만 현재 구성 종목으로 자동 폴백한다 — 즉 기존에 기록된 +19.78%(세후 a, 2016~2021)는
이미 시점별 구성 종목(POINT_IN_TIME)으로 계산됐을 가능성이 크다(설계 문서 5.8이
가정한 "1번=현재 구성 종목"과 반대). 그래서 이 스크립트는 설계 문서의 1→2→3 순서를
그대로 따르되, 결과를 보고 실제로 어느 단계가 +19.78%와 가까운지 그대로 보고한다 —
추측으로 순서를 바꾸지 않는다.

단계:
1. force_universe_mode="CURRENT_CONSTITUENTS"(생존 편향 있음) + 기존 체결 가정
2. force_universe_mode=None(POINT_IN_TIME, 엔진 기본값) + 기존 체결 가정 — 기준
3-참고. POINT_IN_TIME + stop_next_day_open=True(손절만 다음날 시가, 슬리피지 없음)
   — 예전 버전의 3단계. 단일 변수(손절 타이밍)만 격리해서 보고 싶을 때 참고용으로 남긴다.
3. POINT_IN_TIME + stop_next_day_open=True + apply_slippage=True(모든 체결에 설계 5.2의
   슬리피지 0.05% 적용) — **정식 3단계**. 매도는 이제 전부(손절 포함) 다음날 시가 +
   슬리피지로 체결한다. 매수는 core/execution.py의 기존 규칙(익일 지정가=전일종가×1.01,
   시가가 그 이하면 시가)을 그대로 쓴다 — B0.5 전략 고유 규칙이라 core/ 수정 금지 원칙상
   건드리지 않았고, 그 체결가에도 슬리피지는 적용된다.

⚠️ **데이터 품질 미달 → 아래 모든 CAGR·MDD·샤프 숫자는 INCONCLUSIVE(참고용)이다.**
scripts/fund_dataqc_report.py 실행 결과(outputs/backtest/fund_dataqc/report.md), 학습
구간(2015~2021) 중 구성종목 가격 확보율이 98%를 넘는 해가 하나도 없다(68.8~90.0%,
전체 81.04%). 2단계(시점별 구성종목)에도 편출 종목이 그만큼 빠져 있으므로 "생존 편향
크기"는 신뢰할 수 있게 측정할 수 없다 — 1단계와 2단계 둘 다 각자 다른 방식으로 불완전한
유니버스를 쓴 결과이기 때문이다. 이 스크립트는 survivorship_bias_pct를 숫자로 내지 않고
"측정 불가" 텍스트로 대신한다.

봉인 구간(2022~)은 건드리지 않는다 — 이 재현은 학습 구간(2016~2021)만 쓴다
(train_end="2021-12-31", 기존 B0.5 축가 그대로).

실행(반드시 python -u로 — 진행 로그가 즉시 보이게):
    python -u -m scripts.fund_b05_reproduction

로그: outputs/backtest/fund_b05_reproduction/report.log
결과: outputs/backtest/fund_b05_reproduction/report.json
trials.db에도 각 단계를 한 줄씩 기록한다(store/trials.py, N_raw 로그).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from engine import backtest as bt  # noqa: E402
from store import trials as trials_store  # noqa: E402

OUT_DIR = bt.OUTPUT_ROOT / "fund_b05_reproduction"
LOG_PATH = OUT_DIR / "report.log"
RESULT_PATH = OUT_DIR / "report.json"
DATAQC_REPORT_PATH = bt.OUTPUT_ROOT / "fund_dataqc" / "report.json"

KNOWN_B05_POSTTAX_A_PCT = 19.78  # docs/audits/2026-09-29-sizing.md, 2016~2021, 세후(a)
MAX_SLOTS_B05 = 5  # "B0.5 = B0 + 슬롯 5" (docs/p5_3_instructions.md 9번)
INCONCLUSIVE_NOTE = "데이터 품질 미달 → INCONCLUSIVE(참고용). scripts/fund_dataqc_report.py 결과 참고."


def log(msg: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def pick_closest_stage(stage1_diff: float | None, stage2_diff: float | None) -> str:
    """기존 기록과의 오차(diff, %p)가 더 작은 쪽 단계 이름을 고른다 (순수 함수 — 테스트 대상).

    diff가 정확히 0.0(완전 일치)이어도 "값이 없다"로 취급하면 안 된다 — 예전 버전은
    `stage2_diff or 999`를 써서 stage2_diff==0.0을 "없음"으로 잘못 보고 항상 1단계를
    골랐다(파이썬에서 0.0은 falsy). 이번엔 None만 "없음"으로 본다.

    입력: stage1_diff, stage2_diff(각각 None 가능 — 그 단계 계산이 실패했을 때)
    출력: "1단계" 또는 "2단계"
    """
    if stage1_diff is None:
        return "2단계"
    if stage2_diff is None:
        return "1단계"
    return "1단계" if abs(stage1_diff) <= abs(stage2_diff) else "2단계"


def _prepare(label: str, cfg: dict, warmup_start: date, end: date, *, force_universe_mode):
    t0 = time.time()
    log(f"=== {label}: prepare_data (force_universe_mode={force_universe_mode}) ===")
    data = bt.prepare_data(cfg, warmup_start, end, force_universe_mode=force_universe_mode)
    log(
        f"{label}: universe_mode={data.universe_mode} · SURVIVORSHIP_BIAS={data.survivorship_bias} · "
        f"종목 {len(data.indicator_map)}개 (실패 {len(data.failed_tickers)}) · {time.time() - t0:.0f}초"
    )
    return data, time.time() - t0


def _run_stage(
    label: str, data, prepare_elapsed: float, cfg: dict, start: date, end: date, *, stop_next_day_open, apply_slippage: bool = False
) -> dict:
    t0 = time.time()
    log(f"{label}: simulate_portfolio (stop_next_day_open={stop_next_day_open}, apply_slippage={apply_slippage}, max_slots={MAX_SLOTS_B05}) ...")
    result = bt.simulate_portfolio(
        data, cfg, start, end, apply_costs=True, apply_tax=True, max_slots=MAX_SLOTS_B05,
        stop_next_day_open=stop_next_day_open, apply_slippage=apply_slippage,
    )
    liquidated = bt.compute_liquidated_cagr(result, data, cfg, start, end)
    equity_metrics = bt.compute_equity_metrics(result.equity_rows)
    cagr_a = liquidated.get("cagr_liquidated_pct")
    sim_elapsed = time.time() - t0
    log(f"{label}: 세후(a) CAGR = {cagr_a}% (미청산 CAGR {equity_metrics.get('cagr_pct')}%, MDD {equity_metrics.get('mdd_pct')}%) · {sim_elapsed:.0f}초")

    trades = [t for t in result.trades if t["side"] == "진입"]
    return {
        "label": label,
        "universe_mode": data.universe_mode,
        "survivorship_bias": data.survivorship_bias,
        "failed_ticker_count": len(data.failed_tickers),
        "stop_next_day_open": stop_next_day_open,
        "apply_slippage": apply_slippage,
        "cagr_posttax_a_pct": cagr_a,
        "cagr_unrealized_pct": equity_metrics.get("cagr_pct"),
        "mdd_pct": equity_metrics.get("mdd_pct"),
        "sharpe": equity_metrics.get("sharpe"),
        "entry_count": len(trades),
        "elapsed_sec": round(prepare_elapsed + sim_elapsed),
        "status": INCONCLUSIVE_NOTE,
    }


def main() -> None:
    cfg = bt.load_config()
    if cfg["account"]["total_krw"] != cfg["backtest"]["total_krw"]:
        raise SystemExit("cfg[\"account\"][\"total_krw\"] != cfg[\"backtest\"][\"total_krw\"] — KJB-1.1 가드에 걸립니다.")

    warmup_start = bt._parse_date(cfg["backtest"]["warmup_start"])
    train_end = bt._parse_date(cfg["backtest"]["train_end"])  # 2021-12-31, 봉인 이전
    start = date(2016, 1, 1)
    tolerance = cfg["fund_sim"]["b05_reproduction_tolerance_pct"]

    log(f"=== B0.5 재현 테스트 시작 (2016-01-01 ~ {train_end.isoformat()}, 기존 기록 {KNOWN_B05_POSTTAX_A_PCT}%) ===")

    data_current, elapsed_current = _prepare(
        "1단계(현재 구성 종목, 생존 편향)", cfg, warmup_start, train_end, force_universe_mode="CURRENT_CONSTITUENTS"
    )
    stage1 = _run_stage("1단계(현재 구성 종목, 생존 편향)", data_current, elapsed_current, cfg, start, train_end, stop_next_day_open=False)

    # 2·3단계는 force_universe_mode가 같아(POINT_IN_TIME, 엔진 기본값) prepare_data를 한 번만 한다
    # — 체결 타이밍·슬리피지는 simulate_portfolio 단계에서만 갈린다.
    data_pit, elapsed_pit = _prepare(
        "2·3단계(시점별 구성 종목, 엔진 기본값)", cfg, warmup_start, train_end, force_universe_mode=None
    )
    stage2 = _run_stage("2단계(시점별 구성 종목, 엔진 기본값)", data_pit, elapsed_pit, cfg, start, train_end, stop_next_day_open=False)
    stage3_ref = _run_stage(
        "3-참고단계(시점별 구성 종목 + 손절만 다음날 시가, 슬리피지 없음 — 예전 3단계)",
        data_pit, 0.0, cfg, start, train_end, stop_next_day_open=True, apply_slippage=False,
    )
    stage3 = _run_stage(
        "3단계(시점별 구성 종목 + 매도 전부 다음날 시가 + 슬리피지 0.05%, 설계 5.2)",
        data_pit, 0.0, cfg, start, train_end, stop_next_day_open=True, apply_slippage=True,
    )

    stage1_diff = round(stage1["cagr_posttax_a_pct"] - KNOWN_B05_POSTTAX_A_PCT, 2) if stage1["cagr_posttax_a_pct"] is not None else None
    stage2_diff = round(stage2["cagr_posttax_a_pct"] - KNOWN_B05_POSTTAX_A_PCT, 2) if stage2["cagr_posttax_a_pct"] is not None else None
    execution_assumption_effect_pct = (
        round(stage3["cagr_posttax_a_pct"] - stage2["cagr_posttax_a_pct"], 2)
        if stage3["cagr_posttax_a_pct"] is not None and stage2["cagr_posttax_a_pct"] is not None
        else None
    )
    stop_timing_only_effect_pct = (
        round(stage3_ref["cagr_posttax_a_pct"] - stage2["cagr_posttax_a_pct"], 2)
        if stage3_ref["cagr_posttax_a_pct"] is not None and stage2["cagr_posttax_a_pct"] is not None
        else None
    )
    slippage_only_effect_pct = (
        round(stage3["cagr_posttax_a_pct"] - stage3_ref["cagr_posttax_a_pct"], 2)
        if stage3["cagr_posttax_a_pct"] is not None and stage3_ref["cagr_posttax_a_pct"] is not None
        else None
    )

    closest_stage = pick_closest_stage(stage1_diff, stage2_diff)
    closest_diff = stage1_diff if closest_stage == "1단계" else stage2_diff
    engine_matches = closest_diff is not None and abs(closest_diff) <= tolerance

    dataqc_summary = None
    if DATAQC_REPORT_PATH.exists():
        dqc = json.loads(DATAQC_REPORT_PATH.read_text(encoding="utf-8"))
        dataqc_summary = {
            "overall_coverage_pct": dqc.get("overall", {}).get("coverage_pct"),
            "overall_judgement": dqc.get("overall", {}).get("judgement"),
            "min_coverage_pct_threshold": dqc.get("min_coverage_pct_threshold"),
            "source": str(DATAQC_REPORT_PATH.relative_to(ROOT)),
        }

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "period": {"start": start.isoformat(), "end": train_end.isoformat()},
        "known_b05_posttax_a_pct": KNOWN_B05_POSTTAX_A_PCT,
        "tolerance_pct": tolerance,
        "status": INCONCLUSIVE_NOTE,
        "data_quality": dataqc_summary
        or "outputs/backtest/fund_dataqc/report.json 없음 — scripts/fund_dataqc_report.py를 먼저 돌리세요.",
        "stage1_current_constituents": stage1,
        "stage2_point_in_time": stage2,
        "stage3_ref_stop_only_no_slippage": stage3_ref,
        "stage3_point_in_time_full_execution_model": stage3,
        "engine_reproduction": {
            "closest_stage": closest_stage,
            "diff_pct": closest_diff,
            "within_tolerance": engine_matches,
            "note": (
                "설계 문서 5.8은 '1단계=현재 구성 종목'이 기존 기록과 일치할 것으로 가정했지만, "
                "prepare_data가 기본으로 시점별 구성 종목을 쓰므로 실제로는 어느 단계가 더 "
                "가까운지 숫자로 확인해야 한다(위 closest_stage). 이 재현 오차는 엔진 코드가 "
                "예전 기록을 그대로 재현하는지 보는 코드 정확성 체크로, 데이터 품질 판정과는 별개다."
            ),
        },
        "survivorship_bias_pct": "측정 불가 — 2단계(시점별 구성종목)에도 편출 종목이 12~32% 빠져 있어 1·2단계 모두 불완전한 유니버스다. 근거: outputs/backtest/fund_dataqc/report.md",
        "execution_assumption_effect_pct": execution_assumption_effect_pct,
        "execution_assumption_effect_breakdown": {
            "stop_timing_only_pct": stop_timing_only_effect_pct,
            "slippage_only_pct": slippage_only_effect_pct,
            "note": "stop_timing_only = 3-참고단계 - 2단계(손절 타이밍만 변경). slippage_only = 3단계 - 3-참고단계(그 위에 슬리피지만 추가).",
        },
    }

    RESULT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"=== 완료. 결과: {RESULT_PATH} ===")
    log(f"1단계(현재 구성 종목) 세후(a) = {stage1['cagr_posttax_a_pct']}% (기존 대비 {stage1_diff:+}%p)" if stage1_diff is not None else "1단계 계산 불가")
    log(f"2단계(시점별 구성 종목) 세후(a) = {stage2['cagr_posttax_a_pct']}% (기존 대비 {stage2_diff:+}%p)" if stage2_diff is not None else "2단계 계산 불가")
    log(f"3-참고단계(손절만 익일시가, 슬리피지 없음) 세후(a) = {stage3_ref['cagr_posttax_a_pct']}%")
    log(f"3단계(매도 전부 익일시가 + 슬리피지 0.05%) 세후(a) = {stage3['cagr_posttax_a_pct']}%")
    log(f"엔진 재현: {closest_stage}이 기존 기록에 가장 가까움(오차 {closest_diff}%p, 허용 {tolerance}%p 이내: {engine_matches})")
    log("생존 편향 크기: 측정 불가(데이터 품질 미달 — outputs/backtest/fund_dataqc/report.md 참고)")
    log(f"체결 가정 영향(3단계-2단계) = {execution_assumption_effect_pct}%p (손절 타이밍만 {stop_timing_only_effect_pct}%p + 슬리피지만 {slippage_only_effect_pct}%p)")
    if dataqc_summary:
        log(f"데이터 품질: 전체 확보율 {dataqc_summary['overall_coverage_pct']}% (기준 {dataqc_summary['min_coverage_pct_threshold']}%) -> {dataqc_summary['overall_judgement']}")
    log(f"⚠️ {INCONCLUSIVE_NOTE}")

    run_at = datetime.now().isoformat(timespec="seconds")
    conn = trials_store.connect()
    try:
        for stage in (stage1, stage2, stage3_ref, stage3):
            trials_store.record_trial(
                conn, run_at=run_at, label=f"B0.5 재현 {stage['label']}", metrics=stage,
                date_range_start=start.isoformat(), date_range_end=train_end.isoformat(),
                universe_mode=stage["universe_mode"], notes=f"docs/design/fund_sim.md 5.8 B0.5 재현 테스트 — {INCONCLUSIVE_NOTE}",
            )
    finally:
        conn.close()
    log("trials.db에 4건 기록 완료")


if __name__ == "__main__":
    main()
