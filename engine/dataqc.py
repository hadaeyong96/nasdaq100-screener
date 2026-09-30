"""데이터 품질 검사 (AI 펀드 F2, docs/design/fund_sim.md 5.1).

engine.backtest.prepare_data가 이미 받아 온 데이터(시점별 구성 종목 체크포인트 +
종목별 지표 DataFrame)를 검사만 한다 — 네트워크·파일 접근 없이 전부 순수 함수다.
매매 신호·수량 계산에는 전혀 쓰지 않는다: 판정은 오직 구간 전체가 INCONCLUSIVE
(설계 5.7)인지 아닌지를 정하는 데만 쓴다.

검사 3가지:
1. 날짜별(체크포인트) 구성 종목 수가 정상 범위(config.yaml dataqc.expected_constituent_count)인지.
2. 시점별 유니버스 기준 (종목, 거래일) 중 실제로 가격을 받은 비율(coverage) —
   dataqc.min_coverage_pct 미만이면 INCONCLUSIVE.
3. 하루 등락률이 dataqc.extreme_daily_move_pct를 넘는 이상치(분할 누락 의심 포함,
   두 원인을 프로그램으로 구분할 근거가 없어 하나로 표시한다).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from data.universe_history import universe_on


@dataclass(frozen=True)
class ConstituentCountIssue:
    """구성 종목 수가 정상 범위를 벗어난 체크포인트 하나."""

    as_of: str
    count: int


@dataclass(frozen=True)
class ExtremeMoveIssue:
    """하루 등락률이 기준을 넘은 (종목, 날짜) 하나 — 이상치 또는 분할 누락 의심."""

    ticker: str
    date: str
    pct_change: float


@dataclass
class DataQualityReport:
    """build_report의 결과. inconclusive=True면 이 구간의 판정은 설계 5.7에 따라
    PASS/FAIL이 아니라 INCONCLUSIVE로만 낼 수 있다."""

    coverage_pct: float
    ticker_days_expected: int
    ticker_days_priced: int
    constituent_count_issues: list[ConstituentCountIssue] = field(default_factory=list)
    extreme_move_issues: list[ExtremeMoveIssue] = field(default_factory=list)
    inconclusive: bool = False
    reasons: list[str] = field(default_factory=list)


def check_constituent_counts(
    checkpoints: list[tuple[date, frozenset[str]]], min_count: int, max_count: int
) -> list[ConstituentCountIssue]:
    """체크포인트마다 구성 종목 수가 [min_count, max_count] 밖이면 기록한다 (순수 함수).

    입력: checkpoints(data.universe_history.membership_checkpoints 결과), min_count, max_count
    출력: 범위를 벗어난 체크포인트 목록 (날짜 오름차순, checkpoints 순서 그대로)
    """
    issues = []
    for as_of, members in checkpoints:
        n = len(members)
        if not (min_count <= n <= max_count):
            issues.append(ConstituentCountIssue(as_of=as_of.isoformat(), count=n))
    return issues


def compute_price_coverage(
    checkpoints: list[tuple[date, frozenset[str]]], trading_days: list[date], indicator_map: dict[str, pd.DataFrame]
) -> tuple[int, int]:
    """시점별 유니버스 기준 (종목, 거래일) 중 실제로 가격(종가)을 받은 셀 수를 센다 (순수 함수).

    입력: checkpoints, trading_days(검사할 거래일 목록), indicator_map({ticker: df})
    출력: (기대 셀 수, 실제로 종가가 있는 셀 수) — coverage_pct = priced / expected × 100
    """
    expected = 0
    priced = 0
    for d in trading_days:
        members = universe_on(checkpoints, d) if checkpoints else frozenset(indicator_map.keys())
        ts = pd.Timestamp(d)
        for ticker in members:
            expected += 1
            df = indicator_map.get(ticker)
            if df is not None and ts in df.index:
                close = df.loc[ts, "close"]
                if not pd.isna(close):
                    priced += 1
    return expected, priced


def detect_extreme_daily_moves(indicator_map: dict[str, pd.DataFrame], threshold_pct: float) -> list[ExtremeMoveIssue]:
    """종목별 종가의 하루 등락률이 ±threshold_pct를 넘는 날을 찾는다 (순수 함수).

    분할이 반영 안 된 채 들어온 가격은 분할 배수만큼 하루 만에 뛰거나 꺼진 것처럼
    보이므로, 이 검사 하나로 "진짜 급등락"과 "분할 누락"을 함께 잡는다(프로그램만으로
    둘을 구분할 근거가 없어 하나로 표시 — 설계 5.1).

    입력: indicator_map({ticker: df, "close" 열 필요}), threshold_pct(예: 50)
    출력: 기준을 넘은 (종목, 날짜) 목록, 티커 -> 날짜 순
    """
    issues = []
    for ticker in sorted(indicator_map):
        df = indicator_map[ticker]
        if "close" not in df.columns or len(df) < 2:
            continue
        pct_change = df["close"].pct_change() * 100
        for ts, pct in pct_change.items():
            if pd.isna(pct):
                continue
            if abs(pct) >= threshold_pct:
                issues.append(ExtremeMoveIssue(ticker=ticker, date=pd.Timestamp(ts).date().isoformat(), pct_change=round(float(pct), 2)))
    return issues


def build_report(
    checkpoints: list[tuple[date, frozenset[str]]],
    trading_days: list[date],
    indicator_map: dict[str, pd.DataFrame],
    cfg: dict,
) -> DataQualityReport:
    """세 검사를 모두 돌려 DataQualityReport를 만든다 (순수 함수).

    입력: checkpoints, trading_days(검사 구간 거래일 목록), indicator_map, cfg(config.yaml —
         cfg["dataqc"] 사용)
    출력: DataQualityReport. coverage_pct < dataqc.min_coverage_pct면 inconclusive=True.
    """
    dqc = cfg["dataqc"]
    count_range = dqc["expected_constituent_count"]
    count_issues = check_constituent_counts(checkpoints, count_range["min"], count_range["max"])
    expected, priced = compute_price_coverage(checkpoints, trading_days, indicator_map)
    coverage_pct = round(priced / expected * 100, 2) if expected else 0.0
    extreme_issues = detect_extreme_daily_moves(indicator_map, dqc["extreme_daily_move_pct"])

    reasons: list[str] = []
    inconclusive = False
    if coverage_pct < dqc["min_coverage_pct"]:
        inconclusive = True
        reasons.append(f"구성종목 가격 확보율 {coverage_pct}% < 기준 {dqc['min_coverage_pct']}%")

    return DataQualityReport(
        coverage_pct=coverage_pct,
        ticker_days_expected=expected,
        ticker_days_priced=priced,
        constituent_count_issues=count_issues,
        extreme_move_issues=extreme_issues,
        inconclusive=inconclusive,
        reasons=reasons,
    )
