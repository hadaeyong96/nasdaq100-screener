"""HTML 보고서 생성 (P3, 3번 / P3.6).

docs/report_template.html의 구조·디자인·스크립트를 그대로 따르는 Jinja2 템플릿
(notify/templates/report.html.j2)에 오늘 판정 결과를 채워
outputs/report_{모드}_YYYY-MM-DD.html(외부 파일 없이 한 파일)로 저장한다.

입력은 engine/daily.py의 run()이 만든 summary dict + cfg다. 이 모듈은 파일
쓰기만 하고 네트워크·DB에 접근하지 않아 테스트에서 summary를 직접 만들어
호출할 수 있다.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import jinja2
import pandas as pd

from core import macro_status
from notify import macro_explain
from notify.briefing import PUBLIC_DISCLAIMER, report_titles
from core.sizing import format_krw
from core.signals import levels

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=jinja2.select_autoescape(["html"]),
)


def _num(value, digits: int = 2):
    if value is None:
        return None
    return round(float(value), digits)


def _won(value) -> str:
    return format_krw(value)


def _hold_totals(hold_rows: list[dict]) -> dict | None:
    """"보유 현황" 표의 합계 줄에 쓸 값 (종가·평균단가가 있는 행만 — QQQM 대기자금 줄 등은 뺀다).

    입력: hold_rows(engine/daily.py의 보유 현황 행 목록)
    출력: {count, pnl_pct, value, value_krw_str, pnl_krw_str} 또는 계산할 행이 없으면 None
    """
    priced = [r for r in hold_rows if r.get("평가금액") is not None and r.get("평균단가") is not None]
    if not priced:
        return None
    total_value = sum(r["평가금액"] for r in priced)
    total_cost = sum(r["평균단가"] * r["수량"] for r in priced)
    pnl_pct = round((total_value - total_cost) / total_cost * 100, 1) if total_cost else None
    krw_rows = [r for r in priced if r.get("평가금액_krw") is not None]
    value_krw = sum(r["평가금액_krw"] for r in krw_rows) if krw_rows else None
    pnl_krw_rows = [r for r in priced if r.get("평가손익_krw") is not None]
    pnl_krw = sum(r["평가손익_krw"] for r in pnl_krw_rows) if pnl_krw_rows else None
    return {
        "count": len(priced),
        "pnl_pct": pnl_pct,
        "value": round(total_value, 2),
        "value_krw_str": _won(value_krw),
        "pnl_krw_str": _won(pnl_krw),
    }


def _stage_flags(stage: str) -> list[bool]:
    """보유 현황 "진행" 칸(1차·2차·3차)에 쓸 on/off 3칸."""
    order = {"정찰": 1, "확인": 2, "확정": 3, "추세보유": 0, "청산중": 3}
    n = order.get(stage, 0)
    return [i < n for i in range(3)]


# 11칸(차수별) / 12칸("전체" — 티커 다음에 차수 칸, P3.6 6-3번).
_BUY_HEADERS = ["종목", "티커", "결정", "지정가", "수량", "투입금액", "손절가", "손절폭", "최대손실", "점수", "비고", "설명"]
_ALL_BUY_HEADERS = ["종목", "티커", "차수", "결정", "지정가", "수량", "투입금액", "손절가", "손절폭", "점수", "비고", "설명"]
_BUY_TITLES = {
    "b1": "1차 정찰 · 슬롯의 1/9",
    "b2": "2차 확인 · 2/9",
    "b3": "3차 확정 · 6/9",
    "b9": "재진입 · 슬롯 전체",
}
# 실전(live)은 종목별 계획금액으로 수량을 정한다 — 탭 이름도 "계획금액" 기준으로 쓴다.
_BUY_TITLES_LIVE = {
    "b1": "1차 정찰 · 계획금액의 1/9",
    "b2": "2차 확인 · 2/9",
    "b3": "3차 확정 · 6/9",
    "b9": "재진입 · 계획금액 전체",
}
_NUM_HEADERS = {"지정가", "수량", "투입금액", "손절가", "손절폭", "최대손실", "점수"}

# 라이브 어드바이저 1단계 (docs/design/live_advisor.md 3·7번) — 판정 배지 색상.
_JUDGMENT_BADGE_CLASS = {"보유": "b-info", "추가매수": "b-buy", "일부매도 검토": "b-warn", "매도": "b-sell"}


def _buy_row_ctx(r: dict, include_stage_label: bool) -> dict:
    ctx = {
        "ticker": r["ticker"],
        "kr": r["kr"],
        "score": r.get("score", ""),
        "limit": _num(r["limit"]),
        "stop": _num(r["stop"]),
        "stop_pct": r.get("stop_pct"),
        "decision": r["decision"],
        "note": r["note"],
        "qty": r.get("qty", 0),
        "amount_krw": r.get("amount_krw") or 0,
        "amount_krw_str": _won(r.get("amount_krw")),
        "max_loss_krw_str": _won(r.get("max_loss_krw")),
        "key": r.get("key", f"{r['ticker']}-{r.get('stage', '')}"),
        "stage": r.get("stage", ""),
        "is_new_position": bool(r.get("is_new_position")),
        "explain": r.get("explain"),
    }
    if include_stage_label:
        ctx["stage_label"] = r.get("stage_label", "")
    return ctx


def _macro_row_ctx(row: dict) -> dict:
    """engine.daily._build_macro_rows의 행 하나 -> 템플릿에 넘길 context (P3.8, 표시 전용).

    판정·3칸 눈금(★)은 row["badge"](core.macro_status.classify_macro 결과)를 그대로 쓰고,
    설명 섹션(#explain-{slug})의 ①~③은 notify.macro_explain, ④는 오늘 값으로 만든다.
    """
    series = row.get("series") or []
    spark_points = macro_status.sparkline_points(series) if len(series) > 1 else ""
    ref_ys: list[float] = []
    if series:
        mn, mx = min(series), max(series)
        # 기준선은 1년 범위 안에 있을 때만 그린다(범위 밖이면 그래프 끝에 붙어 오해를 준다).
        ref_ys = [round(macro_status.value_to_y(rv, mn, mx), 1) for rv in row.get("ref_values", []) if mn <= rv <= mx]
    value = row["value"]
    slug = row.get("slug") or "etc"
    if slug == "fx":
        value_str = f"{value:,.0f}"
    elif slug == "t10y2y":
        value_str = f"{value:+.2f}"
    elif isinstance(value, float) and not value.is_integer():
        value_str = f"{value:,.2f}"
    else:
        value_str = f"{value:,.0f}"
    change = row.get("change_1w")
    change_str = f"{change:+g}" if change is not None else None
    range_str = f"{min(series):,.2f} ~ {max(series):,.2f}" if series else "-"
    badge = row["badge"]
    explain = macro_explain.EXPLAIN.get(slug, {"title": row["name"], "what": "", "why": "", "example": ""})
    return {
        **row, "slug": slug, "spark_points": spark_points, "ref_ys": ref_ys,
        "value_str": value_str, "change_str": change_str, "range_str": range_str,
        "scale": badge.get("scale", []),
        "explain": {**explain, "now": macro_explain.now_text(slug, value_str, row.get("unit", ""), badge, row.get("note"))},
    }


def _live_judgment_row_ctx(r: dict) -> dict:
    """engine.daily.compute_live_judgments의 행 하나 -> 템플릿에 넘길 context
    (docs/design/live_advisor.md 3·7번 — 판정 배지, 원화 표시, 1주 미만 경고 문구)."""
    warning = r.get("one_share_warning")
    warning_str = None
    if warning:
        warning_str = (
            f"{r['name_kr']}({r['ticker']}): 계획금액 {_won(r.get('plan_budget_krw'))}으로 "
            f"1차 매수 0주 (최소 {_won(warning['min_budget_krw'])} 필요)"
        )
    return {
        **r,
        "badge_class": _JUDGMENT_BADGE_CLASS.get(r["judgment"], "b-info"),
        "plan_budget_krw_str": _won(r.get("plan_budget_krw")),
        "tranche_krw_str": _won(r.get("tranche_krw")),
        "one_share_warning_str": warning_str,
    }


# ── 오늘의 스크리닝 (보고서 맨 위, docs/design/screening_view.md) ─────────────────
# 전략 카드 문구는 여기 상수로 두고, 기한·임계값 숫자는 모두 cfg(assumptions·strategy_levels·risk)에서 읽는다.
_STRATEGY_NAME = "바닥 반전 3단 확인 매수법 (1:2:6 피라미딩)"
_STRATEGY_LINE = (
    "많이 떨어진 종목이 RSI → MACD → 일목 구름 순서로 반등을 증명할 때마다 1 → 2 → 6으로 늘려 산다. "
    "떨어질 때 더 사는 물타기는 하지 않는다."
)
_STRATEGY_ROOT = (
    "뿌리: 후지모토 1:2:6(대본 MACD_11) + \"맞을 때만 더 산다\"(리버모어식 피라미딩). 유명 전략을 그대로 옮긴 것이 아니라, "
    "MACD·RSI·일목 대본을 모아 우리가 만든 전략이다(전략 v3)."
)
_CANSLIM_ROWS = [
    ("사는 자리", "신고가 근처 바닥 패턴(손잡이 컵 등) 돌파", "많이 떨어진 뒤 반등이 확인되는 자리"),
    ("종목 고르기", "실적·성장(펀더멘털) + 차트", "차트 지표만 (RSI·MACD·일목·거래량)"),
    ("비중 늘리기", "오를 때 추가 매수(피라미딩)", "신호가 겹칠 때만 1 → 2 → 6"),
    ("손절", "매수가 −7~8%", "{swing}일 최저가 (차트 기준, 3차부터는 구름 하단과 비교)"),
]
_LANE_COLOR = {"a1": "a1", "a2": "a2", "a3": "a3", "b": "b"}


def _strategy_card(cfg: dict) -> dict:
    """"우리 전략 한눈에" 고정 카드. 숫자는 cfg에서 읽는다."""
    a, risk = cfg["assumptions"], cfg["risk"]
    lv = levels(cfg)
    lo, mid, hi = (f"{lv[k]:g}" for k in ("rsi_oversold", "rsi_mid", "rsi_overbought"))
    exp, swing, gap = a["a1_to_a2_expiry_days"], a["swing_low_period"], a["gap_filter_pct"]
    total_risk = round(risk["a1_budget_pct"] + risk["a2_budget_pct"] + risk["a3_budget_pct"], 2)
    stages = [
        {"cls": "a1", "name": "1차 정찰", "weight": "1/9", "buy": f"RSI가 어제 {lo} 미만 → 오늘 {lo} 이상 (보유 중·재진입 대기 아님)",
         "mean": "떨어지던 힘이 멈춤 (첫 신호)", "stop": f"{swing}일 최저가", "sell": "MACD 데드크로스 → 1차분"},
        {"cls": "a2", "name": "2차 확인", "weight": "2/9",
         "buy": f"1차 체결 다음 거래일부터, 1차일 포함 {exp}거래일 안에 MACD 골든크로스 · RSI {lo}~{hi}",
         "mean": "방향이 진짜로 바뀜", "stop": f"{swing}일 최저가 (1차 기준)", "sell": f"RSI {mid} 이탈 → 2차분"},
        {"cls": "a3", "name": "3차 확정", "weight": "6/9",
         "buy": f"2차 보유 중 · 종가 > 구름 상단 · 앞구름 양운 · 종가 > {cfg['indicators']['ichimoku_shift']}일 전 고가 · MACD > 시그널 & RSI ≥ {mid}",
         "mean": "상승 추세가 자리 잡음 (본 진입)", "stop": f"{swing}일 최저가·구름 하단 중 높은 값", "sell": "구름 이탈 → 남은 전량"},
        {"cls": "b", "name": "재진입", "weight": "한 번에",
         "buy": f"이미 구름 위(양운·후행스팬 돌파) · 골든크로스 · 정규화 MACD {a['s_grade_macd_norm_min_pct']:g}% 이상 · RSI {mid}~{hi} (쿨다운 없음)",
         "mean": "쉬던 상승 추세가 다시 달림", "stop": f"진입일 {swing}일 최저가", "sell": "1·2·3차와 같은 순서"},
    ]
    bans = (
        f"공통 금지(신규 매수만): 실적 발표 {lv['earnings_filter_trading_days']}거래일 이내(모든 차수) · 골든크로스 날 RSI {hi} 이상(2차·재진입) · "
        f"최근 {lv['cross_window_days']}거래일 MACD 교차 {a['whipsaw_max_crosses_20d']}회 이상(2차·재진입) · 구름 안·앞구름 음운(3차·재진입) · "
        f"신호일 시가 갭 {gap:g}% 이상(3차). 매수일 시가가 {gap:g}% 이상 높게 시작하면 직접 보류(3차, 수동 규칙)."
    )
    risk_line = (
        f"한 종목 최대 손실: 계좌의 {total_risk:g}% (1:2:6 단계 합) · 재진입은 {risk['b_budget_pct']:g}% · "
        f"매수는 다음 거래일 종가 × {cfg['entry']['limit_markup']:g} 이하 지정가 · 청산 후 {a['reentry_cooldown_days']}거래일은 새 1차 없음."
    )
    canslim = [(k, o, ours.format(swing=swing)) for k, o, ours in _CANSLIM_ROWS]
    return {"name": _STRATEGY_NAME, "line": _STRATEGY_LINE, "root": _STRATEGY_ROOT, "stages": stages,
            "bans": bans, "risk_line": risk_line, "canslim": canslim}


def _screening_ctx(summary: dict, cfg: dict) -> dict:
    """summary["screening"](core.screening_view.build_screening_view) -> 템플릿 context.

    단계 상자의 막대 폭은 그 레인 첫 단계 대비 비율(%)이다.
    """
    view = summary.get("screening")
    ctx = {"strategy": _strategy_card(cfg), "lanes": [], "check": None, "scan_count": None, "as_of_str": None}
    if not view:
        return ctx
    for key in ("a1", "a2", "a3", "b"):
        lane = view["lanes"][key]
        first = lane["steps"][0]["count"] or 0
        steps = [
            {**st, "pct": (100.0 if i == 0 else (st["count"] / first * 100 if first else 0.0))}
            for i, st in enumerate(lane["steps"])
        ]
        ctx["lanes"].append({**lane, "steps": steps, "color": _LANE_COLOR[key]})
    ctx["check"] = view["funnel_check"]
    ctx["scan_count"] = view["scan_count"]
    ctx["as_of_str"] = view.get("as_of_str")
    return ctx


def build_context(summary: dict, cfg: dict) -> dict:
    """summary dict(engine/daily.py) + cfg -> Jinja2 템플릿에 넘길 context."""
    as_of = summary.get("as_of")
    as_of_str = as_of.date().isoformat() if as_of is not None else "알수없음"
    order_date = (as_of + pd.tseries.offsets.BDay(1)).date().isoformat() if as_of is not None else "알수없음"

    plan_cfg = cfg.get("plan", {})
    total_krw = cfg["account"]["total_krw"]
    risk_cfg = cfg["risk"]
    cfg_js = {
        "maxSlots": plan_cfg.get("max_slots", 8),
        "stageFrac": {"A1": 1 / 9, "A2": 2 / 9, "A3": 6 / 9, "B": 1.0},
        "stageRiskPct": {
            "A1": risk_cfg["a1_budget_pct"],
            "A2": risk_cfg["a2_budget_pct"],
            "A3": risk_cfg["a3_budget_pct"],
            "B": risk_cfg["b_budget_pct"],
        },
        "gap": risk_cfg["gap_buffer_pct"] / 100,
        "minRisk": risk_cfg["min_risk_per_share_pct"] / 100,
    }

    buy_groups = summary.get("buy_groups", {"b1": [], "b2": [], "b3": [], "b9": []})
    all_rows = sorted(
        (r for key in ("b1", "b2", "b3", "b9") for r in buy_groups.get(key, [])),
        key=lambda r: r["score"],
        reverse=True,
    )
    buy_tabs = [
        {
            "key": "all",
            "title": "전체",
            "count": len(all_rows),
            "headers": _ALL_BUY_HEADERS,
            "rows": [_buy_row_ctx(r, include_stage_label=True) for r in all_rows],
        }
    ]
    for key in ("b1", "b2", "b3", "b9"):
        rows = buy_groups.get(key, [])
        buy_tabs.append(
            {
                "key": key,
                "title": (_BUY_TITLES_LIVE if summary.get("mode") == "live" else _BUY_TITLES)[key],
                "count": len(rows),
                "headers": _BUY_HEADERS,
                "rows": [_buy_row_ctx(r, include_stage_label=False) for r in rows],
            }
        )
    buy_count = summary.get("buy_count", len(all_rows))

    stage_label = {"정찰": "1차 정찰", "확인": "2차 확인", "확정": "3차 확정", "추세보유": "재진입", "청산중": "청산중"}
    hold_rows = [
        {
            **r,
            "flags": _stage_flags(r["단계"]),
            "단계_표시": stage_label.get(r["단계"], r["단계"]),
            "평가금액_krw_str": _won(r.get("평가금액_krw")),
            "평가손익_krw_str": _won(r.get("평가손익_krw")),
        }
        for r in summary.get("hold_rows", [])
    ]
    sell_rows = [
        {**r, "예상손익_krw_str": _won(r.get("예상손익_krw"))}
        for r in summary.get("sell_rows", [])
    ]
    # hold_rows·watch_rows·filtered_rows·warn_rows는 summary의 dict를 그대로 쓰므로
    # engine/daily.py가 이미 채운 row["explain"]이 별다른 변환 없이 그대로 넘어간다.

    # 체결 기록 안내 (P3.1 보완 2번): 신호 당일엔 "주문 후 체결 기록 필요"만, 미체결 확정은 다음 날에만.
    pending_names = " · ".join(r["종목명"] for r in summary.get("pending_order_rows", []))
    unfilled_names = " · ".join(r["종목명"] for r in summary.get("unfilled_rows", []))

    hold_totals = _hold_totals(hold_rows)

    funding = summary.get("funding_plan")
    strategy_limit_krw = funding["strategy_limit_krw"] if funding else 0.0
    if funding and strategy_limit_krw:
        held_pct = min(funding["held_krw"] / strategy_limit_krw * 100, 100)
        reserved_pct = min(funding["reserved_krw"] / strategy_limit_krw * 100, 100 - held_pct)
        new_pct = min(funding["new_krw"] / strategy_limit_krw * 100, max(100 - held_pct - reserved_pct, 0))
        remaining_pct = max(100 - held_pct - reserved_pct - new_pct, 0)
    else:
        held_pct = reserved_pct = new_pct = remaining_pct = 0.0

    buy_risk_sum_krw = summary.get("buy_risk_sum_krw", 0)
    buy_risk_pct = summary.get("buy_risk_pct")

    # "읽는 법" 탭 숫자 기준 (P3.7 — 하드코딩하지 않고 config.yaml에서 읽는다).
    assumptions_cfg = cfg["assumptions"]
    ind_cfg = cfg["indicators"]
    guide = {
        "a1_to_a2_expiry_days": assumptions_cfg["a1_to_a2_expiry_days"],
        "reentry_cooldown_days": assumptions_cfg["reentry_cooldown_days"],
        "gap_filter_pct": assumptions_cfg["gap_filter_pct"],
        "whipsaw_max_crosses_20d": assumptions_cfg["whipsaw_max_crosses_20d"],
        "swing_low_period": assumptions_cfg["swing_low_period"],
        "s_grade_macd_norm_min_pct": assumptions_cfg["s_grade_macd_norm_min_pct"],
        "b_grade_macd_norm_max_pct": assumptions_cfg["b_grade_macd_norm_max_pct"],
        "risk_a_pct": round(risk_cfg["a1_budget_pct"] + risk_cfg["a2_budget_pct"] + risk_cfg["a3_budget_pct"], 2),
        "risk_b_pct": risk_cfg["b_budget_pct"],
        "rsi_period": ind_cfg["rsi"]["period"],
        "macd_fast": ind_cfg["macd"]["fast"],
        "macd_slow": ind_cfg["macd"]["slow"],
        "macd_signal": ind_cfg["macd"]["signal"],
        "ichimoku_tenkan": ind_cfg["ichimoku"]["tenkan"],
        "ichimoku_kijun": ind_cfg["ichimoku"]["kijun"],
        "ichimoku_senkou_b": ind_cfg["ichimoku"]["senkou_b"],
        "ichimoku_shift": ind_cfg["ichimoku_shift"],
        "entry_limit_markup": cfg["entry"]["limit_markup"],
    }
    # cfg에 macro 섹션이 없어도(구버전 테스트 fixture 등) 안내 탭이 깨지지 않게 기본값을 둔다
    # — 실제 운영은 config.yaml에 항상 이 섹션이 있어 아래 기본값은 쓰이지 않는다.
    macro_cfg = cfg.get("macro", {})
    macro_th = macro_cfg.get("thresholds", {})
    # 읽는 법 탭의 시장 온도 표: 기준표 3칸을 macro_status에서 그대로 받아 쓴다(보고서 칸과 같은 기준).
    guide["macro"] = {
        code: macro_status.scale_ranges(code, macro_th)
        for code in ("FEAR_GREED", "VIXCLS", "DGS10", "T10Y2Y", "BAMLH0A0HYM2", "DFEDTARU", "DEXKOUS")
        if code in macro_th
    }

    is_live = summary.get("mode") == "live"
    live_judgment_rows = [_live_judgment_row_ctx(r) for r in summary.get("live_judgment_rows", [])]
    changed_judgment_rows = [r for r in live_judgment_rows if r.get("changed")]
    one_share_warning_rows = [r for r in live_judgment_rows if r.get("one_share_warning")]
    # 기준가 기반 계획(구글 시트 계획 탭 개편, 2026-09-30): 이미 보유 수량이 계획
    # 총 주수 이상이면 추가 매수 추천이 0으로 잘리고 여기 경고로 뜬다.
    plan_limit_exceeded_rows = [r for r in live_judgment_rows if r.get("plan_limit_exceeded")]

    # fx_rate는 funding_plan과 별개로 summary 최상위에 항상 있다(engine/daily.py) —
    # live는 funding이 없어도(계좌 총액 기반 자금 계획을 안 쓴다) 오늘 환율은 보여줘야 한다.
    fx_rate = summary.get("fx_rate")
    fx_date_str = summary.get("fx_date") or ""
    fx_is_fallback = bool(summary.get("fx_is_fallback"))

    return {
        "stale": bool(summary.get("stale")),
        "expected_date_str": summary.get("expected_date") or "",
        "actual_date_str": summary.get("actual_date") or "",
        "mode_label": summary.get("mode_label", "실전"),
        "is_live": is_live,
        "as_of_str": as_of_str,
        "order_date_str": order_date,
        "generated_str": datetime.now().strftime("%Y-%m-%d %H:%M"),
        # live는 계좌 총액 기반 자금 계획을 아예 안 쓰므로(docs/design/live_advisor.md 0번)
        # 총자금·전략 한도는 paper에서만 의미가 있다 — 템플릿이 is_live로 숨긴다.
        "total_krw": total_krw,
        "total_krw_str": _won(total_krw),
        "strategy_limit_pct": plan_cfg.get("strategy_limit_pct", 60),
        "fx_rate": fx_rate,
        "fx_rate_str": f"{fx_rate:,.0f}" if fx_rate else "-",
        "fx_date_str": fx_date_str,
        "fx_is_fallback": fx_is_fallback,
        "live_judgment_rows": live_judgment_rows,
        "changed_judgment_rows": changed_judgment_rows,
        "one_share_warning_rows": one_share_warning_rows,
        "plan_limit_exceeded_rows": plan_limit_exceeded_rows,
        "cfg_js": cfg_js,
        "titles": report_titles(cfg),
        "buy_count": buy_count,
        "sell_count": len(sell_rows),
        "held_count": summary.get("held_tickers_count", 0),
        "max_concurrent": summary.get("max_concurrent", 0),
        "filtered_count": len(summary.get("filtered_rows", [])),
        "warn_count": len(summary.get("warn_rows", [])),
        "watch_count": len(summary.get("watch_rows", [])),
        "buy_tabs": buy_tabs,
        "buy_risk_sum_str": _won(buy_risk_sum_krw),
        "buy_risk_pct": f"{buy_risk_pct:.2f}" if buy_risk_pct is not None else "0.00",
        "sell_rows": sell_rows,
        "hold_rows": hold_rows,
        "hold_totals": hold_totals,
        "watch_rows": summary.get("watch_rows", []),
        "filtered_rows": summary.get("filtered_rows", []),
        "warn_rows": summary.get("warn_rows", []),
        "data_status_rows": summary.get("data_status_rows", []),
        "pending_names": pending_names,
        "unfilled_names": unfilled_names,
        "ichimoku_shift": cfg["indicators"]["ichimoku_shift"],
        "funding": funding,
        "strategy_limit_str": _won(strategy_limit_krw if funding else None),
        "slot_str": _won(funding["slot_krw"] if funding else None),
        "held_str": _won(funding["held_krw"] if funding else None),
        "reserved_str": _won(funding["reserved_krw"] if funding else None),
        "new_str": _won(funding["new_krw"] if funding else None),
        "remaining_str": _won(funding["remaining_krw"] if funding else None),
        "qqqm_str": _won(funding["qqqm_target_krw"] if funding else None),
        "held_pct": held_pct,
        "reserved_pct": reserved_pct,
        "new_pct": new_pct,
        "remaining_pct": remaining_pct,
        "data_used": funding["held_krw"] if funding else 0,
        "data_reserved": funding["reserved_krw"] if funding else 0,
        "data_slot": funding["slot_krw"] if funding else 0,
        "data_fx": funding["fx_rate"] if funding else 0,
        "guide": guide,
        "macro_rows": [_macro_row_ctx(r) for r in summary.get("macro_rows", [])],
        "macro_as_of_str": max((r["as_of"] for r in summary.get("macro_rows", []) if r.get("as_of")), default=as_of_str),
        "macro_disclaimer": macro_explain.DISCLAIMER,
        "screening": _screening_ctx(summary, cfg),
    }


def render_report(summary: dict, cfg: dict, output_dir: Path) -> Path:
    """보고서 HTML을 렌더링해 outputs/report_{모드}_YYYY-MM-DD.html로 저장하고 경로를 반환한다.

    파일 이름에 모드(live/paper)를 넣어 실전·모의 보고서가 같은 날짜여도 서로 덮어쓰지
    않게 한다(모의를 매일 나란히 돌리기 시작하면서 생긴 요구사항).
    """
    context = build_context(summary, cfg)
    template = _env.get_template("report.html.j2")
    html = template.render(**context)
    output_dir.mkdir(exist_ok=True, parents=True)
    mode = summary.get("mode", "live")
    path = output_dir / f"report_{mode}_{context['as_of_str']}.html"
    path.write_text(html, encoding="utf-8")
    return path


# ── 단체방 공개용 HTML (투자클럽) ─────────────────────────────────────
# 시장 온도 + 오늘의 추천(진입가·손절가·조건)만 담는다. 보유·수량·평단·손익·
# 계획금액·체결 내역은 build_context와 달리 애초에 읽지 않는다 — 새는 값이
# 없다는 걸 코드 구조로도 보장하기 위해서다 (build_context를 재사용하지 않음).
#
# 오늘의 추천은 신규 진입(b1, A1 정찰) 신호만 담는다 — b2·b3·b9는 라이브에서
# 실제 보유했거나 보유 중인 종목에만 나오는 신호라 종목명 자체가 보유를 드러낸다
# (notify.briefing 모듈 위 주석 참고).
_PUBLIC_RECOMMEND_BUCKET = "b1"


def _public_buy_row_ctx(r: dict) -> dict:
    """buy_groups 행 하나 -> 공개용 템플릿 context. r["note"]·수량·금액은 담지 않는다."""
    return {
        "ticker": r["ticker"],
        "kr": r["kr"],
        "stage_label": r.get("stage_label", ""),
        "limit": _num(r["limit"]),
        "stop": _num(r["stop"]),
        "stop_pct": r.get("stop_pct"),
        "decision": r["decision"],
        "condition": r.get("condition_summary") or "",
    }


def build_public_context(summary: dict, cfg: dict) -> dict:
    """summary + cfg -> 단체방 공개용 Jinja2 템플릿 context.

    build_context와 달리 funding_plan·hold_rows·sell_rows·pending_order_rows 등
    보유·자금 관련 값은 아예 만들지 않는다 (P-group 지시문 — 공개용 결과물에 보유·
    금액 정보가 새지 않아야 한다).
    """
    as_of = summary.get("as_of")
    as_of_str = as_of.date().isoformat() if as_of is not None else "알수없음"

    buy_groups = summary.get("buy_groups", {"b1": [], "b2": [], "b3": [], "b9": []})
    new_entry_rows = sorted(
        buy_groups.get(_PUBLIC_RECOMMEND_BUCKET, []), key=lambda r: r.get("score", 0), reverse=True
    )

    return {
        "as_of_str": as_of_str,
        "titles": report_titles(cfg),
        "generated_str": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "buy_rows": [_public_buy_row_ctx(r) for r in new_entry_rows],
        "buy_count": len(new_entry_rows),
        "macro_rows": [_macro_row_ctx(r) for r in summary.get("macro_rows", [])],
        "macro_as_of_str": max((r["as_of"] for r in summary.get("macro_rows", []) if r.get("as_of")), default=as_of_str),
        "macro_disclaimer": macro_explain.DISCLAIMER,
        "public_disclaimer": PUBLIC_DISCLAIMER,
    }


def render_public_report(summary: dict, cfg: dict, output_dir: Path) -> Path:
    """단체방 공개용 보고서 HTML을 outputs/report_public_YYYY-MM-DD.html로 저장하고 경로를 반환한다.

    live 전용(호출부 책임) — paper는 애초에 텔레그램을 보내지 않으므로 부르지 않는다.
    """
    context = build_public_context(summary, cfg)
    template = _env.get_template("report_public.html.j2")
    html = template.render(**context)
    output_dir.mkdir(exist_ok=True, parents=True)
    path = output_dir / f"report_public_{context['as_of_str']}.html"
    path.write_text(html, encoding="utf-8")
    return path
