"""HTML 보고서 생성 (P3, 3번 / P3.6).

docs/report_template.html의 구조·디자인·스크립트를 그대로 따르는 Jinja2 템플릿
(notify/templates/report.html.j2)에 오늘 판정 결과를 채워
outputs/report_YYYY-MM-DD.html(외부 파일 없이 한 파일)로 저장한다.

입력은 engine/daily.py의 run()이 만든 summary dict + cfg다. 이 모듈은 파일
쓰기만 하고 네트워크·DB에 접근하지 않아 테스트에서 summary를 직접 만들어
호출할 수 있다.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import jinja2
import pandas as pd

from core.sizing import format_krw

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
FILLS_XLSX_PATH = Path(__file__).resolve().parents[1] / "data" / "fills.xlsx"

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
_BUY_HEADERS = ["종목", "티커", "결정", "지정가", "수량", "투입금액", "손절가", "손절폭", "최대손실", "점수", "비고"]
_ALL_BUY_HEADERS = ["종목", "티커", "차수", "결정", "지정가", "수량", "투입금액", "손절가", "손절폭", "점수", "비고"]
_BUY_TITLES = {
    "b1": "1차 정찰 · 슬롯의 1/9",
    "b2": "2차 확인 · 2/9",
    "b3": "3차 확정 · 6/9",
    "b9": "재진입 · 슬롯 전체",
}
_NUM_HEADERS = {"지정가", "수량", "투입금액", "손절가", "손절폭", "최대손실", "점수"}


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
    }
    if include_stage_label:
        ctx["stage_label"] = r.get("stage_label", "")
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
                "title": _BUY_TITLES[key],
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

    return {
        "stale": bool(summary.get("stale")),
        "expected_date_str": summary.get("expected_date") or "",
        "actual_date_str": summary.get("actual_date") or "",
        "mode_label": summary.get("mode_label", "실전"),
        "as_of_str": as_of_str,
        "order_date_str": order_date,
        "generated_str": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_krw": total_krw,
        "total_krw_str": _won(total_krw),
        "strategy_limit_pct": plan_cfg.get("strategy_limit_pct", 60),
        "fx_rate": funding["fx_rate"] if funding else None,
        "fx_rate_str": f"{funding['fx_rate']:,.0f}" if funding else "-",
        "fx_date_str": (funding["fx_date"] or "") if funding else "",
        "fx_is_fallback": bool(funding and funding["fx_is_fallback"]),
        "cfg_js": cfg_js,
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
        "fills_path": str(FILLS_XLSX_PATH),
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
    }


def render_report(summary: dict, cfg: dict, output_dir: Path) -> Path:
    """보고서 HTML을 렌더링해 outputs/report_YYYY-MM-DD.html로 저장하고 경로를 반환한다."""
    context = build_context(summary, cfg)
    template = _env.get_template("report.html.j2")
    html = template.render(**context)
    output_dir.mkdir(exist_ok=True, parents=True)
    path = output_dir / f"report_{context['as_of_str']}.html"
    path.write_text(html, encoding="utf-8")
    return path
