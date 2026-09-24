"""HTML 보고서 생성 (P3, 3번).

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

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
FILLS_XLSX_PATH = Path(__file__).resolve().parents[1] / "data" / "fills.xlsx"

_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=jinja2.select_autoescape(["html"]),
)


def _check(value) -> str:
    return "✔" if value else "✘"


def _num(value, digits: int = 2):
    if value is None:
        return None
    return round(float(value), digits)


def _hold_totals(hold_rows: list[dict]) -> dict | None:
    """"내 보유 종목" 표의 합계 줄에 쓸 값 (종가·평균단가가 있는 행만 — QQQM 대기자금 줄 등은 뺀다).

    입력: hold_rows(engine/daily.py의 보유 현황 행 목록)
    출력: {count, pnl_pct, value} 또는 계산할 행이 없으면 None
    """
    priced = [r for r in hold_rows if r.get("평가금액") is not None and r.get("평균단가") is not None]
    if not priced:
        return None
    total_value = sum(r["평가금액"] for r in priced)
    total_cost = sum(r["평균단가"] * r["수량"] for r in priced)
    pnl_pct = round((total_value - total_cost) / total_cost * 100, 1) if total_cost else None
    return {"count": len(priced), "pnl_pct": pnl_pct, "value": round(total_value, 2)}


def _stage_flags(stage: str) -> list[bool]:
    """보유 현황 "진행" 칸(1차·2차·3차)에 쓸 on/off 3칸."""
    order = {"정찰": 1, "확인": 2, "확정": 3, "추세보유": 0, "청산중": 3}
    n = order.get(stage, 0)
    return [i < n for i in range(3)]


_BUY_HEADERS = {
    "b1": ["종목", "RSI 30 돌파", "MACD", "구름", "거래량", "점수", "지정가", "수량", "투입금액", "계좌%", "손절가", "손절폭", "손절 기준", "위험금액", "실적일", "결정", "비고"],
    "b2": ["종목", "1차 날짜", "MACD 골든크로스", "RSI", "등급", "점수", "지정가", "수량", "투입금액", "계좌%", "손절가", "손절폭", "손절 기준", "위험금액", "실적일", "결정", "비고"],
    "b3": ["종목", "구름 위", "앞구름 양운", "후행스팬 돌파", "모멘텀", "시가 갭", "지정가", "수량", "투입금액", "계좌%", "손절가", "손절폭", "손절 기준", "위험금액", "실적일", "결정", "비고"],
    "b9": ["종목", "구름 위 · 양운", "MACD 골든크로스", "RSI 50~70", "등급", "점수", "지정가", "수량", "투입금액", "계좌%", "손절가", "손절폭", "손절 기준", "위험금액", "실적일", "결정", "비고"],
}
_BUY_TITLES = {"b1": "1차 정찰 · 11%", "b2": "2차 확인 · 22%", "b3": "3차 확정 · 67%", "b9": "재진입 · 1회"}


def _buy_row_cells(row: dict) -> list[str]:
    """탭별 조건 열 5칸을 문자열 목록으로 만든다 (지정가·손절가 등 나머지는 공통 칸)."""
    stage = row["stage"]
    score_text = str(row.get("score", 0))
    grade_text = row.get("grade") or "-"

    if stage == "A1":
        return [
            f"{_check(True)} {row.get('rsi_prev', '-')} → {row.get('rsi_now', '-')}",
            "2차 대기",
            "아래",
            f"{row['vol_ratio']}배" if row.get("vol_ratio") is not None else "-",
            score_text,
        ]
    if stage == "A2":
        return [
            row.get("a1_date") or "-",
            f"정규화 {row['macd_norm']:+.2f}%" if row.get("macd_norm") is not None else "-",
            f"{row.get('rsi_now', '-')}",
            grade_text,
            score_text,
        ]
    if stage == "A3":
        return [
            _check(row.get("cloud_ok")),
            _check(row.get("future_yang_ok")),
            _check(row.get("chikou_ok")),
            _check(row.get("momentum_ok")),
            f"{row['gap_pct']:+.1f}%" if row.get("gap_pct") is not None else "-",
        ]
    # B (재진입)
    return [
        _check(row.get("trend_ok")),
        f"정규화 {row['macd_norm']:+.2f}%" if row.get("macd_norm") is not None else "-",
        f"{row.get('rsi_now', '-')}",
        grade_text,
        score_text,
    ]


def build_context(summary: dict, cfg: dict) -> dict:
    """summary dict(engine/daily.py) + cfg -> Jinja2 템플릿에 넘길 context."""
    as_of = summary.get("as_of")
    as_of_str = as_of.date().isoformat() if as_of is not None else "알수없음"
    order_date = (as_of + pd.tseries.offsets.BDay(1)).date().isoformat() if as_of is not None else "알수없음"

    equity = cfg["account"]["equity_usd"]
    risk_cfg = cfg["risk"]
    risk_total_pct = risk_cfg["a1_budget_pct"] + risk_cfg["a2_budget_pct"] + risk_cfg["a3_budget_pct"]
    cfg_js = {
        "riskTotal": risk_total_pct / 100,
        "stageShare": {"b1": 1 / 9, "b2": 2 / 9, "b3": 6 / 9},
        "riskB": risk_cfg["b_budget_pct"] / 100,
        "gap": risk_cfg["gap_buffer_pct"] / 100,
        "minRisk": risk_cfg["min_risk_per_share_pct"] / 100,
        "maxAlloc": risk_cfg["max_position_pct"] / 100,
    }

    buy_groups = summary.get("buy_groups", {"b1": [], "b2": [], "b3": [], "b9": []})
    buy_tabs = []
    for key in ("b1", "b2", "b3", "b9"):
        rows = buy_groups.get(key, [])
        buy_tabs.append(
            {
                "key": key,
                "title": _BUY_TITLES[key],
                "count": len(rows),
                "headers": _BUY_HEADERS[key],
                "rows": [
                    {
                        "ticker": r["ticker"],
                        "kr": r["kr"],
                        "cond": _buy_row_cells(r),
                        "score": r.get("score", ""),
                        "limit": _num(r["limit"]),
                        "stop": _num(r["stop"]),
                        "stop_pct": r.get("stop_pct"),
                        "stop_basis": r["stop_basis"],
                        "earnings": r["earnings"],
                        "decision": r["decision"],
                        "note": r["note"],
                    }
                    for r in rows
                ],
            }
        )

    stage_label = {"정찰": "1차 정찰", "확인": "2차 확인", "확정": "3차 확정", "추세보유": "재진입", "청산중": "청산중"}
    hold_rows = [
        {**r, "flags": _stage_flags(r["단계"]), "단계_표시": stage_label.get(r["단계"], r["단계"])}
        for r in summary.get("hold_rows", [])
    ]

    # 체결 기록 안내 (P3.1 보완 2번): 신호 당일엔 "주문 후 체결 기록 필요"만, 미체결 확정은 다음 날에만.
    pending_names = " · ".join(r["종목명"] for r in summary.get("pending_order_rows", []))
    unfilled_names = " · ".join(r["종목명"] for r in summary.get("unfilled_rows", []))

    hold_totals = _hold_totals(hold_rows)
    account_pct = round(hold_totals["value"] / equity * 100, 1) if (hold_totals and equity) else None

    return {
        "stale": bool(summary.get("stale")),
        "expected_date_str": summary.get("expected_date") or "",
        "actual_date_str": summary.get("actual_date") or "",
        "mode_label": summary.get("mode_label", "실전"),
        "as_of_str": as_of_str,
        "order_date_str": order_date,
        "generated_str": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "equity": equity,
        "equity_str": f"{equity:,.0f}",
        "cfg_js": cfg_js,
        "buy_count": summary.get("buy_count", 0),
        "sell_count": len(summary.get("sell_rows", [])),
        "held_count": summary.get("held_tickers_count", 0),
        "max_concurrent": summary.get("max_concurrent", 0),
        "filtered_count": len(summary.get("filtered_rows", [])),
        "warn_count": len(summary.get("warn_rows", [])),
        "watch_count": len(summary.get("watch_rows", [])),
        "buy_tabs": buy_tabs,
        "sell_rows": summary.get("sell_rows", []),
        "hold_rows": hold_rows,
        "hold_totals": hold_totals,
        "account_pct": account_pct,
        "watch_rows": summary.get("watch_rows", []),
        "filtered_rows": summary.get("filtered_rows", []),
        "warn_rows": summary.get("warn_rows", []),
        "data_status_rows": summary.get("data_status_rows", []),
        "pending_names": pending_names,
        "unfilled_names": unfilled_names,
        "fills_path": str(FILLS_XLSX_PATH),
        "ichimoku_shift": cfg["indicators"]["ichimoku_shift"],
        "max_position_pct": risk_cfg["max_position_pct"],
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
