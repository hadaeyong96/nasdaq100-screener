"""계획 탭 자동 기록 — 어떤 종목을 추가할지 고르는 순수 함수 (docs/design/auto_plan.md).

네트워크·파일·DB·현재 시각에 접근하지 않는다(core/ 원칙). 실행일(KST)은 호출부가 넘긴다.
시트 쓰기는 data/sheets.py, 흐름 연결은 engine/daily.py 몫이다.
"""

from __future__ import annotations

import re
from datetime import date

import pandas as pd

from core import signals as sig

ENTRY_KINDS = ("A1", "A2", "A3", "B")
AUTO_MEMO_PREFIX = "자동 추가"
EXPIRE_MARK = "만료"


def auto_plan_settings(cfg: dict) -> dict:
    """config.yaml live.auto_plan을 기본값과 합쳐 돌려준다.

    입력: cfg 전체
    출력: {"enabled": bool, "default_budget_krw": float, "max_new_per_day": int}
         (섹션이 없으면 enabled=False — 이 기능이 생기기 전과 같은 동작)
    """
    section = ((cfg.get("live") or {}).get("auto_plan")) or {}
    return {
        "enabled": bool(section.get("enabled", False)),
        "default_budget_krw": float(section.get("default_budget_krw", 3_000_000)),
        "max_new_per_day": int(section.get("max_new_per_day", 5)),
    }


def build_auto_memo(run_date: date, stage_label: str, stop_price: float) -> str:
    """자동 추가 메모 문자열.

    입력: run_date(실행일 KST), stage_label(차수 표시 이름, 예: "1차 정찰"), stop_price(손절가 $)
    출력: "자동 추가 YYYY-MM-DD · {차수 표시 이름} · 손절 $X.XX"
    """
    return f"{AUTO_MEMO_PREFIX} {run_date.isoformat()} · {stage_label} · 손절 ${stop_price:.2f}"


def select_new_plan_rows(
    signals: list[dict], existing_tickers, settings: dict, run_date: date
) -> list[dict]:
    """오늘 진입 신호 중 계획에 없는 종목을 골라 계획 탭에 추가할 줄을 만든다.

    입력: signals([{"ticker","stage"(A1·A2·A3·B),"stage_label","limit"(지정가 $),"stop"(손절가 $ 또는 None),
              "score"}]), existing_tickers(계획 탭에 이미 있는 티커들), settings(auto_plan_settings 결과),
         run_date(실행일 KST)
    출력: [{"ticker","budget_krw","reg_date"(YYYY-MM-DD),"ref_price","memo"}] — 점수 높은 순(같으면
         티커 오름차순), 최대 max_new_per_day개. 손절가 없는 신호는 뺀다(메모를 만들 수 없음).
         한 종목에 신호가 여럿이면 점수가 가장 높은 하나만 쓴다.
    """
    existing = {str(t).strip().upper() for t in existing_tickers if str(t).strip()}
    best: dict[str, dict] = {}
    for s in signals:
        ticker = str(s["ticker"]).strip().upper()
        if s.get("stage") not in ENTRY_KINDS or ticker in existing or s.get("stop") is None:
            continue
        if ticker not in best or (s.get("score") or 0) > (best[ticker].get("score") or 0):
            best[ticker] = {**s, "ticker": ticker}
    ranked = sorted(best.values(), key=lambda s: (-(s.get("score") or 0), s["ticker"]))
    rows = []
    for s in ranked[: max(settings["max_new_per_day"], 0)]:
        rows.append(
            {
                "ticker": s["ticker"],
                "budget_krw": settings["default_budget_krw"],
                "reg_date": run_date.isoformat(),
                "ref_price": round(float(s["limit"]), 2),
                "memo": build_auto_memo(run_date, s["stage_label"], float(s["stop"])),
            }
        )
    return rows


def _auto_memo_pattern(stage_label: str) -> re.Pattern:
    return re.compile(
        rf"^{re.escape(AUTO_MEMO_PREFIX)} (\d{{4}}-\d{{2}}-\d{{2}}) · {re.escape(stage_label)} · 손절 \$\d+\.\d{{2}}$"
    )


def select_expiry_memos(
    plan_rows: list[dict],
    held_qty_by_ticker: dict[str, float],
    trading_dates_by_ticker: dict[str, pd.DatetimeIndex],
    as_of,
    run_date: date,
    expiry_days: int,
    a1_stage_label: str,
) -> list[dict]:
    """만료 메모를 덧붙일 계획 줄을 고른다 (docs/design/auto_plan.md "만료 메모", 기간 기준).

    조건(모두): 메모가 1차 자동 추가 원문과 정확히 일치, 보유 0주, 메모 등록일부터 as_of까지
    그 종목 거래일 수가 expiry_days 이상(core.signals.check_a1_expired와 같은 판정).
    덧붙인 메모는 원문 패턴과 더는 일치하지 않으므로 한 번만 붙는다.

    입력: plan_rows([{"ticker","memo"}]), held_qty_by_ticker({티커: 체결 탭 기준 순보유 주식 수}),
         trading_dates_by_ticker({티커: 그 종목 일봉 날짜 인덱스}), as_of(오늘 기준일),
         run_date(실행일 KST), expiry_days, a1_stage_label(1차 차수 표시 이름)
    출력: [{"ticker","old_memo","new_memo"}]
    """
    pattern = _auto_memo_pattern(a1_stage_label)
    as_of_ts = pd.Timestamp(as_of)
    out = []
    for row in plan_rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        memo = str(row.get("memo") or "")
        m = pattern.match(memo)
        if not ticker or m is None:
            continue
        if (held_qty_by_ticker.get(ticker) or 0) > 0:
            continue
        dates = trading_dates_by_ticker.get(ticker)
        if dates is None:
            continue
        reg = pd.Timestamp(m.group(1))
        bars = int(((dates >= reg) & (dates <= as_of_ts)).sum())
        if not sig.check_a1_expired(bars, expiry_days):
            continue
        out.append({"ticker": ticker, "old_memo": memo, "new_memo": f"{memo} · {EXPIRE_MARK} {run_date.isoformat()}"})
    return out


def merge_plan_rows(plan_df: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    """새로 추가한 계획 줄을 기존 계획 DataFrame에 합친다(메모리 병합, 같은 실행의 수량 계산용).

    입력: plan_df(ticker, budget_krw, ref_price, memo), new_rows(select_new_plan_rows 결과 중 실제 추가한 줄)
    출력: 새 DataFrame. 이미 있는 티커는 기존 줄을 그대로 둔다(덮어쓰지 않음).
    """
    columns = ["ticker", "budget_krw", "ref_price", "memo"]
    existing = set(plan_df["ticker"]) if plan_df is not None and not plan_df.empty else set()
    add = [{c: r[c] for c in columns} for r in new_rows if r["ticker"] not in existing]
    if not add:
        return plan_df
    base = plan_df if plan_df is not None and not plan_df.empty else pd.DataFrame(columns=columns)
    return pd.concat([base, pd.DataFrame(add, columns=columns)], ignore_index=True)
