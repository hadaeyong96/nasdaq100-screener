"""오늘 확정 기준일 하루를 처리해 매수·매도·손절 신호와 추천 수량을 뽑고,
HTML 보고서·텔레그램 브리핑까지 만든다.

실행:
    python -m engine.daily              live/paper 모드로 오늘까지 처리, 보고서+텔레그램(토큰 있으면)
    python -m engine.daily --no-send     보고서만 만들고 텔레그램은 보내지 않음
    python -m engine.daily --mode paper  config.yaml의 mode보다 이 값을 우선
    python -m engine.daily --replay      state.db가 비어 있으면 최근 replay.lookback_days
                                          거래일을 하루씩 되돌려 처리해 상태를 만든 뒤 오늘까지 진행
                                          (테스트·백테스트(P5)용. 운용 시작 기본 경로가 아니다)
    python -m engine.daily --dry-run     DB에 쓰지 않고 결과만 출력

## 운용 모드 (P3)

live 모드는 보유를 오직 data/fills.csv의 실제 체결 기록으로만 만든다(가상 체결
없음). paper 모드는 추천대로 체결됐다고 가정하는 모의 운용이고, DB를
data/paper_state.db로 완전히 분리한다.

두 모드 모두 "오늘"부터 시작한다. 각 DB는 마지막으로 처리를 끝낸 기준일을
`meta` 테이블에 `last_processed_date`로 저장해 두고, 매 실행마다 상태를
init_state에서부터 그 다음 거래일~오늘(이번 기준일)까지 순서대로 다시
계산한다(`simulate_since`, P3.2 3번). 그래야
- 사용자가 체결 기록을 며칠 늦게 넣어도(fills.csv에 지난 날짜로 한 줄 추가)
  다음 실행에서 그 날짜부터 다시 계산돼 올바른 단계·손절가로 반영되고,
- 스크립트를 며칠 걸러 실행해도 건너뛴 거래일의 이벤트가 누락되지 않는다.
DB에 저장된 상태를 이어받아 증분으로만 갱신하면 이미 지나간 날짜의 체결
기록이 반영될 기회가 없어져 이 성질이 깨진다. 이번 기준일이
last_processed_date 이하이면(같은 날 재실행 등) 아무것도 하지 않는다.

이번 기준일이 실행 시각 기준 가장 최근에 마감된 거래일(NYSE 캘린더,
data/market_calendar.py)보다 오래됐으면 "데이터 지연 모드"로 전환해 상태
전이·주문대기 생성·이벤트 기록을 하지 않고 지연 알림만 보낸다(P3.2 2번).

--replay(레거시, P2)는 이 규칙과 무관하게 예전 그대로 남겨 둔다: state.db가
비어 있을 때만 replay.lookback_days거래일을 가상 체결로 되돌려 처리한다.
테스트·백테스트(P5) 전용이고 운용 시작의 기본 경로가 아니다.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):  # 윈도우 콘솔 cp949 UnicodeEncodeError 방지
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from core import auto_plan  # noqa: E402
from core import explain as expl  # noqa: E402
from core import filters as filt  # noqa: E402
from core import macro_status  # noqa: E402
from core import screening_view as sview  # noqa: E402
from core import sizing  # noqa: E402
from core import signals as sig  # noqa: E402
from core import state as st  # noqa: E402
from core.indicators import compute_indicators  # noqa: E402
from data import fx  # noqa: E402
from data import macro as macrodata  # noqa: E402
from data.earnings import get_earnings_dates  # noqa: E402
from data.fills import fill_key, load_fills, load_plan, summarize_cash_rows  # noqa: E402
from data.market_calendar import latest_closed_trading_day, trading_days_between  # noqa: E402
from data.prices import US_EASTERN, fetch_universe_prices  # noqa: E402
from data import sheets  # noqa: E402
from data.universe import get_universe  # noqa: E402
from notify import briefing, report_html, telegram  # noqa: E402
from store import db  # noqa: E402

OUTPUT_DIR = ROOT / "outputs"
KST = ZoneInfo("Asia/Seoul")  # 계획 자동 기록 등록일(실행일 KST)

_BUY_KINDS = ("A1", "A2", "A3", "B")
_SELL_KINDS = ("STOP", "A1_EXPIRE", "E3", "E1", "E2")
_SELL_REASON = {
    "STOP": "손절",
    "A1_EXPIRE": "A1 만료",
    "E3": "구조 붕괴(E3)",
    "E1": "모멘텀 약화(E1, 데드크로스)",
    "E2": "추세 약화(E2, RSI 50 이탈)",
}
# 표기 규칙: "묶음" 대신 매도 범위를 이렇게 쓴다.
_SELL_RANGE_LABEL = {
    "STOP": "전량",
    "E3": "3차분(67%) 또는 잔량",
    "A1_EXPIRE": "1차분 (전량)",
    "E1": "1차분 매도(11%)",
    "E2": "2차분 매도(22%)",
}
_STAGE_LABEL = {"A1": "1차 · RSI 30 탈출", "A2": "2차 · 골든크로스", "A3": "3차 · 구름 돌파", "B": "추세 재진입"}
_STAGE_TO_BUCKET = {"A1": "b1", "A2": "b2", "A3": "b3", "B": "b9"}
# "전체" 매수 탭의 차수 칸 (P3.6 2번).
_STAGE_FULL_LABEL = {"A1": "1차 정찰", "A2": "2차 확인", "A3": "3차 확정", "B": "재진입"}
# 새 포지션(보유 중이 아니던 종목의 첫 신호)만 자금 계획의 "남은 한도" 배분을 받는다 —
# A2·A3는 core/state.py가 대기 상태에서만 A1·B를 판정하므로 항상 기존 포지션의 추가 매수다.
_NEW_POSITION_KINDS = ("A1", "B")
_STAGE_RISK_KEY = {"A1": "a1_budget_pct", "A2": "a2_budget_pct", "A3": "a3_budget_pct", "B": "b_budget_pct"}

# ── 손절 예약 알림 (P3.5) ────────────────────────────────────────────────
# 우선순위: 손절(실제 매도 신호) > 손절 근접 > 손절 예약 변경 > 손절 예약 필요.
# 배지는 종목당 하나만 보유 현황 비고 칸에 쓰고, 나머지는 경고 탭에 모두 남긴다.
_STOP_ALERT_STOP = ("stop", "b-sell", "손절 · 예약 체결 확인")
_MODE_LABEL = {"live": "실전", "paper": "모의"}

# 표기 규칙: "휩소" 대신 "잦은 교차 (횡보)". E1·E2·E3는 "모멘텀 약화·추세 약화·구조 붕괴"와 함께 표기.
_FILTER_REASON_LABEL = {
    "골든크로스 당일 RSI 70 이상": "과열 (RSI 70 이상)",
}

# ── 라이브 어드바이저 1단계 (docs/design/live_advisor.md 3번) ──────────────
# core.state.process_day가 내는 이벤트를 새 신호 로직 없이 4가지 판정으로 재분류한다.
_JUDGMENT_HOLD = "보유"
_JUDGMENT_ADD = "추가매수"
_JUDGMENT_TRIM = "일부매도 검토"
_JUDGMENT_SELL = "매도"
# 우선순위: 매도 > 일부매도 검토 > 추가매수 > 보유 (같은 날 여러 이벤트가 겹치면 더 급한 쪽).
_SELL_JUDGMENT_KINDS = ("E3", "STOP", "A1_EXPIRE")
_TRIM_JUDGMENT_KINDS = ("E1", "E2")


def _stop_alert_candidates(
    close: float | None, stop: float | None, stop_near_pct: float, changed: dict | None, needs_order: bool
) -> list[tuple[str, str, str]]:
    """손절 근접·예약 변경·예약 필요 알림 후보를 우선순위 순으로 만든다 (P3.5 1번).

    호출부가 이미 오늘 손절(STOP) 신호가 난 종목은 걸러내고 부른다 — 손절
    신호일에는 이 알림들을 보이지 않는다. 순수 함수(네트워크·상태 없음).
    입력: close(종가), stop(현재 손절가), stop_near_pct(설정값), changed(오늘
         stop_changed 이벤트 dict 또는 None), needs_order(오늘 새로 체결된
         매수가 있어 손절 예약이 필요한지)
    출력: [(type, badge_class, text), ...] 우선순위 순(근접>변경>신규), 없으면 []
    """
    candidates: list[tuple[str, str, str]] = []
    if stop is not None and close is not None and sig.stop_near(close, stop, stop_near_pct):
        candidates.append(("근접", "b-warn", f"손절 근접 · 예약 ${stop:,.2f} 확인"))
    if changed is not None:
        candidates.append(
            ("변경", "b-info", f"손절 예약 변경 · ${changed['old_stop']:,.2f} → ${changed['new_stop']:,.2f}")
        )
    if needs_order:
        text = f"손절 예약 필요 · ${stop:,.2f}" if stop is not None else "손절 예약 필요"
        candidates.append(("신규", "b-info", text))
    return candidates


def _stop_event_tickers(today_events: list[dict]) -> set[str]:
    """오늘 손절(STOP) 신호가 난 종목 (P3.5 1번 — 이 종목은 근접 알림을 보이지 않는다)."""
    return {e["ticker"] for e in today_events if e["kind"] == "STOP"}


def _stop_changed_by_ticker(today_events: list[dict]) -> dict[str, dict]:
    """오늘 손절가가 바뀐(stop_changed) 종목별 이벤트 (P3.5 1번)."""
    return {e["ticker"]: e for e in today_events if e["kind"] == "stop_changed"}


def _need_stop_order_tickers(today_events: list[dict]) -> set[str]:
    """오늘 새로 체결된 매수가 있어 손절 예약이 필요한 종목 (P3.5 1번 — "손절 예약 필요").

    live 모드의 실제 체결(FILL)과 paper 모드의 가상 체결(VIRTUAL_FILL)을 모두 본다.
    """
    return {
        e["ticker"]
        for e in today_events
        if e["kind"] in ("FILL", "VIRTUAL_FILL") and e.get("side") == "buy" and (e.get("qty") or 0) > 0
    }


def _order_guidance(kind: str, qty: int, stop_price: float | None, cfg: dict) -> str:
    """매도·손절 탭의 "주문 안내" 칸 문구 (P3.5 5번). 손절만 문구가 다르다."""
    if kind == "STOP":
        fallback = cfg["orders"]["stop_fallback_type"]
        stop_txt = f"${stop_price:,.2f}" if stop_price is not None else "미확인"
        return (
            f"손절 예약 확인 · 증권사 손절 예약({stop_txt})이 오늘 체결됐으면 체결 기록만 입력. "
            f"체결 안 됐으면 다음 거래일 장 시작 시 {qty}주 전량 {fallback} 매도 예약"
        )
    return f"다음 거래일 장 시작 시 {qty}주 매도 주문 예약"


def _judgment_for_events(events: list[dict]) -> tuple[str, dict | None]:
    """오늘 이벤트를 4가지 판정(보유·추가매수·일부매도 검토·매도)으로 재분류한다
    (docs/design/live_advisor.md 3번 — 새 신호 로직 없음, core.state의 이벤트 종류를
    그대로 재분류만 한다). 우선순위는 모듈 상수 _SELL_JUDGMENT_KINDS 등 주석 참고.

    출력: (판정 문자열, 그 판정을 낸 이벤트 dict 또는 None(보유일 때))
    """
    by_kind = {e["kind"]: e for e in events}
    for kind in _SELL_JUDGMENT_KINDS:
        if kind in by_kind:
            return _JUDGMENT_SELL, by_kind[kind]
    for kind in _TRIM_JUDGMENT_KINDS:
        if kind in by_kind:
            return _JUDGMENT_TRIM, by_kind[kind]
    for kind in _BUY_KINDS:
        if kind in by_kind:
            return _JUDGMENT_ADD, by_kind[kind]
    return _JUDGMENT_HOLD, None


def _judgment_reason(judgment: str, event: dict | None) -> str:
    """판정 카드에 보일 이유 한 줄 (docs/design/live_advisor.md 3번, 7번)."""
    if judgment == _JUDGMENT_HOLD or event is None:
        return "오늘 신호 없음"
    if judgment in (_JUDGMENT_SELL, _JUDGMENT_TRIM):
        suffix = "매도" if judgment == _JUDGMENT_SELL else "매도 검토"
        return f"{_SELL_REASON[event['kind']]} · {_SELL_RANGE_LABEL[event['kind']]} {suffix}"
    return f"{_STAGE_LABEL[event['kind']]} 신호"


def _live_target_tickers(plan_by_ticker: dict[str, float], fills_df: pd.DataFrame) -> set[str]:
    """계획∪체결 종목 (docs/design/live_advisor.md 3·7번) — 라이브 판정 카드는 이
    종목들만 대상으로 한다. "오늘의 추천" 탭(build_report_summary의 buy_groups)은
    이 함수와 무관하게 여전히 전체 스캔 결과를 그대로 보여준다."""
    fills_tickers = set(fills_df["ticker"].unique()) if fills_df is not None and not fills_df.empty else set()
    return set(plan_by_ticker) | fills_tickers


def _plan_budget_map(plan_df: pd.DataFrame) -> dict[str, float]:
    """계획 DataFrame(ticker, budget_krw, ref_price, memo)을 {ticker: budget_krw}로 바꾼다."""
    if plan_df is None or plan_df.empty:
        return {}
    return dict(zip(plan_df["ticker"], plan_df["budget_krw"]))


def _plan_ref_price_map(plan_df: pd.DataFrame) -> dict[str, float]:
    """계획 DataFrame의 "기준가($)" 열 -> {ticker: ref_price} (기준가를 적은 종목만).

    기준가가 있으면 core.sizing.plan_tranche_qty가 그날그날 바뀌는 지정가 대신
    이 고정값으로 총 주수를 정한다 — 시트의 1차~남은(주) 수식과 같은 결과를
    내기 위해서다(구글 시트 계획 탭 개편, 2026-09-30).
    """
    if plan_df is None or plan_df.empty or "ref_price" not in plan_df.columns:
        return {}
    return {
        ticker: price
        for ticker, price in zip(plan_df["ticker"], plan_df["ref_price"])
        if price is not None and not pd.isna(price)
    }


def _reference_entry_price(
    ticker: str, today_events: list[dict], indicator_map: dict, as_of_by_ticker: dict, cfg: dict
) -> float | None:
    """이 종목의 오늘 참조 진입가(달러, 지정가 기준). 오늘 매수 신호(A1·A2·A3·B)가
    있으면 그 신호의 지정가를 그대로 쓰고, 없으면(판정이 보유·매도라 신호가 없는
    날에도) 오늘 종가 기준 지정가를 참고용으로 쓴다 — 계획금액이 1주도 못 사는지
    신호 여부와 무관하게 상시 경고하기 위해 필요하다(docs/design/live_advisor.md 7번).
    """
    for e in today_events:
        if e["ticker"] == ticker and e["kind"] in _BUY_KINDS:
            return sig.entry_limit_price(e["price"], cfg)
    df = indicator_map.get(ticker)
    date = as_of_by_ticker.get(ticker)
    if df is None or date is None or date not in df.index:
        return None
    close = df.loc[date, "close"]
    if pd.isna(close):
        return None
    return sig.entry_limit_price(float(close), cfg)


def compute_live_judgments(
    states: dict,
    today_events: list[dict],
    indicator_map: dict,
    as_of_by_ticker: dict,
    name_map: dict,
    plan_by_ticker: dict[str, float],
    fills_df: pd.DataFrame,
    fx_rate: float | None,
    cfg: dict,
    plan_ref_price_by_ticker: dict[str, float] | None = None,
) -> list[dict]:
    """라이브 전용: 계획∪체결 종목마다 오늘 판정을 정하고 states[ticker]["last_judgment"]를
    그 자리에서 갱신한다 (docs/design/live_advisor.md 3·7번). 호출부(run())가 이 함수를
    부른 뒤에 db.save_position으로 저장해야 오늘 판정이 다음 실행의 "바뀐 판정" 비교
    기준이 된다 — 이 함수 자체는 DB에 쓰지 않는다.

    paper 모드는 이 함수를 부르지 않는다(호출부 책임) — states에 last_judgment를 남기지
    않는다(engine/daily.py 모듈 설명의 P5-1 0번 예외 참고).

    입력: states(오늘 시뮬레이션 반영 후 상태), today_events, indicator_map, as_of_by_ticker,
         name_map, plan_by_ticker({ticker: 계획금액(원)}), fills_df(계획∪체결 종목 판단용),
         fx_rate(원/달러, 없으면 수량·경고 계산 생략), cfg, plan_ref_price_by_ticker
         ({ticker: 계획 시트 "기준가($)"} — 있으면 core.sizing.plan_tranche_qty가 기준가
         기반 고정 총 주수 방식을 쓴다. 없는 종목은 지금처럼 오늘 지정가 기준)
    출력: [{"ticker","name_kr","judgment","reason","changed","stop_price","has_plan",
          "plan_budget_krw","qty","tranche_krw","tranche_usd","one_share_warning",
          "plan_limit_exceeded"}, ...] 티커 오름차순.
          one_share_warning은 {"min_budget_krw","ref_price"} 또는 None.
          plan_limit_exceeded: 기준가 기반 계획에서 이미 보유 수량이 총 주수 이상이면 True.
    """
    plan_ref_price_by_ticker = plan_ref_price_by_ticker or {}
    target_tickers = _live_target_tickers(plan_by_ticker, fills_df)
    events_by_ticker: dict[str, list[dict]] = {}
    for e in today_events:
        events_by_ticker.setdefault(e["ticker"], []).append(e)

    rows = []
    for ticker in sorted(target_tickers):
        state_ = states.get(ticker)
        if state_ is None:  # 지표를 못 받은 종목(상장폐지 등) — 판정 불가
            continue
        events = events_by_ticker.get(ticker, [])
        judgment, event = _judgment_for_events(events)
        reason = _judgment_reason(judgment, event)

        previous = state_.get("last_judgment")
        changed = previous is not None and previous != judgment
        state_["last_judgment"] = judgment

        has_plan = ticker in plan_by_ticker
        budget = plan_by_ticker.get(ticker)
        qty = tranche_krw = tranche_usd = None
        plan_limit_exceeded = False
        if judgment == _JUDGMENT_ADD and has_plan and fx_rate:
            sized = sizing.plan_tranche_qty(
                budget, event["kind"], sig.entry_limit_price(event["price"], cfg), fx_rate,
                ref_price=plan_ref_price_by_ticker.get(ticker),
            )
            held_qty = sum(q for q in state_["units"].values() if q > 0)
            clamped = sizing.clamp_plan_qty_to_remaining(sized["qty"], sized.get("total_shares"), held_qty)
            qty, tranche_krw, tranche_usd = clamped["qty"], sized["tranche_krw"], sized["tranche_usd"]
            plan_limit_exceeded = clamped["over_limit"]

        one_share_warning = None
        if has_plan and fx_rate:
            ref_price = _reference_entry_price(ticker, today_events, indicator_map, as_of_by_ticker, cfg)
            if ref_price is not None:
                min_budget = sizing.min_budget_for_one_share_krw(ref_price, fx_rate)
                if budget < min_budget:
                    one_share_warning = {"min_budget_krw": min_budget, "ref_price": ref_price}

        rows.append(
            {
                "ticker": ticker,
                "name_kr": name_map.get(ticker, "") or ticker,
                "judgment": judgment,
                "reason": reason,
                "changed": changed,
                "stop_price": state_.get("stop"),
                "has_plan": has_plan,
                "plan_budget_krw": budget,
                "qty": qty,
                "tranche_krw": tranche_krw,
                "tranche_usd": tranche_usd,
                "one_share_warning": one_share_warning,
                "plan_limit_exceeded": plan_limit_exceeded,
            }
        )
    return rows


def apply_auto_plan(
    plan_df: pd.DataFrame,
    fills_df: pd.DataFrame,
    budget_krw: float | None,
    sheets_client,
    write: bool,
) -> tuple[pd.DataFrame, list[str], dict]:
    """계획 탭 자동 기록 v2 규칙 D (docs/design/auto_plan.md) — live 전용, 호출부가 모드를 거른다.

    첫 매수 후 계획 탭에 없는 종목(보유 > 0)을 골라(core.auto_plan) 시트 계획 탭에 쓰고(write=True이고
    sheets_client가 있을 때만), 결과와 무관하게 그 줄을 계획 DataFrame에 메모리로 합친다 —
    시트 쓰기가 실패해도 오늘 판정·수량은 그 계획으로 계산된다. 실패는 예외 대신 경고 문구로 돌려준다.

    입력: plan_df(읽은 계획), fills_df(대기자금 줄을 뺀 체결 기록), budget_krw(투자현황 "종목당 계획금액",
         없으면 None — 쓰지 않고 경고), sheets_client, write(False면 시트를 건드리지 않음 — dry-run)
    출력: (합친 plan_df, 경고 목록, {"selected","written","skipped"})
    """
    info = {"selected": [], "written": [], "skipped": []}
    existing = list(plan_df["ticker"]) if plan_df is not None and not plan_df.empty else []
    if budget_krw is None:
        pending = auto_plan.select_first_buy_plan_rows(fills_df, existing, 0)
        if not pending:
            return plan_df, [], info
        tickers = ", ".join(r["ticker"] for r in pending)
        return plan_df, [f"계획 자동 기록 생략: 투자현황 탭에 종목당 계획금액이 없어 계획 줄을 쓰지 않았습니다 ({tickers})"], info

    new_rows = auto_plan.select_first_buy_plan_rows(fills_df, existing, budget_krw)
    info["selected"] = [r["ticker"] for r in new_rows]

    warnings: list[str] = []
    rows_to_merge = new_rows
    if write and sheets_client is not None and new_rows:
        try:
            result = sheets.write_auto_plan(sheets_client, new_rows)
            info["written"] = [r["ticker"] for r in result.written]
            info["skipped"] = result.skipped
            # 시트에 이미 있던 티커(계획 DataFrame에서 빠졌던 줄)는 메모리에도 합치지 않는다.
            rows_to_merge = [r for r in new_rows if r["ticker"] not in set(result.already_present)]
        except Exception as exc:
            warnings.append(f"계획 자동 기록 실패: {exc}")
    return auto_plan.merge_plan_rows(plan_df, rows_to_merge), warnings, info


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_mode(cfg: dict, cli_mode: str | None) -> str:
    """운용 모드를 정한다. --mode가 config.yaml의 mode보다 우선한다."""
    mode = cli_mode or cfg.get("mode", "live")
    if mode not in ("live", "paper"):
        raise ValueError(f"알 수 없는 모드: {mode!r} (live 또는 paper만 허용)")
    return mode


def _label_filter_reason(reason: str) -> str:
    """매매 금지 이유를 보고서 표기 규칙에 맞게 바꾼다."""
    if reason in _FILTER_REASON_LABEL:
        return _FILTER_REASON_LABEL[reason]
    if "휩소" in reason:
        return "잦은 교차 (횡보)"
    return reason


def _ledger_row(rec: dict) -> dict:
    """반영한 실제 체결 한 줄 -> store fill_ledger 행 (live)."""
    fx_rate = rec.get("fx_rate")
    fx_rate = None if fx_rate is None or pd.isna(fx_rate) else float(fx_rate)
    price, qty = float(rec["price"]), int(rec["qty"])
    return {
        "date": str(pd.Timestamp(rec["date"]).date()), "mode": "live", "ticker": rec["ticker"], "side": rec["side"],
        "qty": qty, "price_usd": price, "fx_rate": fx_rate,
        "amount_krw": round(price * qty * fx_rate) if fx_rate else None, "qqqm_close": None,
    }


def _dates_since_start(df: pd.DataFrame, start_date) -> list:
    """이 종목에 대해 start_date(포함, last_processed_date 다음 거래일)부터 오늘까지
    처리할 날짜 목록을 정한다 (P3.2 3번 — 건너뛴 거래일이 있으면 모두 포함된다).

    종목이 start_date 이후 상장했으면(신규 상장) 그 종목의 첫 거래일부터 쓴다.
    start_date가 df 범위보다 미래면(그 종목만 데이터가 하루 뒤처지는 등) 마지막
    날짜 하나만 쓴다.
    """
    mask = df.index >= pd.Timestamp(start_date)
    dates = list(df.index[mask])
    return dates if dates else [df.index[-1]]


def _dates_to_process(df: pd.DataFrame, do_replay: bool, lookback_days: int) -> list:
    """레거시 --replay 경로 전용: 오늘만, 또는 되돌려 보기 포함 날짜 목록."""
    if not do_replay:
        return [df.index[-1]]
    n = min(lookback_days + 1, len(df))
    return list(df.index[-n:])


def _average_entry_price(state: dict) -> float | None:
    pairs = [(p, state["units"].get(u, 0)) for u, p in state["entries"].items() if p is not None]
    pairs = [(p, q) for p, q in pairs if q > 0]
    if not pairs:
        return None
    total_qty = sum(q for _, q in pairs)
    return sum(p * q for p, q in pairs) / total_qty


def _held_count(states: dict) -> int:
    return sum(1 for s in states.values() if any(q > 0 for q in s["units"].values()))


def _held_signals_snapshot(states: dict, indicator_map: dict, date) -> list[dict]:
    """core.sizing.size_buy_signals의 held 입력을 오늘(date) 기준 상태에서 만든다 (P5-1 0번).

    보유 중(단위 수량>0)인 종목만 담는다. "주문대기"이면서 이전 신호로 이미 어느
    단계를 지나 있었으면(예: A2 신호가 나 정찰→주문대기로 넘어간 경우) 예약 계산은
    신호 전 상태(pending.prev_state)를 쓴다 — 오늘의 새 신호 자체는 아직 반영 전이므로.

    입력: states({ticker: 상태}), indicator_map({ticker: df}), date(종가 조회용)
    출력: [{"ticker","qty","close","state_label"}, ...]
    """
    out = []
    for ticker, state_ in states.items():
        qty_held = sum(q for q in state_["units"].values() if q > 0)
        if qty_held <= 0:
            continue
        df = indicator_map.get(ticker)
        close = None
        if df is not None and date in df.index:
            c = df.loc[date, "close"]
            close = float(c) if not pd.isna(c) else None
        state_label = state_["state"]
        if state_label == "주문대기" and state_.get("pending"):
            state_label = state_["pending"].get("prev_state", state_label)
        out.append({"ticker": ticker, "qty": qty_held, "close": close, "state_label": state_label})
    return out


def _todays_buy_signals(events_by_ticker: dict, states: dict, cfg: dict) -> tuple[list[dict], dict]:
    """오늘 발생한 매수 이벤트들을 core.sizing.size_buy_signals의 signals 입력으로 바꾼다 (P5-1 0번).

    입력: events_by_ticker({ticker: [오늘 이벤트, ...]}), states(process_day 반영 후 —
         stop이 이미 갱신돼 있다), cfg
    출력: (signals 목록, {key: (ticker, event, entry_price)} — 체결 적용 시 역참조용)
    """
    signals: list[dict] = []
    lookup: dict = {}
    for ticker, events in events_by_ticker.items():
        for event in events:
            if event["kind"] not in _BUY_KINDS:
                continue
            entry_price = sig.entry_limit_price(event["price"], cfg)
            stop_price = states[ticker].get("stop")
            key = f"{ticker}-{event['kind']}"
            signals.append(
                {
                    "key": key,
                    "stage": event["kind"],
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "score": event.get("score", 0),
                    "is_new_position": event["kind"] in _NEW_POSITION_KINDS,
                }
            )
            lookup[key] = (ticker, event, entry_price)
    return signals, lookup


def simulate_since(
    indicator_map: dict,
    per_ticker_dates: dict,
    states: dict,
    cfg: dict,
    earnings_map: dict,
    gap_dates_by_ticker: dict,
    fills_df: pd.DataFrame,
    virtual_fill: bool,
    max_concurrent: int,
    fx_rate_by_date: dict | None = None,
    applied_fill_keys: Counter | None = None,
) -> dict:
    """여러 종목의 날짜별 상태 전이를 한 번에 처리한다 (네트워크·DB 없음, 테스트 가능).

    실제 체결(live, virtual_fill=False) 반영 규칙:
      - 주문대기 확정 전: `주문대기`인 종목은 (신호일, 처리일] 날짜의 체결을 process_day
        **전에** 반영한다 — 실제 매수는 신호 다음 날이라, 먼저 반영해야
        core.state._resolve_pending이 UNFILLED가 아니라 원래 단계(정찰 등)로 확정한다.
      - 그 밖의 체결은 process_day 뒤에, 처리일 이하 날짜 중 아직 반영 안 된 것을 반영한다
        (주말·휴장일 날짜, data_gap으로 건너뛴 날의 체결도 다음 처리일에 반영된다).
      - applied_fill_keys(이전 실행까지 반영한 체결 키 Counter — store fill_ledger)에 있는
        체결은 다시 반영하지 않는다(중복 반영 방지). 새로 반영한 체결은 applied_fills로 돌려준다.

    매매 금지 구간·동시 보유 한도는 종목을 가로질러 점수를 비교해야 해서, 날짜마다
    먼저 모든 종목의 새 진입 후보(core.state.preview_new_entry)를 모아 점수로
    추려낸 뒤에야 각 종목을 실제로 처리한다(engine/daily.py 모듈 설명 참고).

    가상 체결(virtual_fill)의 수량도 라이브 보고서와 똑같이 core.sizing.size_buy_signals
    (자금 계획: 슬롯+위험 상한+남은 한도+예약)로 정한다 (P5-1 0번 — 라이브 보고서 수량과
    paper 가상 체결 수량이 갈리면 안 된다). 그래서 하루 안에서도 모든 종목의 process_day를
    먼저 끝내 오늘의 매수 신호를 다 모은 뒤에야 한 번에 크기를 매긴다(점수 순 남은 한도
    배분이 종목을 가로질러야 하므로).

    입력: indicator_map({ticker: df}), per_ticker_dates({ticker: [처리할 날짜, ...]}),
         states({ticker: 시작 상태}), cfg, earnings_map({ticker: 실적일 또는 None}),
         gap_dates_by_ticker({ticker: {data_gap 날짜, ...}}), fills_df(실제 체결 기록),
         virtual_fill(True면 추천대로 체결됐다고 가정하는 가상 체결을 쓴다 — paper 모드·
         레거시 --replay용. False면 fills_df의 실제 체결 기록만 쓴다 — live 모드),
         max_concurrent(동시 보유 한도), fx_rate_by_date({날짜.isoformat(): 원/달러 환율} —
         virtual_fill일 때만 쓴다. 그 날짜가 없으면 그날은 수량 0)
    출력: {states, all_events, today_events, warnings, data_gap_tickers,
          as_of_by_ticker, start_by_ticker}
    """
    fx_rate_by_date = fx_rate_by_date or {}
    applied_fills: list[dict] = []
    fill_rows_by_ticker: dict[str, list[dict]] = {}
    if not virtual_fill and fills_df is not None and not fills_df.empty:
        already = Counter(applied_fill_keys or {})
        for rec in fills_df.sort_values("date", kind="stable").to_dict(orient="records"):
            key = fill_key(rec)
            rec["_done"] = already[key] > 0  # 이전 실행에서 이미 반영한 줄(같은 키가 여러 줄이면 그 수만큼)
            if rec["_done"]:
                already[key] -= 1
            fill_rows_by_ticker.setdefault(rec["ticker"], []).append(rec)

    def _apply_live_fills(ticker: str, date, after=None) -> list[dict]:
        """(after, date] 날짜의 아직 반영 안 된 체결을 states[ticker]에 반영하고 FILL 이벤트를 돌려준다."""
        out = []
        for rec in fill_rows_by_ticker.get(ticker, []):
            fdate = pd.Timestamp(rec["date"])
            if rec["_done"] or fdate > pd.Timestamp(date) or (after is not None and fdate <= pd.Timestamp(after)):
                continue
            rec["_done"] = True
            fill = {k: rec[k] for k in ("date", "ticker", "unit", "side", "price", "qty")}
            states[ticker] = st.apply_fill(states[ticker], fill, cfg)
            applied_fills.append({k: v for k, v in rec.items() if k != "_done"})
            out.append({**fill, "date": date, "fill_date": str(fdate.date()), "kind": "FILL"})
        return out

    master_dates = sorted(set().union(*per_ticker_dates.values())) if per_ticker_dates else []

    all_events: list[dict] = []
    today_events: list[dict] = []
    warnings: list[str] = []
    data_gap_tickers: list[str] = []
    as_of_by_ticker: dict = {}
    start_by_ticker: dict = {}

    for date in master_dates:
        active = [t for t in indicator_map if date in per_ticker_dates[t]]

        # ── Pass 1: data_gap 제외 + 오늘 새 진입(A1·B) 후보를 모두 모아 점수로 추린다 ──
        skip_today: set = set()
        candidates = []
        for ticker in active:
            if date in gap_dates_by_ticker.get(ticker, set()):
                skip_today.add(ticker)
                warnings.append(f"{ticker} {date.date()} data_gap - 신호 판정에서 제외")
                if ticker not in data_gap_tickers:
                    data_gap_tickers.append(ticker)
                continue
            cand = st.preview_new_entry(indicator_map[ticker], date, states[ticker], cfg, earnings_map.get(ticker))
            if cand:
                candidates.append({**cand, "ticker": ticker})

        candidates.sort(key=lambda c: c["score"], reverse=True)
        slots = max(max_concurrent - _held_count(states), 0)
        admitted = {c["ticker"] for c in candidates[:slots]}

        # ── Pass 2a: 상태 전이를 모든 종목에 대해 먼저 끝낸다 (같은 core 함수를 라이브·
        # paper·되돌려 보기 모두에 쓴다) — 가상 체결 크기는 오늘의 매수 신호를 모두 모은
        # 뒤에야(Pass 2b) 종목을 가로질러 점수 순으로 정할 수 있어서다.
        events_by_ticker: dict[str, list[dict]] = {}
        for ticker in active:
            if ticker in skip_today:
                continue
            df = indicator_map[ticker]
            pre_fill_events: list[dict] = []
            pending = states[ticker].get("pending")
            if not virtual_fill and states[ticker]["state"] == "주문대기" and pending:
                # 주문대기 확정 전에 (신호일, 처리일] 체결을 먼저 반영한다 (신호 다음 날 매수).
                pre_fill_events = _apply_live_fills(ticker, date, after=pending["date"])
            events, states[ticker] = st.process_day(
                df, date, states[ticker], cfg, earnings_date=earnings_map.get(ticker), new_entry_allowed=(ticker in admitted)
            )
            for event in events:
                event["ticker"] = ticker
            events_by_ticker[ticker] = pre_fill_events + events

        # ── Pass 2b: 가상 체결(paper·레거시 --replay) — 라이브 보고서와 같은
        # core.sizing.size_buy_signals로 수량을 정한다 (P5-1 0번).
        if virtual_fill:
            fx_rate = fx_rate_by_date.get(pd.Timestamp(date).date().isoformat())
            held = _held_signals_snapshot(states, indicator_map, date)
            signals, lookup = _todays_buy_signals(events_by_ticker, states, cfg)
            sized = sizing.size_buy_signals(signals, held, cfg, fx_rate)
            for key, row in sized["rows"].items():
                if row["qty"] <= 0:
                    continue
                ticker, event, entry_price = lookup[key]
                fill = {"unit": event["unit"], "side": "buy", "price": entry_price, "qty": row["qty"]}
                states[ticker] = st.apply_fill(states[ticker], fill, cfg)
                events_by_ticker[ticker].append(
                    {"date": date, "ticker": ticker, "kind": "VIRTUAL_FILL", "unit": event["unit"], **fill}
                )

        # ── Pass 2c: 실제 체결(live) 반영 + 마무리 ──
        for ticker in active:
            if ticker in skip_today:
                continue
            events = events_by_ticker[ticker]
            if not virtual_fill:
                # 처리일 이하 날짜 중 아직 반영 안 된 체결 (신호 당일 체결, 주말·휴장일 날짜 등)
                events.extend(_apply_live_fills(ticker, date))

            for event in events:
                event["state_after"] = states[ticker]["state"]
                event["stop_after"] = states[ticker].get("stop")

            all_events.extend(events)
            if date == per_ticker_dates[ticker][-1]:
                today_events.extend(events)
                as_of_by_ticker[ticker] = date
            if date == per_ticker_dates[ticker][0]:
                start_by_ticker[ticker] = date

    return {
        "states": states,
        "all_events": all_events,
        "today_events": today_events,
        "warnings": warnings,
        "data_gap_tickers": data_gap_tickers,
        "as_of_by_ticker": as_of_by_ticker,
        "start_by_ticker": start_by_ticker,
        "applied_fills": applied_fills,
    }


def _pending_order_rows(states: dict, name_map, mode: str) -> list[dict]:
    """live 모드에서 오늘 매수 신호가 나 체결 확인을 기다리는(`주문대기`) 종목 (P3.1 보완 2번).

    신호 당일 하루만 이 목록에 남는다 — 다음 실행에서 core.state._resolve_pending이
    체결 기록 여부로 원래 단계 또는 미체결(_unfilled_rows_from_events)로 정리한다.
    """
    if mode != "live":
        return []
    rows = []
    for ticker, state_ in states.items():
        pending = state_.get("pending")
        if state_["state"] != "주문대기" or not pending:
            continue
        rows.append(
            {
                "티커": ticker,
                "종목명": name_map.get(ticker, "") or ticker,
                "단계": _STAGE_LABEL.get(pending["kind"], pending["kind"]),
            }
        )
    return rows


def _unfilled_rows_from_events(today_events: list[dict], name_map, mode: str) -> list[dict]:
    """live 모드에서 오늘 "미체결(기록 없음)"로 확정된 종목 (P3.1 보완 1·2번).

    core.state._resolve_pending은 주문대기로 둔 다음 거래일에 체결 기록이 없으면
    "UNFILLED" 이벤트를 한 번만 낸다 — 그러므로 이 목록도 그 다음 날 보고서에만
    나타난다(신호 당일에는 나타나지 않는다).
    """
    if mode != "live":
        return []
    rows = []
    for event in today_events:
        if event["kind"] != "UNFILLED":
            continue
        ticker = event["ticker"]
        rows.append(
            {
                "티커": ticker,
                "종목명": name_map.get(ticker, "") or ticker,
                "단계": _STAGE_LABEL.get(event["stage"], event["stage"]),
                "내용": "미체결 (기록 없음)",
            }
        )
    return rows


def _compute_funnel(today_events: list[dict], indicator_map: dict, as_of_by_ticker: dict) -> dict:
    """통과 현황(1차 RSI 30 돌파 → 2차 골든크로스 → 3차 구름 4요소 → 4차 매매금지 → 5차 보유한도).

    보고서에는 표시하지 않고 outputs/funnel_{모드}_YYYY-MM-DD.csv와 events에만 남긴다 (지시문 3번).
    """
    stage1 = stage2 = 0
    for ticker, df in indicator_map.items():
        date = as_of_by_ticker.get(ticker)
        if date is None or date not in df.index:
            continue
        idx = df.index.get_loc(date)
        row = df.loc[date]
        prev_rsi = df.iloc[idx - 1]["rsi"] if idx > 0 else float("nan")
        if sig.check_a1(prev_rsi, row.get("rsi")):
            stage1 += 1
        if bool(row.get("gc")) if not pd.isna(row.get("gc")) else False:
            stage2 += 1

    stage3 = sum(
        1
        for e in today_events
        if e["kind"] == "A3" or (e["kind"] == "BLOCKED" and e.get("stage") == "A3")
    )
    stage4 = sum(1 for e in today_events if e["kind"] == "BLOCKED" and e.get("blocked_type") == "ban")
    stage5 = sum(1 for e in today_events if e["kind"] == "BLOCKED" and e.get("blocked_type") == "limit")

    return {
        "1차 RSI 30 돌파": stage1,
        "2차 MACD 골든크로스": stage2,
        "3차 일목구름 4요소": stage3,
        "4차 매매금지 필터": stage4,
        "5차 보유한도": stage5,
    }


def _buy_stage_summary(kind: str, base: dict, earnings_date, as_of_date) -> str:
    """9칸 매수 표(P3.4 1번)의 비고 칸: 지표 근거(RSI 변화·등급·거래량 부족·실적 임박 등)를
    한 줄로 요약한다. 조건 열이 없어진 대신 이 문자열 하나로 판단 근거를 남긴다."""
    parts: list[str] = []
    if kind == "A1":
        if base.get("rsi_prev") is not None and base.get("rsi_now") is not None:
            parts.append(f"RSI {base['rsi_prev']} → {base['rsi_now']}")
        vol_ratio = base.get("vol_ratio")
        if vol_ratio is not None and vol_ratio < 1:
            parts.append(f"거래량 부족({vol_ratio}배)")
    elif kind == "A2":
        if base.get("grade"):
            parts.append(f"등급 {base['grade']}")
        if base.get("macd_norm") is not None:
            parts.append(f"MACD 정규화 {base['macd_norm']:+.2f}%")
    elif kind == "A3":
        gap_pct = base.get("gap_pct")
        if gap_pct is not None and abs(gap_pct) >= 1:
            parts.append(f"시가 갭 {gap_pct:+.1f}%")
    else:  # B (재진입)
        parts.append("재진입 1회 전량 · 위험 1%")
        if base.get("grade"):
            parts.append(f"등급 {base['grade']}")

    if earnings_date is not None and as_of_date is not None:
        days = (pd.Timestamp(earnings_date) - pd.Timestamp(as_of_date)).days
        if 0 <= days <= 14:
            parts.append(f"실적 임박 {earnings_date.month}/{earnings_date.day}")

    return " · ".join(parts)


def _build_buy_row(
    event: dict, df: pd.DataFrame, states_after: dict, cfg: dict, name_map, earnings_map, future_trading_days=None
) -> dict:
    """매수 이벤트 하나를 보고서·텔레그램에 쓸 행 dict로 만든다 (탭별 조건 열 포함, P3.6 6-2번).

    수량·투입금액·최대손실은 여기서 계산하지 않는다 — build_report_summary가 오늘의
    모든 매수 신호를 모은 뒤 core.sizing.size_buy_signals 한 번으로 정해 채운다
    (P5-1 0번, 라이브 보고서·paper 가상 체결·백테스트가 항상 같은 수량을 내게 하기 위해).
    이 함수는 지정가·손절가와 탭별 조건 열, 지표 근거 비고만 만든다.
    """
    ticker = event["ticker"]
    date = event["date"]
    row = df.loc[date]
    entry_price = sig.entry_limit_price(event["price"], cfg)
    stop_price = states_after.get("stop")
    earnings_date = earnings_map.get(ticker)
    stop_ok = stop_price is not None
    stop_pct = (stop_price - entry_price) / entry_price * 100 if stop_ok else None

    idx = df.index.get_loc(date)
    prev_rsi = df.iloc[idx - 1]["rsi"] if idx > 0 else float("nan")

    base = {
        "ticker": ticker,
        "kr": name_map.get(ticker, "") or ticker,
        "stage": event["kind"],
        "bucket": _STAGE_TO_BUCKET[event["kind"]],
        "stage_label": _STAGE_FULL_LABEL[event["kind"]],
        "key": f"{ticker}-{event['kind']}",
        "is_new_position": event["kind"] in _NEW_POSITION_KINDS,
        "limit": entry_price,
        "stop": stop_price if stop_ok else None,
        "qty": 0,
        "amount_krw": 0,
        "max_loss_krw": 0,
        "target_qty": 0,
        "risk_cap_qty": 0,
        "stop_pct": round(stop_pct, 1) if stop_pct is not None else None,
        "stop_basis": {"A1": "10일 최저가", "A2": "10일 최저가", "A3": "10일 최저가·구름 하단 중 높은 값", "B": "진입일 10일 최저가"}[
            event["kind"]
        ],
        "earnings": earnings_date.isoformat() if earnings_date else "확인불가",
        "decision": "매수" if stop_ok else "보류",
        "note": "" if stop_ok else "손절가 계산 불가로 수량 미산정 — 매수 보류",
        "score": event.get("score", 0),
        "grade": event.get("grade") or "",
        # 설명(core/explain.py)용 — 실제 우선순위 점수 구성 요소는 차수와 무관하게 항상 쓰인다.
        "vol_ratio": round(row.get("vol_ratio"), 1) if not pd.isna(row.get("vol_ratio")) else None,
        "cloud_thickness_pct": (
            round(t, 1) if not pd.isna(t := filt.cloud_thickness_pct(row.get("cloud_top"), row.get("cloud_bot"), row.get("close"))) else None
        ),
        "bb_width_pct": round(row.get("bb_width_pct"), 4) if not pd.isna(row.get("bb_width_pct")) else None,
        # "왜?" ①·③용 오늘 실제 지표 값 (표시 전용, core.screening_view.buy_facts — 판정에는 쓰지 않는다)
        "close": round(float(row["close"]), 2) if not pd.isna(row.get("close")) else None,
        "facts": sview.buy_facts(
            df, date, event["kind"], states_after, cfg, earnings_date=earnings_date, future_trading_days=future_trading_days
        ),
    }

    if event["kind"] == "A1":
        # 2차 기한 = A1 당일 포함 a1_to_a2_expiry_days번째 거래일(그날까지 A2 가능, 표시 전용).
        deadline = base["facts"].get("a2_deadline")
        base.update(
            {
                "rsi_prev": round(prev_rsi, 1) if not pd.isna(prev_rsi) else None,
                "rsi_now": round(row.get("rsi"), 1) if not pd.isna(row.get("rsi")) else None,
                "a2_expiry_date": deadline.strftime("%m/%d") if deadline is not None else None,
            }
        )
    elif event["kind"] == "A2":
        base.update(
            {
                "a1_date": str(pd.Timestamp(states_after.get("a1_date")).date()) if states_after.get("a1_date") is not None else "",
                "macd_norm": round(row.get("macd_norm"), 2) if not pd.isna(row.get("macd_norm")) else None,
                "rsi_now": round(row.get("rsi"), 1) if not pd.isna(row.get("rsi")) else None,
            }
        )
    elif event["kind"] == "A3":
        gap_pct = None
        prev_close = df.iloc[idx - 1]["close"] if idx > 0 else float("nan")
        if not pd.isna(row.get("open")) and not pd.isna(prev_close) and prev_close:
            gap_pct = round((row["open"] / prev_close - 1) * 100, 1)
        base.update(
            {
                "cloud_ok": bool(row.get("close") > row.get("cloud_top")) if not pd.isna(row.get("cloud_top")) else False,
                "future_yang_ok": bool(row.get("future_yang")) if not pd.isna(row.get("future_yang")) else False,
                "chikou_ok": bool(row.get("chikou_ok")) if not pd.isna(row.get("chikou_ok")) else False,
                "momentum_ok": bool(row.get("macd") > row.get("signal") and row.get("rsi") >= 50)
                if not (pd.isna(row.get("macd")) or pd.isna(row.get("signal")) or pd.isna(row.get("rsi")))
                else False,
                "gap_pct": gap_pct,
            }
        )
    else:  # B
        base.update(
            {
                "trend_ok": True,
                "macd_norm": round(row.get("macd_norm"), 2) if not pd.isna(row.get("macd_norm")) else None,
                "rsi_now": round(row.get("rsi"), 1) if not pd.isna(row.get("rsi")) else None,
            }
        )

    stage_note = _buy_stage_summary(event["kind"], base, earnings_date, date)
    base["note"] = " · ".join(part for part in (base["note"], stage_note) if part)
    # 계획금액·자금 계획 관련 문구가 섞이기 전의 순수 지표 근거 (단체방 공개용 —
    # notify.briefing.build_public_briefing_text / report_html.build_public_context가
    # 이 필드만 쓴다. base["note"]는 이후 live/paper 사이징에서 "계획 없음"·"남은 한도
    # 부족" 같은 개인 자금 관련 문구가 덧붙는다).
    base["condition_summary"] = stage_note
    return base


_MACRO_FRED_DEFS = [
    ("DGS10", "미국 10년물 국채금리", "%"),
    ("T10Y2Y", "장단기 금리차 (10년−2년)", "%p"),
    ("BAMLH0A0HYM2", "하이일드 스프레드", "%"),
    ("DFEDTARU", "미국 기준금리 (상단)", "%"),
    ("DEXKOUS", "원/달러 환율", "원"),
]
_MACRO_SLUG = {"DGS10": "dgs10", "T10Y2Y": "t10y2y", "BAMLH0A0HYM2": "hy", "DFEDTARU": "fed", "DEXKOUS": "fx"}
_MACRO_CODE_LABEL = {"DEXKOUS": "DEXKOUS · KRW=X"}  # DEXKOUS는 최근 며칠 KRW=X로 보완하므로 출처를 함께 표시


def _fred_missing_codes(macro_rows: list[dict], macro_warnings: list[str]) -> list[str]:
    """FRED 5개 지표 중 수집에 실패했거나(조회 실패·값 없음) 보고서 칸에서 빠진 코드 목록.

    입력: _build_macro_rows의 (rows, warnings). 출력: ["DGS10", ...] (정의 순서)
    """
    shown = {str(r.get("code", "")).split(" ")[0] for r in macro_rows}
    missing = []
    for code, _, _ in _MACRO_FRED_DEFS:
        failed = any(w.startswith(f"{code} 조회 실패") or w.startswith(f"{code}: 받은 값도") for w in macro_warnings)
        if failed or code not in shown:
            missing.append(code)
    return missing


def _trading_days_between_iso(a: str, b: str) -> list:
    from datetime import date as _date

    return list(trading_days_between(_date.fromisoformat(a), _date.fromisoformat(b)))


def _n_periods_ago(series: dict, latest_date: str, n: int) -> float | None:
    dates = sorted(d for d in series if d <= latest_date)
    idx = len(dates) - 1
    return series[dates[idx - n]] if idx - n >= 0 else None


def _build_macro_rows(cfg: dict, as_of_date) -> tuple[list[dict], list[str]]:
    """시장 온도 6칸(공포·탐욕 + FRED 5개) 데이터를 만든다 (P3.8, 표시 전용).

    판정(🟢안정·🟡주의·🔴위험)은 core.macro_status.classify_macro 한 곳에서, 기준값은
    config.yaml macro.thresholds에서 온다. 매매 신호에는 전혀 쓰지 않는다.

    입력: cfg(config.yaml의 macro 설정), as_of_date(보고서 기준일, date)
    출력: (rows, warnings). rows 각 항목: slug(앵커 id용), name, code, value, unit, as_of,
         change_1w, series(1년치, 스파크라인용), badge(classify_macro 결과 — status, symbol,
         label, text, zone, scale), is_stale, note(보조 설명), ref_values.
    실패해도 조용히 넘기지 않는다 — 못 받은 지표는 warnings에 남기고 칸에서 뺀다
    (호출부가 "지연"으로 표시).
    """
    macro_cfg = cfg.get("macro", {})
    if not macro_cfg.get("enabled", False):
        return [], []

    warnings: list[str] = []
    rows: list[dict] = []
    th = macro_cfg["thresholds"]
    n3m = macro_cfg.get("lookback_trading_days_3m", 63)
    stale_days = macro_cfg.get("stale_days", 5)
    report_date_iso = as_of_date.isoformat()

    # 1) 공포·탐욕 (CNN, 실패·3일 초과 지연 시 VIX 대체)
    fg_cfg = macro_cfg.get("fear_greed", {})
    fg = macrodata.get_fear_greed(as_of_date, stale_fallback_days=fg_cfg.get("fallback_after_days", 3))
    if fg["warning"]:
        warnings.append(fg["warning"])

    if fg["value"] is not None and not fg["use_vix_fallback"]:
        badge = macro_status.classify_macro("FEAR_GREED", fg["value"], th)
        rows.append({
            "slug": "fear-greed", "name": "공포·탐욕 지수", "code": "CNN Fear & Greed", "value": round(fg["value"]),
            "unit": "/100", "as_of": fg["as_of"], "change_1w": None, "series": fg["series_1y"],
            "badge": badge, "is_stale": fg["is_fallback"], "short_range": True,
            "note": f"지금 구간: {badge['zone']}",
            "ref_values": [th["FEAR_GREED"]["extreme_fear"], th["FEAR_GREED"]["fear"],
                           th["FEAR_GREED"]["neutral_high"], th["FEAR_GREED"]["greed"]],
        })
    else:
        if fg["use_vix_fallback"]:
            warnings.append("공포·탐욕 지수 3일 넘게 지연 - VIX(VIXCLS)로 대체")
        vix = macrodata.fetch_fred_indicator(fg_cfg.get("fallback", "VIXCLS"), as_of_date)
        if vix["warning"]:
            warnings.append(vix["warning"])
        series = vix["series"]
        if series:
            latest_date = max(series)
            value = series[latest_date]
            rows.append({
                "slug": "vix", "name": "변동성 지수 VIX (공포·탐욕 대체)", "code": "VIXCLS", "value": round(value, 2),
                "unit": "", "as_of": latest_date, "change_1w": None,
                "series": [v for _, v in sorted(series.items())],
                "badge": macro_status.classify_macro("VIXCLS", value, th),
                "is_stale": macro_status.is_stale(latest_date, report_date_iso, stale_days, _trading_days_between_iso),
                "note": "CNN 공포·탐욕 지수를 3일 넘게 못 받아 VIX로 대신 보여줘요",
                "ref_values": [th["VIXCLS"]["caution"], th["VIXCLS"]["danger"]],
            })

    # 2~6) FRED 지표
    for code, name, unit in _MACRO_FRED_DEFS:
        res = macrodata.fetch_fred_indicator(code, as_of_date)
        if res["warning"]:
            warnings.append(res["warning"])
        series = res["series"]

        if code == "DEXKOUS":
            # DEXKOUS는 보통 1주일 정도 늦게 갱신된다 — 그 뒤 며칠은 기존 KRW=X로
            # 보완한다(P3.8 1번 표). DEXKOUS가 있는 날짜는 그대로 두고, DEXKOUS에
            # 아직 없는 최근 날짜만 KRW=X로 채운다.
            try:
                recent_krwx = fx.fetch_usd_krw_range(as_of_date - timedelta(days=14), as_of_date)
            except Exception as exc:
                recent_krwx = {}
                warnings.append(f"DEXKOUS 최근 며칠 KRW=X 보완 실패: {exc}")
            if recent_krwx:
                series, _ = fx.merge_fx_with_fallback(series, recent_krwx)

        if not series:
            continue
        latest_date = max(series)
        value = series[latest_date]
        prev_3m = _n_periods_ago(series, latest_date, n3m)
        prev_1w = _n_periods_ago(series, latest_date, 5)

        value_before = None
        note = None
        ref_values: list[float] = []
        if code == "DFEDTARU":
            months = th["DFEDTARU"]["lookback_months"]
            value_before = macro_status.value_months_ago(series, latest_date, months)
            if value_before is not None:
                note = f"{months}개월 전 {value_before:g}% → 지금 {value:g}%"
        elif code == "DGS10":
            ref_values = [th["DGS10"]["caution"], th["DGS10"]["danger"]]
            if prev_3m is not None:
                note = f"3개월 변화 {value - prev_3m:+.2f}%p (보조 설명)"
        elif code == "T10Y2Y":
            ref_values = [th["T10Y2Y"]["stable"], th["T10Y2Y"]["danger"]]
        elif code == "BAMLH0A0HYM2":
            ref_values = [th["BAMLH0A0HYM2"]["caution"], th["BAMLH0A0HYM2"]["danger"]]
        elif code == "DEXKOUS":
            ref_values = [th["DEXKOUS"]["caution"], th["DEXKOUS"]["danger"]]
            note = "달러로 미국 주식을 사는 입장 기준"
        badge = macro_status.classify_macro(code, value, th, value_before=value_before)

        earliest_date = min(series)
        short_range = (pd.Timestamp(as_of_date) - pd.Timestamp(earliest_date)).days < 365
        if code == "BAMLH0A0HYM2" and short_range:
            warnings.append(f"{code}: 1년치를 못 받아 받은 만큼만 표시(기간 짧음)")

        rows.append({
            "slug": _MACRO_SLUG[code], "name": name, "code": _MACRO_CODE_LABEL.get(code, code),
            "value": round(value, 2), "unit": unit,
            "as_of": latest_date, "change_1w": round(value - prev_1w, 2) if prev_1w is not None else None,
            "series": [v for _, v in sorted(series.items())], "badge": badge,
            "is_stale": macro_status.is_stale(latest_date, report_date_iso, stale_days, _trading_days_between_iso),
            "short_range": short_range, "prev_3m": round(prev_3m, 2) if prev_3m is not None else None,
            "note": note, "ref_values": ref_values,
        })
    return rows, warnings


def _empty_run_summary(
    mode: str,
    as_of_date,
    warnings: list[str],
    *,
    stale: bool = False,
    expected_date=None,
    actual_date=None,
    skipped: bool = False,
) -> dict:
    """상태 전이 없이 끝내는 실행(P3.2 2·3번: 데이터 지연 모드 / 이미 처리된 기준일)의
    요약을 만든다. 나머지 항목은 모두 "신호 없음"으로 채운다."""
    return {
        "mode": mode,
        "mode_label": _MODE_LABEL[mode],
        "as_of": pd.Timestamp(as_of_date) if as_of_date is not None else None,
        "stale": stale,
        "skipped": skipped,
        "expected_date": expected_date.isoformat() if expected_date else None,
        "actual_date": actual_date.isoformat() if actual_date else None,
        "replay_needed": False,
        "buy_groups": {"b1": [], "b2": [], "b3": [], "b9": []},
        "buy_count": 0,
        "filtered_rows": [],
        "sell_rows": [],
        "warn_rows": [],
        "data_status_rows": [],
        "pending_order_rows": [],
        "unfilled_rows": [],
        "hold_rows": [],
        "stop_alerts": [],
        "watch_rows": [],
        "stage_counts": {},
        "held_tickers_count": 0,
        "max_concurrent": 0,
        "warnings": warnings,
        "data_gap_tickers": [],
        "earnings_unknown_count": 0,
        "all_events": [],
        "funnel": {},
        "funding_plan": None,
        "buy_risk_sum_krw": 0,
        "buy_risk_pct": None,
        "macro_rows": [],
        "plan_by_ticker": {},
        "live_judgment_rows": [],
        "fx_rate": None,
        "fx_date": None,
        "fx_is_fallback": False,
    }


def build_report_summary(
    mode: str,
    cfg: dict,
    indicator_map: dict,
    name_map: dict,
    earnings_map: dict,
    positions: dict,
    today_events: list[dict],
    as_of_by_ticker: dict,
    data_gap_tickers: list[str],
    fills_result,
    run_warnings: list[str],
    max_concurrent: int,
    fx_result,
    replay_needed: bool = False,
    plan_by_ticker: dict[str, float] | None = None,
    live_judgment_rows: list[dict] | None = None,
    plan_ref_price_by_ticker: dict[str, float] | None = None,
    default_budget_krw: float | None = None,
    remaining_cash_krw: float | None = None,
    recent_events: list[dict] | None = None,
    future_trading_days=None,
) -> dict:
    """오늘 이벤트·현재 상태·시세로 보고서·텔레그램용 summary dict를 만든다 (DB에 쓰지 않는다).

    run()의 라이브 처리 경로와 scripts/regenerate_report.py(상태를 다시 계산하지
    않고 현재 상태·저장된 이벤트로 보고서만 다시 만들 때) 양쪽이 같은 로직을
    쓰도록 여기 하나로 모았다 — 둘이 갈라지면 재생성한 보고서가 실제 라이브
    보고서와 달라질 수 있다.

    입력: mode, cfg, indicator_map({ticker: df}), name_map, earnings_map,
         positions(현재 상태 — run()이면 방금 계산한 states, 재생성이면
         db.load_all_positions), today_events(오늘 발생한 이벤트 —
         run()이면 sim["today_events"], 재생성이면 db에 저장된 이벤트),
         as_of_by_ticker, data_gap_tickers, fills_result(data.fills.load_fills
         결과), run_warnings, max_concurrent, fx_result(data.fx.get_usd_krw_rate
         결과 — 오늘 자금 계획·원화 표기에 쓴다), replay_needed, plan_by_ticker
         (live 전용 — {ticker: 계획금액(원)}, docs/design/live_advisor.md 2번),
         live_judgment_rows(live 전용 — compute_live_judgments 결과),
         plan_ref_price_by_ticker(live 전용 — {ticker: 계획 시트 "기준가($)"}, 있으면
         core.sizing.plan_tranche_qty가 기준가 기반 고정 총 주수 방식을 쓴다),
         default_budget_krw(live 전용 — 투자현황 "종목당 계획금액", 계획 없는 추천 줄의 기본 금액.
         None이면 계획 없는 줄은 수량 0, docs/design/auto_plan.md v2 규칙 B),
         remaining_cash_krw(live 전용 — 총 투자금 − 투자 원금. None이면 남은 현금 경고 생략, 규칙 C),
         recent_events(최근 a1_to_a2_expiry_days 거래일의 이벤트 — "오늘의 스크리닝" 2차 레인의
         "2차 대상 아님" 참고 목록용, 표시 전용. None이면 그 목록만 빈다),
         future_trading_days(오늘 다음 NYSE 거래일 목록 — 2차 기한 표시용. None이면 주말만 빼는 근사)

    mode == "live"이면 계좌 총액 기반 자금 계획(core.sizing.size_buy_signals)을 전혀
    쓰지 않고 종목별 계획금액만으로 수량을 정한다(docs/design/live_advisor.md 0번) —
    funding_plan은 항상 None. mode == "paper"면 이 부분은 이전과 완전히 같다(engine/
    daily.py 모듈 설명의 P5-1 0번 예외 — 일부러 live/paper 수량 공식을 다르게 둔다).
    출력: summary dict ("report_path" 제외 — render_report 호출은 호출부 몫)
    """
    fills_errors = fills_result.errors
    as_of = max(as_of_by_ticker.values()) if as_of_by_ticker else None
    fx_rate = fx_result.rate if fx_result else None

    # ── 오늘의 스크리닝 (표시 전용): 차수별 깔때기 — core.screening_view (순수 함수) ──
    screening = sview.build_screening_view(
        indicator_map, as_of_by_ticker, positions, today_events, cfg,
        name_map=name_map, recent_events=recent_events, future_trading_days=future_trading_days,
    )

    # ── 오늘 매수 신호: 지정가·손절가·탭별 조건 열을 붙인다 (수량은 아직 안 붙인다) ──
    buy_groups: dict[str, list] = {"b1": [], "b2": [], "b3": [], "b9": []}
    earnings_unknown_count = sum(1 for d in earnings_map.values() if d is None)
    for event in today_events:
        if event["kind"] not in _BUY_KINDS:
            continue
        ticker = event["ticker"]
        df = indicator_map[ticker]
        row = _build_buy_row(event, df, positions[ticker], cfg, name_map, earnings_map, future_trading_days)
        lane_facts = screening["facts_by_key"].get(row["key"])
        row["funnel"] = {"steps": lane_facts["steps"], "rank": lane_facts["rank"]} if lane_facts else None
        buy_groups[row["bucket"]].append(row)
    for bucket in buy_groups:
        buy_groups[bucket].sort(key=lambda r: r["score"], reverse=True)
    buy_count = sum(len(v) for v in buy_groups.values())
    all_buy_rows = [r for stage_rows in buy_groups.values() for r in stage_rows]

    if mode == "live":
        # ── 계획 기반 사이징 (docs/design/live_advisor.md 0·2·6번): 계좌 총액·
        # allocate_remaining_limit류 "남은 한도 배분" 없이, 종목별 계획금액만으로
        # 독립적으로 차수·수량을 정한다. 계획 없는 종목은 1차 진입가·손절가·조건은
        # 그대로 보이고(위에서 이미 buy_groups에 다 들어감) 금액만 비운다.
        plan_by_ticker = plan_by_ticker or {}
        plan_ref_price_by_ticker = plan_ref_price_by_ticker or {}
        for r in all_buy_rows:
            has_plan = r["ticker"] in plan_by_ticker
            r["has_plan"] = has_plan
            r["sizing"] = {"mode": "live", "fx_rate": fx_rate, "reduced": []}  # "왜?" ③ 수량 계산식 (표시 전용)
            if has_plan and r["stop"] is not None and fx_rate:
                sized_row = sizing.plan_tranche_qty(
                    plan_by_ticker[r["ticker"]], r["stage"], r["limit"], fx_rate,
                    ref_price=plan_ref_price_by_ticker.get(r["ticker"]),
                )
                held_qty = sum(q for q in positions.get(r["ticker"], {}).get("units", {}).values() if q > 0)
                clamped = sizing.clamp_plan_qty_to_remaining(sized_row["qty"], sized_row.get("total_shares"), held_qty)
                r["qty"] = clamped["qty"]
                r["amount_krw"] = round(sized_row["tranche_krw"]) if clamped["qty"] else 0
                r["sizing"].update(
                    budget_krw=plan_by_ticker[r["ticker"]], budget_source="계획금액", tranche_krw=sized_row["tranche_krw"],
                    ref_price=plan_ref_price_by_ticker.get(r["ticker"]), total_shares=sized_row.get("total_shares"),
                    raw_qty=sized_row["qty"],
                )
                if clamped["over_limit"]:
                    r["note"] = " · ".join(p for p in (r["note"], "계획 한도 초과") if p)
                    r["sizing"]["reduced"].append(f"이미 보유한 {held_qty}주가 계획 총 주수 이상(계획 한도 초과)")
                elif sized_row["qty"] == 0:
                    r["note"] = " · ".join(p for p in (r["note"], "계획금액으로 1주 미만") if p)
                    r["sizing"]["reduced"].append("계획금액의 이번 차수 몫으로는 1주를 못 삼")
                elif clamped["qty"] < sized_row["qty"]:
                    r["sizing"]["reduced"].append(f"계획 총 주수에서 이미 보유한 {held_qty}주를 빼고 남은 만큼만")
            elif not has_plan and default_budget_krw and r["stop"] is not None and fx_rate:
                # 계획 없는 추천: 투자현황 "종목당 계획금액"으로 수량만 보여준다(시트에 안 씀, 규칙 B).
                # 기준가 없는 방식(오늘 지정가) — 판정 표(plan_by_ticker)에는 넣지 않는다.
                sized_row = sizing.plan_tranche_qty(default_budget_krw, r["stage"], r["limit"], fx_rate)
                r["qty"] = sized_row["qty"]
                r["amount_krw"] = round(sized_row["tranche_krw"]) if sized_row["qty"] else 0
                r["default_budget"] = True
                notes = ["기본 금액"] + ([] if sized_row["qty"] else ["기본 금액으로 1주 미만"])
                r["note"] = " · ".join(p for p in (r["note"], *notes) if p)
                r["sizing"].update(
                    budget_krw=default_budget_krw, budget_source="기본 금액(투자현황 종목당 계획금액)",
                    tranche_krw=sized_row["tranche_krw"], raw_qty=sized_row["qty"],
                )
                if not sized_row["qty"]:
                    r["sizing"]["reduced"].append("기본 금액의 이번 차수 몫으로는 1주를 못 삼")
            else:
                r["qty"] = 0
                r["amount_krw"] = 0
                if not has_plan:
                    r["note"] = " · ".join(p for p in (r["note"], "계획 없음") if p)
            r["max_loss_krw"] = 0  # 위험 예산(2% 룰) 상한 없음 — 계획금액 자체가 위험 한도
            r["risk_capped"] = False
            r["limited"] = False
            r["explain"] = expl.explain_buy(r["stage"], r, cfg) if r["stop"] is not None else None
        if remaining_cash_krw is not None and all_buy_rows:
            # 남은 현금 경고 (docs/design/auto_plan.md v2 규칙 C)
            cash = auto_plan.cash_shortfall([r["amount_krw"] for r in all_buy_rows], remaining_cash_krw)
            for r, over in zip(all_buy_rows, cash["row_exceeds"]):
                if over:
                    r["note"] = " · ".join(p for p in (r["note"], "현금 부족") if p)
            if cash["total_exceeds"]:
                run_warnings.append(
                    f"현금 부족: 오늘 추천 투입금액 합계 {cash['total_krw']:,.0f}원이 남은 현금 {remaining_cash_krw:,.0f}원보다 큽니다"
                )
        funding_plan = None
        buy_risk_sum_krw = 0
    else:
        # ── 자금 계획 (P3.6 6-2·6-3번 / P5-1 0번): core.sizing.size_buy_signals 하나로
        # 오늘의 모든 매수 신호 수량을 한 번에 정한다 — 라이브 보고서·paper 가상 체결·
        # 백테스트가 항상 같은 수량을 내게 하기 위해서다(engine/daily.py 모듈 설명 참고).
        # positions는 종목마다 기준일이 다를 수 있어(재생성 등) as_of_by_ticker로 종목별 날짜를 쓴다
        # (_held_signals_snapshot은 날짜 하나만 받아 simulate_since 안에서만 쓴다).
        held = []
        for ticker, state_ in positions.items():
            qty_held = sum(q for q in state_["units"].values() if q > 0)
            if qty_held <= 0:
                continue
            date = as_of_by_ticker.get(ticker)
            close = None
            if date is not None and ticker in indicator_map and date in indicator_map[ticker].index:
                c = indicator_map[ticker].loc[date, "close"]
                close = float(c) if not pd.isna(c) else None
            state_label = state_["state"]
            if state_label == "주문대기" and state_.get("pending"):
                state_label = state_["pending"].get("prev_state", state_label)
            held.append({"ticker": ticker, "qty": qty_held, "close": close, "state_label": state_label})

        signals = [
            {
                "key": r["key"],
                "stage": r["stage"],
                "entry_price": r["limit"],
                "stop_price": r["stop"],
                "score": r["score"],
                "is_new_position": r["is_new_position"],
            }
            for r in all_buy_rows
        ]
        sized = sizing.size_buy_signals(signals, held, cfg, fx_rate)
        for r in all_buy_rows:
            s = sized["rows"].get(r["key"], {"qty": 0, "amount_krw": 0, "max_loss_krw": 0, "risk_capped": False, "limited": False})
            r["qty"] = s["qty"]
            r["amount_krw"] = s["amount_krw"]
            r["max_loss_krw"] = s["max_loss_krw"]
            r["risk_capped"] = s["risk_capped"]
            r["limited"] = s["limited"]
            r["sizing"] = {  # "왜?" ③ 수량 계산식 (표시 전용)
                "mode": "paper", "fx_rate": fx_rate, "slot_krw": sizing.slot_krw(cfg),
                "target_qty": s.get("target_qty"), "risk_cap_qty": s.get("risk_cap_qty"),
                "risk_pct": cfg["risk"][_STAGE_RISK_KEY[r["stage"]]],
                "reduced": (["손절이 멀어 위험 상한으로 줄임"] if s["risk_capped"] and s["qty"] > 0 else [])
                + (["오늘 남은 전략 한도가 부족"] if s["limited"] else []),
            }
            extra_notes = []
            if s["risk_capped"] and s["qty"] > 0:
                extra_notes.append("손절이 멀어 수량 축소")
            if s["limited"]:
                extra_notes.append("남은 한도 부족" if s["qty"] > 0 else "남은 한도 부족 — 매수 보류")
            if extra_notes:
                r["note"] = " · ".join(p for p in (r["note"], *extra_notes) if p)
            r["explain"] = expl.explain_buy(r["stage"], r, cfg) if r["stop"] is not None else None

        funding_plan = None
        buy_risk_sum_krw = sum(r.get("max_loss_krw") or 0 for r in all_buy_rows)
        if sized["funding_plan"] is not None:
            total_krw = cfg["account"]["total_krw"]
            new_spend_krw = sum(r.get("amount_krw") or 0 for r in all_buy_rows)
            qqqm_target_krw = total_krw * (1 - cfg["plan"]["cash_buffer_pct"] / 100) - sized["funding_plan"]["held_krw"] - new_spend_krw
            funding_plan = {
                **sized["funding_plan"],
                "fx_rate": fx_rate,
                "fx_date": fx_result.rate_date if fx_result else None,
                "fx_is_fallback": bool(fx_result and fx_result.is_fallback),
                "total_krw": total_krw,
                "qqqm_target_krw": qqqm_target_krw,
            }

    # ── 오늘 걸러진 신호 (매매 금지·동시 보유 한도) ────────────────────────────
    filtered_rows = []
    for event in today_events:
        if event["kind"] != "BLOCKED":
            continue
        ticker = event["ticker"]
        df = indicator_map[ticker]
        date = event["date"]
        row = df.loc[date] if date in df.index else None
        idx = df.index.get_loc(date) if date in df.index else None
        gc_count = filt.macd_cross_count(df, date) if (event["stage"] in ("A2", "B") and date in df.index) else None
        gap_pct = None
        if event["stage"] == "A3" and row is not None and idx:
            prev_close = df.iloc[idx - 1]["close"]
            if not pd.isna(row.get("open")) and not pd.isna(prev_close) and prev_close:
                gap_pct = round((row["open"] / prev_close - 1) * 100, 1)
        explain_ctx = {
            "kr": name_map.get(ticker, "") or ticker,
            "score": event.get("score", 0),
            "rsi_now": round(row.get("rsi"), 1) if row is not None and not pd.isna(row.get("rsi")) else None,
            "gc_count_20d": gc_count,
            "gap_pct": gap_pct,
            "earnings_date": earnings_map[ticker].isoformat() if earnings_map.get(ticker) else None,
            "max_concurrent": max_concurrent,
        }
        filtered_rows.append(
            {
                "티커": ticker,
                "종목명": name_map.get(ticker, "") or ticker,
                "단계": _STAGE_LABEL.get(event["stage"], event["stage"]),
                "유형": "동시보유한도" if event["blocked_type"] == "limit" else "매매금지",
                "사유": ", ".join(_label_filter_reason(r) for r in event["reasons"]),
                "점수": event.get("score", 0),
                "explain": expl.explain_filtered(event["reasons"], explain_ctx, cfg),
            }
        )
    filtered_rows.sort(key=lambda r: r["점수"], reverse=True)

    # ── 오늘 매도·손절 신호 ─────────────────────────────────────────────
    sell_rows = []
    for event in today_events:
        if event["kind"] not in _SELL_KINDS:
            continue
        ticker = event["ticker"]
        df = indicator_map[ticker]
        date = event["date"]
        close = float(df.loc[date, "close"]) if date in df.index and not pd.isna(df.loc[date, "close"]) else None
        entry_price = event.get("entry_price")
        pnl_pct = round((close - entry_price) / entry_price * 100, 1) if (close and entry_price) else None
        pnl_krw = (
            round((close - entry_price) * event["qty"] * fx_rate)
            if (close is not None and entry_price is not None and fx_rate)
            else None
        )
        row = df.loc[date] if date in df.index else None
        idx = df.index.get_loc(date) if date in df.index else None
        prev_rsi = df.iloc[idx - 1]["rsi"] if (idx is not None and idx > 0) else float("nan")
        explain_ctx = {
            "kr": name_map.get(ticker, "") or ticker,
            "close": close,
            "stop": event.get("stop_price"),
            "macd": round(row.get("macd"), 2) if row is not None and not pd.isna(row.get("macd")) else None,
            "signal": round(row.get("signal"), 2) if row is not None and not pd.isna(row.get("signal")) else None,
            "rsi_prev": round(prev_rsi, 1) if not pd.isna(prev_rsi) else None,
            "rsi_now": round(row.get("rsi"), 1) if row is not None and not pd.isna(row.get("rsi")) else None,
            "cloud_bot": round(row.get("cloud_bot"), 2) if row is not None and not pd.isna(row.get("cloud_bot")) else None,
            "chikou_broken": bool(row.get("chikou_broken")) if row is not None and not pd.isna(row.get("chikou_broken")) else False,
        }
        sell_rows.append(
            {
                "티커": ticker,
                "종목명": name_map.get(ticker, "") or ticker,
                "kind": event["kind"],
                "신호": _SELL_REASON[event["kind"]],
                "매도범위": _SELL_RANGE_LABEL[event["kind"]],
                "수량": event["qty"],
                "평균단가": round(entry_price, 2) if entry_price is not None else None,
                "종가": round(close, 2) if close is not None else None,
                "예상손익_krw": pnl_krw,
                "손익률": pnl_pct,
                "주문안내": _order_guidance(event["kind"], event["qty"], event.get("stop_price"), cfg),
                "비고": "",
                "explain": expl.explain_sell(event["kind"], explain_ctx, cfg),
            }
        )

    # ── 경고 (보유 종목만, 매도 아님) + 미체결(live) ────────────────────────
    warn_rows = []
    for ticker, df in indicator_map.items():
        state_ = positions[ticker]
        if not any(q > 0 for q in state_["units"].values()):
            continue
        date = as_of_by_ticker.get(ticker)
        if date is None:
            continue
        row = df.loc[date]
        idx = df.index.get_loc(date)
        prev_rsi = df.iloc[idx - 1]["rsi"] if idx > 0 else float("nan")
        name_kr = name_map.get(ticker, "") or ticker

        if sig.kijun_breach(row.get("close"), row.get("kijun")):
            ctx = {"kr": name_kr, "close": row.get("close"), "kijun": row.get("kijun")}
            warn_rows.append({"티커": ticker, "종목명": name_kr, "내용": "기준선 이탈 (매도 아님)", "badge_class": "b-info", "explain": expl.explain_warn("KIJUN_BREACH", ctx, cfg)})
        if sig.rsi_overheat_relief(prev_rsi, row.get("rsi")):
            ctx = {"kr": name_kr, "rsi_prev": round(prev_rsi, 1) if not pd.isna(prev_rsi) else None, "rsi_now": round(row.get("rsi"), 1) if not pd.isna(row.get("rsi")) else None}
            warn_rows.append({"티커": ticker, "종목명": name_kr, "내용": "RSI 과열 해소 (매도 아님)", "badge_class": "b-info", "explain": expl.explain_warn("RSI_RELIEF", ctx, cfg)})
        avg_entry = _average_entry_price(state_)
        if avg_entry is not None and sig.target_reached(avg_entry, row.get("close"), state_.get("stop")):
            ctx = {"kr": name_kr, "avg_entry": avg_entry, "stop": state_.get("stop"), "close": row.get("close")}
            warn_rows.append({"티커": ticker, "종목명": name_kr, "내용": "목표 도달 (손익비 2배, 매도 아님)", "badge_class": "b-info", "explain": expl.explain_warn("TARGET_REACHED", ctx, cfg)})

    # ── 손절 예약 알림 (P3.5 1번): 근접·변경·신규 3종 + 손절 발생 종목 최우선 ──────
    stop_near_pct = cfg["alerts"]["stop_near_pct"]
    stop_event_tickers = _stop_event_tickers(today_events)
    changed_by_ticker = _stop_changed_by_ticker(today_events)
    need_order_tickers = _need_stop_order_tickers(today_events)
    stop_alerts: list[dict] = []  # 텔레그램 "🛡️ 손절 예약" 줄용 (종목당 대표 알림 하나)
    hold_alert_by_ticker: dict[str, tuple[str, str]] = {}  # ticker -> (badge_class, text)

    # 신호 당일(오늘)은 "주문 후 체결 기록 필요" 안내만, 미체결 확정은 다음 날에만 (P3.1 보완 2번).
    pending_order_rows = _pending_order_rows(positions, name_map, mode)
    unfilled_rows = _unfilled_rows_from_events(today_events, name_map, mode)

    data_status_rows = []
    for ticker in data_gap_tickers:
        data_status_rows.append(
            {"티커": ticker, "종목명": name_map.get(ticker, "") or ticker, "내용": "data_gap - 신호 판정 제외"}
        )
    if earnings_unknown_count:
        data_status_rows.append({"티커": "", "종목명": "", "내용": f"실적일 확인불가 종목 {earnings_unknown_count}개"})
    for line in fills_errors:
        data_status_rows.append({"티커": "", "종목명": "", "내용": line})
    if fx_result is not None and fx_result.warning:
        data_status_rows.append({"티커": "", "종목명": "", "내용": f"환율: {fx_result.warning}"})

    # ── 보유 현황 / 내 보유 종목 (같은 데이터) ──────────────────────────────
    today_signal_by_ticker: dict[str, str] = {}
    for row in sell_rows:
        today_signal_by_ticker.setdefault(row["티커"], f"{row['매도범위']} 매도 ({row['신호']})")
    for row in warn_rows:
        if row["티커"]:
            today_signal_by_ticker.setdefault(row["티커"], row["내용"])

    hold_rows = []
    for ticker, state_ in positions.items():
        if not any(q > 0 for q in state_["units"].values()):
            continue
        df = indicator_map[ticker]
        date = as_of_by_ticker.get(ticker) or df.index[-1]
        close = float(df.loc[date, "close"]) if not pd.isna(df.loc[date, "close"]) else None
        avg_entry = _average_entry_price(state_)
        qty = sum(q for q in state_["units"].values() if q > 0)
        pnl_pct = round((close - avg_entry) / avg_entry * 100, 1) if (close and avg_entry) else None
        stop = state_.get("stop")
        stop_dist_pct = round((stop - close) / close * 100, 1) if (stop is not None and close) else None
        name_kr = name_map.get(ticker, "") or ticker

        # 손절 신호일에는 근접 알림을 보이지 않는다 (매도 탭에서 이미 다룬다, P3.5 1번).
        candidates: list[tuple[str, str, str]] = (
            []
            if ticker in stop_event_tickers
            else _stop_alert_candidates(
                close, stop, stop_near_pct, changed_by_ticker.get(ticker), ticker in need_order_tickers
            )
        )

        if ticker in stop_event_tickers:  # 방어적: 손절은 전량 매도라 보통 이 목록엔 이미 없다
            hold_alert_by_ticker[ticker] = _STOP_ALERT_STOP[1:]
        elif candidates:
            alert_type, badge_class, text = candidates[0]
            hold_alert_by_ticker[ticker] = (badge_class, text)
            stop_alerts.append({"티커": ticker, "종목명": name_kr, "type": alert_type, "text": text})
            for extra_type, extra_class, extra_text in candidates[1:]:
                extra_code = {"근접": "STOP_NEAR", "변경": "STOP_CHANGED", "신규": "STOP_NEEDED"}[extra_type]
                changed = changed_by_ticker.get(ticker)
                extra_ctx = {
                    "kr": name_kr, "close": close, "stop": stop, "stop_near_pct": stop_near_pct,
                    "old_stop": changed["old_stop"] if changed else None, "new_stop": changed["new_stop"] if changed else None,
                }
                warn_rows.append(
                    {"티커": ticker, "종목명": name_kr, "내용": extra_text, "badge_class": extra_class, "explain": expl.explain_warn(extra_code, extra_ctx, cfg)}
                )

        badge = hold_alert_by_ticker.get(ticker)
        value_krw = round(qty * close * fx_rate) if (close is not None and fx_rate) else None
        pnl_krw = round((close - avg_entry) * qty * fx_rate) if (close is not None and avg_entry is not None and fx_rate) else None
        hold_rows.append(
            {
                "티커": ticker,
                "종목명": name_kr,
                "단계": state_["state"],
                "수량": qty,
                "평균단가": round(avg_entry, 2) if avg_entry is not None else None,
                "종가": round(close, 2) if close is not None else None,
                "평가금액": round(qty * close, 2) if close is not None else None,
                "평가금액_krw": value_krw,
                "평가손익_krw": pnl_krw,
                "손익률": pnl_pct,
                "손절가": round(stop, 2) if stop is not None else None,
                "손절까지": stop_dist_pct,
                "오늘신호": today_signal_by_ticker.get(ticker, ""),
                "배지클래스": badge[0] if badge else None,
                "배지": badge[1] if badge else None,
            }
        )
    # ── 대기자금(QQQM) 요약 (P3.4 4번): 신호 판정에는 쓰지 않고 보유 표에 한 줄만 보여준다 ──
    cash_summary = summarize_cash_rows(fills_result.cash_rows)
    if cash_summary and cash_summary["qty"]:
        hold_rows.append(
            {
                "티커": "QQQM",
                "종목명": "QQQM (대기자금)",
                "단계": "대기자금",
                "수량": cash_summary["qty"],
                "평균단가": round(cash_summary["avg_price"], 2) if cash_summary["avg_price"] is not None else None,
                "종가": None,
                "평가금액": None,
                "손익률": None,
                "손절가": None,
                "손절까지": None,
                "오늘신호": "",
            }
        )

    hold_rows.sort(key=lambda r: r["티커"])

    # ── 관찰 목록: 다음 단계를 기다리는 종목 ────────────────────────────────
    watch_rows = []
    for ticker, state_ in positions.items():
        df = indicator_map[ticker]
        date = as_of_by_ticker.get(ticker)
        if date is None or date not in df.index:
            continue
        row = df.loc[date]
        name_kr = name_map.get(ticker, "") or ticker
        if state_["state"] == "정찰" and state_["units"].get("1", 0) > 0 and state_.get("a1_date") in df.index:
            bars_since = df.index.get_loc(date) - df.index.get_loc(state_["a1_date"])
            expiry_days = cfg["assumptions"]["a1_to_a2_expiry_days"]
            remaining = max(expiry_days - bars_since - 1, 0)
            # 2차 기한 = A1 당일 포함 expiry_days번째 거래일(그날까지 A2 가능) — 표시 전용
            deadline = sview.a2_deadline(df.index, state_["a1_date"], cfg, date, future_trading_days)
            watch_ctx = {
                "kr": name_kr,
                "macd_diff": round(row["macd"] - row["signal"], 2) if not (pd.isna(row.get("macd")) or pd.isna(row.get("signal"))) else None,
                "expiry_date": deadline.strftime("%m/%d") if deadline is not None else None,
            }
            watch_rows.append(
                {
                    "티커": ticker,
                    "종목명": name_kr,
                    "현재단계": "1차 (체결 시)" if state_["units"].get("1", 0) else "1차 (미체결)",
                    "기다리는신호": "2차 · MACD 골든크로스",
                    "남은거래일": remaining,
                    "explain": expl.explain_watch("WAIT_A2", watch_ctx, cfg),
                }
            )
        elif state_["state"] == "확인" and state_["units"].get("2", 0) > 0:
            watch_ctx = {
                "kr": name_kr,
                "cloud_ok": bool(row.get("close") > row.get("cloud_top")) if not pd.isna(row.get("cloud_top")) else False,
                "future_yang_ok": bool(row.get("future_yang")) if not pd.isna(row.get("future_yang")) else False,
                "chikou_ok": bool(row.get("chikou_ok")) if not pd.isna(row.get("chikou_ok")) else False,
                "rsi_now": round(row.get("rsi"), 1) if not pd.isna(row.get("rsi")) else None,
            }
            watch_rows.append(
                {
                    "티커": ticker,
                    "종목명": name_kr,
                    "현재단계": "2차 확인",
                    "기다리는신호": "3차 · 구름 4요소",
                    "남은거래일": None,
                    "explain": expl.explain_watch("WAIT_A3", watch_ctx, cfg),
                }
            )

    # ── 단계별 종목 수 ───────────────────────────────────────────────────
    stage_counts = {s: 0 for s in st.STATES}
    for state_ in positions.values():
        stage_counts[state_["state"]] = stage_counts.get(state_["state"], 0) + 1
    held_tickers_count = _held_count(positions)

    # ── 통과 현황(5단계 funnel): 순수 계산만 — 파일·DB 기록은 호출부(run()) 몫 ──
    funnel = _compute_funnel(today_events, indicator_map, as_of_by_ticker)

    return {
        "mode": mode,
        "mode_label": _MODE_LABEL[mode],
        "as_of": as_of,
        "replay_needed": replay_needed,
        "buy_groups": buy_groups,
        "buy_count": buy_count,
        "filtered_rows": filtered_rows,
        "sell_rows": sell_rows,
        "warn_rows": warn_rows,
        "data_status_rows": data_status_rows,
        "pending_order_rows": pending_order_rows,
        "unfilled_rows": unfilled_rows,
        "hold_rows": hold_rows,
        "stop_alerts": stop_alerts,
        "watch_rows": watch_rows,
        "stage_counts": stage_counts,
        "held_tickers_count": held_tickers_count,
        "max_concurrent": max_concurrent,
        "warnings": run_warnings,
        "data_gap_tickers": data_gap_tickers,
        "earnings_unknown_count": earnings_unknown_count,
        "all_events": today_events,
        "funnel": funnel,
        "screening": screening,
        "funding_plan": funding_plan,
        "buy_risk_sum_krw": buy_risk_sum_krw,
        "buy_risk_pct": (
            None
            if mode == "live"
            else ((buy_risk_sum_krw / cfg["account"]["total_krw"] * 100) if cfg["account"].get("total_krw") else None)
        ),
        "plan_by_ticker": plan_by_ticker or {},
        "live_judgment_rows": live_judgment_rows or [],
        # funding_plan과 별개로 항상 채운다 — live는 funding_plan이 없어도(mode=="live")
        # 보고서가 오늘 환율을 그대로 보여줘야 한다(docs/design/live_advisor.md 6·7번).
        "fx_rate": fx_rate,
        "fx_date": fx_result.rate_date if fx_result else None,
        "fx_is_fallback": bool(fx_result and fx_result.is_fallback),
    }


def _screening_inputs(conn, indicator_map: dict, as_of_by_ticker: dict, today_events: list[dict], cfg: dict):
    """"오늘의 스크리닝" 표시용 입력을 DB·거래소 달력에서 읽는다 (계산은 core.screening_view).

    출력: (recent_events, future_trading_days)
      recent_events: 오늘 이전 (a1_to_a2_expiry_days − 1)거래일의 DB 이벤트 + 오늘 이벤트(sim 결과).
        모드별 DB(paper는 paper DB)에서 읽는다 — 2차 레인의 "2차 대상 아님" 참고 목록용.
      future_trading_days: 오늘 다음 NYSE 거래일(약 6주) — 2차 기한 표시용.
    """
    as_of = max(as_of_by_ticker.values()) if as_of_by_ticker else None
    if as_of is None:
        return list(today_events), None
    window = cfg["assumptions"]["a1_to_a2_expiry_days"]
    master = sorted(set().union(*(df.index for df in indicator_map.values())))
    past_days = [d for d in master if d < as_of][-(window - 1):] if window > 1 else []
    recent: list[dict] = []
    for d in past_days:
        recent.extend(db.get_events_for_date(conn, str(pd.Timestamp(d).date())))
    recent.extend(today_events)
    start = (pd.Timestamp(as_of) + pd.Timedelta(days=1)).date()
    future = list(trading_days_between(start, start + timedelta(days=45)))
    return recent, future


def run(cfg: dict, mode: str, do_replay: bool, dry_run: bool, sheets_client=None) -> dict:
    """엔진을 한 번 실행한다. 결과 요약 dict를 반환한다 (완료 보고·보고서·텔레그램용).

    sheets_client: live 모드에서 data.sheets.read_sheets에 그대로 넘기는 테스트용
    주입 값(기본 None → 실제 인증). 계획·체결 입력 소스는 live만 다음 우선순위:
    1) 구글 시트(GOOGLE_SERVICE_ACCOUNT_JSON·GOOGLE_SHEETS_ID가 있으면),
    2) 없으면(SheetsConfigError) 로컬 fills.xlsx로 조용히 폴백(경고만 남김) —
    로컬 개발·기존 테스트가 구글 인증 없이 그대로 돌게 하기 위해서다. 시트는
    설정돼 있는데 읽다가 실패하면(네트워크·인증 오류 등) 조용히 넘기지 않고
    그대로 실패시킨다(CLAUDE.md 보안·네트워크 실패 원칙) — GitHub Actions가
    실패 알림을 보낸다."""
    print(f"[모드: {_MODE_LABEL[mode]} ({mode})]")
    print(f"[daily] {telegram.env_status()}")  # 값은 절대 출력하지 않는다 (CLAUDE.md 보안, P3.2 0번)
    print("나스닥 100 구성 종목 목록을 가져오는 중...")
    universe = get_universe()
    name_map_df = universe.set_index("ticker")[["name_kr"]]
    name_map = name_map_df["name_kr"].to_dict()

    print("일봉 시세를 받는 중... (캐시가 있으면 재사용)")
    price_result = fetch_universe_prices(universe["ticker"].tolist(), cfg)
    print(f"  성공 {len(price_result.prices)}종목 / 실패 {len(price_result.failed)}종목")

    print("지표를 계산하는 중...")
    indicator_map = {t: compute_indicators(df, cfg) for t, df in price_result.prices.items()}

    # ── 데이터 지연 모드 (P3.2 2번): 이번 기준일이 실행 시각 기준 가장 최근에 마감된
    # 거래일(NYSE 캘린더)보다 오래됐으면 상태 전이·주문대기·이벤트 기록을 하지 않고
    # 지연 알림만 보낸다. --replay(백테스트 전용 경로)에는 적용하지 않는다.
    actual_as_of_ts = max((df.index[-1] for df in indicator_map.values()), default=None)
    actual_date = actual_as_of_ts.date() if actual_as_of_ts is not None else None
    if not do_replay and actual_as_of_ts is not None:
        now_et = datetime.now(US_EASTERN)
        expected_date = latest_closed_trading_day(now_et)
        if actual_date < expected_date:
            print(
                f"[daily] *** 데이터 지연: 기대 기준일 {expected_date.isoformat()}, "
                f"실제 {actual_date.isoformat()} — 오늘은 매매 신호 없음 ***"
            )
            macro_rows, macro_warnings = _build_macro_rows(cfg, actual_date)
            summary = _empty_run_summary(
                mode,
                actual_date,
                [f"데이터 지연: 기대 기준일 {expected_date.isoformat()}, 실제 {actual_date.isoformat()}. 오늘은 매매 신호 없음"]
                + macro_warnings,
                stale=True,
                expected_date=expected_date,
                actual_date=actual_date,
            )
            summary["macro_rows"] = macro_rows
            summary["report_path"] = report_html.render_report(summary, cfg, OUTPUT_DIR)
            return summary

    print("실적 발표일을 확인하는 중...")
    earnings_map = get_earnings_dates(list(indicator_map.keys()))

    # ── 계획·체결 입력 (docs/design/live_advisor.md 1·2번): live는 구글 시트를 먼저
    # 시도하고, 인증 정보가 없으면(로컬 개발) fills.xlsx로 조용히 폴백한다. paper는
    # 지금처럼 fills.xlsx만 쓴다(계획 개념 자체가 paper에는 없다).
    plan_sheets_client = None  # 계획을 구글 시트에서 읽었을 때만 — 계획 자동 기록에 쓴다
    if mode == "live":
        try:
            sheets_result = sheets.read_sheets(client=sheets_client, cfg=cfg)
            fills_result = sheets_result.fills
            plan_df, plan_errors = sheets_result.plan_df, sheets_result.plan_errors
            plan_sheets_client = sheets_result.client
        except sheets.SheetsConfigError:
            print("[daily] 구글 시트 인증 정보가 없어 로컬 fills.xlsx를 대신 씁니다.")
            fills_result = load_fills()
            plan_df, plan_errors = load_plan()
    else:
        fills_result = load_fills()
        plan_df, plan_errors = pd.DataFrame(columns=["ticker", "budget_krw", "ref_price", "memo"]), []
    fills_df = fills_result.df
    fills_errors = fills_result.errors
    for line in fills_errors:
        print(f"  {line}")
    plan_by_ticker = _plan_budget_map(plan_df)
    plan_ref_price_by_ticker = _plan_ref_price_map(plan_df)
    for line in plan_errors:
        print(f"  {line}")

    conn = db.connect(db.db_path_for_mode(mode))
    max_concurrent = cfg["risk"]["max_concurrent_positions"]
    run_warnings: list[str] = list(fills_errors) + list(plan_errors)

    # ── 환율 (P3.6 6-4번): 기준일 종가 환율을 받는다. 못 받으면 직전 캐시 값 + 경고 ──
    if actual_date is not None:
        fx_result = fx.get_usd_krw_rate(actual_date)
    else:
        fx_result = fx.FxRateResult(None, None, True, "기준일을 확인할 수 없어 환율 조회를 생략했습니다")
    if fx_result.warning:
        run_warnings.append(fx_result.warning)
        print(f"[daily] {fx_result.warning}")

    gap_dates_by_ticker = {t: {pd.Timestamp(d) for d in price_result.data_gap.get(t, [])} for t in indicator_map}

    rebuild_from = None
    if do_replay:
        # ── 레거시 경로(P2): state.db가 비어 있을 때만 가상 체결로 되돌려 본다 ──
        positions = db.load_all_positions(conn)
        replay_needed = len(positions) == 0
        lookback_days = cfg["replay"]["lookback_days"]
        states = {
            t: positions.get(t) or st.init_state(t, name_map.get(t, "")) for t in indicator_map
        }
        per_ticker_dates = {t: _dates_to_process(df, replay_needed, lookback_days) for t, df in indicator_map.items()}
        fx_rate_by_date = fx.get_usd_krw_rate_map(sorted({d for dates in per_ticker_dates.values() for d in dates}))
        sim = simulate_since(
            indicator_map, per_ticker_dates, states, cfg, earnings_map, gap_dates_by_ticker, fills_df,
            virtual_fill=replay_needed, max_concurrent=max_concurrent, fx_rate_by_date=fx_rate_by_date,
        )
        events_to_persist = sim["all_events"]
    else:
        # ── P3 기본 경로: last_processed_date 다음 거래일부터 이번 기준일까지 모든
        # 거래일을 순서대로 다시 계산한다(P3.2 3번, "처리 날짜 누락 방지"). 이번
        # 기준일이 last_processed_date 이하이면(같은 날 재실행 등) 아무것도 하지 않는다.
        last_processed_str = db.get_meta(conn, "last_processed_date")
        next_start = (
            pd.Timestamp(last_processed_str) + pd.Timedelta(days=1)
            if last_processed_str is not None
            else pd.Timestamp(actual_date)  # 이 모드로 처음 실행 — 오늘부터 시작(과거로 replay하지 않는다)
        )

        # 라이브 체결 소급 반영 (fill_ledger = 지난 실행까지 반영한 체결): 이미 처리한 날짜로
        # 늦게 적은 체결이 있거나, 반영했던 체결이 시트에서 바뀌거나 지워졌으면 상태를
        # 체결 이력 처음(워밍업 포함)부터 다시 계산한다. 같은 기준일 재실행이어도 이때는 돈다.
        applied_keys: Counter = Counter()
        rebuild_from = None  # 소급 반영이 필요한 가장 이른 체결일
        if mode == "live":
            applied_keys = Counter(fill_key({**r, "price": r["price_usd"]}) for r in db.get_fill_ledger(conn, "live"))
            sheet_keys = Counter(fill_key(r) for r in fills_df.to_dict(orient="records")) if not fills_df.empty else Counter()
            header_broken = any("헤더 오류" in e for e in fills_result.errors)
            if header_broken:
                print("[daily] 체결 탭 헤더 오류 — 이번 실행은 체결 소급 반영 판단을 건너뜁니다(보유를 지우지 않음).")
            else:
                late = [k for k in (sheet_keys - applied_keys) if pd.Timestamp(k[0]) < next_start]
                removed = list(applied_keys - sheet_keys)
                affected = sorted({k[0] for k in late + removed})
                if affected:
                    rebuild_from = pd.Timestamp(affected[0])
                    print(
                        f"[daily] 체결 기록 소급 반영: 늦게 입력 {len(late)}건, 바뀜·삭제 {len(removed)}건 "
                        f"→ {rebuild_from.date()} 이후 상태를 다시 계산합니다."
                    )

        if rebuild_from is None and last_processed_str is not None and actual_date <= pd.Timestamp(last_processed_str).date():
            conn.close()
            reason = (
                f"기준일 {actual_date.isoformat()}은 이미 처리됨"
                f"(last_processed_date={last_processed_str}) — 이번 실행은 아무것도 하지 않습니다."
            )
            print(f"[daily] {reason}")
            return _empty_run_summary(mode, actual_date, [reason], skipped=True)

        # last_processed_date까지 저장된 실제 보유 상태를 이어받는다 — 아니면 이번
        # 증분 구간에서 새 이벤트가 없는 종목은 매번 대기로 되돌아가 버려서, 이미
        # 보유 중인 포지션(단계·수량·손절가)이 저장할 때마다 지워진다(발견된 버그).
        positions = db.load_all_positions(conn)
        if rebuild_from is not None:
            # 워밍업: 가장 이른 체결일(또는 소급일) N거래일 전부터 초기 상태로 다시 돈다 —
            # 신호일(주문대기)부터 재현돼야 신호 다음 날 체결이 원래 단계로 확정된다.
            earliest = min([rebuild_from] + ([pd.Timestamp(fills_df["date"].min())] if not fills_df.empty else []))
            warmup = int(cfg.get("replay", {}).get("live_rebuild_warmup_days", 60))
            master = sorted(set().union(*(df.index for df in indicator_map.values())))
            first_idx = next((i for i, d in enumerate(master) if d >= earliest), len(master) - 1)
            next_start = master[max(first_idx - warmup, 0)]
            persist_from = min(rebuild_from, pd.Timestamp(last_processed_str) + pd.Timedelta(days=1)) if last_processed_str else rebuild_from
            states = {t: st.init_state(t, name_map.get(t, "")) for t in indicator_map}
            for t, saved in positions.items():  # 판정 변경 감지용 직전 판정만 이어받는다
                if t in states and saved.get("last_judgment"):
                    states[t]["last_judgment"] = saved["last_judgment"]
            applied_keys = Counter()
        else:
            states = {t: positions.get(t) or st.init_state(t, name_map.get(t, "")) for t in indicator_map}
        per_ticker_dates = {t: _dates_since_start(df, next_start) for t, df in indicator_map.items()}
        fx_rate_by_date = fx.get_usd_krw_rate_map(sorted({d for dates in per_ticker_dates.values() for d in dates}))
        sim = simulate_since(
            indicator_map, per_ticker_dates, states, cfg, earnings_map, gap_dates_by_ticker, fills_df,
            virtual_fill=(mode == "paper"), max_concurrent=max_concurrent, fx_rate_by_date=fx_rate_by_date,
            applied_fill_keys=applied_keys,
        )
        replay_needed = False
        events_to_persist = sim["all_events"]  # 건너뛴 거래일이 있어도 모두 기록한다 (누락 방지, P3.2 3번)
        if rebuild_from is not None:
            # 워밍업 구간 이벤트는 기록하지 않고, 다시 계산한 구간은 지우고 새로 쓴다.
            events_to_persist = [e for e in events_to_persist if pd.Timestamp(e["date"]) >= persist_from]
            if not dry_run:
                db.delete_events_since(conn, str(persist_from.date()))
        if not dry_run:
            db.set_meta(conn, "last_processed_date", str(actual_date))
            if mode == "live":
                if rebuild_from is not None:
                    db.delete_fill_ledger(conn, "live")
                db.record_fill_ledger(conn, [_ledger_row(r) for r in sim["applied_fills"]])
        for r in sim["applied_fills"]:
            print(f"  [체결 반영] {pd.Timestamp(r['date']).date()} {r['ticker']} {r['side']} {r['qty']}주 @ {r['price']} (차수 {r['unit']})")

    states = sim["states"]
    today_events = sim["today_events"]
    run_warnings.extend(sim["warnings"])
    data_gap_tickers = sim["data_gap_tickers"]
    as_of_by_ticker = sim["as_of_by_ticker"]

    positions = states

    # ── 계획 탭 자동 기록 v2 (docs/design/auto_plan.md): 투자현황 탭을 읽고, 첫 매수 후 계획에
    # 없는 종목의 계획 줄을 시트에 쓴 뒤 메모리 계획에 합친다 → 아래 판정·사이징이 그 계획을 쓴다.
    # 계획 없는 추천은 "종목당 계획금액"으로 수량만 낸다(시트에 안 씀). 시트에서 읽지 않았으면
    # (로컬 폴백) 아무것도 안 한다. paper는 부르지 않는다.
    auto_plan_info = None
    default_budget_krw = None
    remaining_cash_krw = None
    if mode == "live" and plan_sheets_client is not None and auto_plan.auto_plan_settings(cfg)["enabled"]:
        portfolio, portfolio_warnings = sheets.read_portfolio(plan_sheets_client)
        default_budget_krw = portfolio["per_ticker_budget_krw"]
        auto_plan_warnings = list(portfolio_warnings)
        plan_df, write_warnings, auto_plan_info = apply_auto_plan(
            plan_df, fills_df, default_budget_krw, plan_sheets_client, write=not dry_run,
        )
        auto_plan_warnings += write_warnings
        if portfolio["total_krw"] is not None:
            principal_krw, missing_fx = auto_plan.invested_principal_krw(fills_df)
            remaining_cash_krw = portfolio["total_krw"] - principal_krw
            if missing_fx:
                auto_plan_warnings.append(f"투자 원금 계산: 환율이 없는 체결 {missing_fx}건을 뺐습니다")
            print(f"[daily] 남은 현금 {remaining_cash_krw:,.0f}원 (총 투자금 − 투자 원금 {principal_krw:,.0f}원)")
        run_warnings.extend(auto_plan_warnings)
        for line in auto_plan_warnings:
            print(f"[daily] {line}")
        if auto_plan_info["selected"]:
            print(f"[daily] 계획 자동 추가(첫 매수): 선택 {auto_plan_info['selected']}, 시트 기록 {auto_plan_info['written']}")
        plan_by_ticker = _plan_budget_map(plan_df)
        plan_ref_price_by_ticker = _plan_ref_price_map(plan_df)

    # ── 라이브 판정 (docs/design/live_advisor.md 3·7번): paper는 절대 부르지 않는다 —
    # states에 last_judgment를 남기지 않아야 한다(모듈 설명의 P5-1 0번 예외 참고).
    # 저장(db.save_position) 전에 states를 갱신해야 오늘 판정이 이번 실행에 저장된다.
    live_judgment_rows: list[dict] = []
    if mode == "live":
        live_judgment_rows = compute_live_judgments(
            states, today_events, indicator_map, as_of_by_ticker, name_map,
            plan_by_ticker, fills_df, fx_result.rate if fx_result else None, cfg,
            plan_ref_price_by_ticker=plan_ref_price_by_ticker,
        )

    for ticker, state_ in states.items():
        if not dry_run:
            db.save_position(conn, state_)

    if not dry_run:
        db.record_events(conn, events_to_persist)

    # ── 재현성 확인용 가격 스냅샷 (P2.1 보완 3번): dry-run에도 남긴다(진단 목적) ──
    run_at = datetime.now().isoformat(timespec="seconds")
    snapshot_rows = []
    for ticker, df in price_result.prices.items():
        last = df.iloc[-1]
        snapshot_rows.append(
            {
                "ticker": ticker,
                "date": str(df.index[-1].date()),
                "close": float(last["close"]) if pd.notna(last["close"]) else None,
                "close_source": last.get("close_source"),
                "meta_time": df.attrs.get("close_meta_time"),
            }
        )
    if not dry_run:
        db.record_price_snapshots(conn, run_at, snapshot_rows)

    print("시장 온도(공포·탐욕 + FRED 지표)를 받는 중...")
    macro_rows, macro_warnings = _build_macro_rows(cfg, actual_date) if actual_date is not None else ([], [])
    for line in macro_warnings:
        print(f"  [시장 온도] {line}")
    for row in macro_rows:  # 텔레그램과 같은 판정 문구 (예: "10년물 5.17% 🔴위험") — Actions 로그로 확인용
        print(f"  [시장 온도 값] {briefing._macro_item_text(row)} (기준일 {row.get('as_of')})")
    run_warnings.extend(macro_warnings)

    recent_events, future_trading_days = _screening_inputs(conn, indicator_map, as_of_by_ticker, today_events, cfg)
    summary = build_report_summary(
        mode, cfg, indicator_map, name_map, earnings_map, positions, today_events,
        as_of_by_ticker, data_gap_tickers, fills_result, run_warnings, max_concurrent,
        fx_result, replay_needed=replay_needed,
        plan_by_ticker=plan_by_ticker, live_judgment_rows=live_judgment_rows,
        plan_ref_price_by_ticker=plan_ref_price_by_ticker,
        default_budget_krw=default_budget_krw, remaining_cash_krw=remaining_cash_krw,
        recent_events=recent_events, future_trading_days=future_trading_days,
    )
    summary["macro_rows"] = macro_rows
    if auto_plan_info is not None:
        summary["auto_plan"] = auto_plan_info
    # 계획·체결 입력 오류(헤더 불일치, 잘못된 줄 등) — 텔레그램 요약에 한 줄 경고로 띄운다.
    summary["input_errors"] = list(fills_errors) + list(plan_errors)
    summary["fred_missing"] = _fred_missing_codes(macro_rows, macro_warnings) if cfg.get("macro", {}).get("enabled") else []
    # 체결 기록 변경으로 재계산한 실행은 같은 기준일에 이미 보냈어도 다시 보낸다(정정본).
    summary["rebuilt_from"] = str(rebuild_from.date()) if rebuild_from is not None else None
    as_of = summary["as_of"]
    funnel = summary["funnel"]

    # ── 통과 현황(5단계 funnel): 보고서에는 안 쓰고 CSV·events에만 남긴다 ───────
    if as_of is not None:
        _write_funnel(funnel, as_of, mode)
        if not dry_run:
            db.record_events(conn, [{"date": str(as_of.date()), "ticker": "", "kind": "FUNNEL", **funnel}])

    if not dry_run:
        db.record_run(
            conn,
            run_at=run_at,
            as_of_date=str(as_of.date()) if as_of is not None else "",
            ticker_count=len(indicator_map),
            warning_count=len(run_warnings),
        )
    conn.close()

    _write_outputs(summary, cfg)
    report_path = report_html.render_report(summary, cfg, OUTPUT_DIR)
    summary["report_path"] = report_path
    if mode == "live":
        # 단체방 공개용 보고서 (시장 온도 + 오늘의 추천만, 보유·수량·평단·손익·계획금액
        # 없음) — paper는 텔레그램을 아예 보내지 않는 하드 가드가 있어 만들지 않는다.
        summary["public_report_path"] = report_html.render_public_report(summary, cfg, OUTPUT_DIR)
    return summary


def _format_cell(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _to_markdown_table(rows: list[dict]) -> str:
    """의존성(tabulate) 없이 표를 마크다운으로 바꾼다."""
    if not rows:
        return "없음"
    columns = list(rows[0].keys())
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(c)) for c in columns) + " |")
    return "\n".join(lines)


def _write_funnel(funnel: dict, as_of, mode: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    as_of_str = as_of.date().isoformat()
    pd.DataFrame([{"단계": k, "건수": v} for k, v in funnel.items()]).to_csv(
        OUTPUT_DIR / f"funnel_{mode}_{as_of_str}.csv", index=False, encoding="utf-8-sig"
    )


def _write_outputs(summary: dict, cfg: dict | None = None) -> None:
    """summary -> outputs/signals_{모드}_YYYY-MM-DD(.md, _buy/_sell/_blocked.csv). 제목은 cfg report.short_title."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    as_of = summary["as_of"]
    as_of_str = as_of.date().isoformat() if as_of is not None else "알수없음"
    mode = summary.get("mode", "live")

    all_buy_rows = [r for rows in summary["buy_groups"].values() for r in rows]
    pd.DataFrame(all_buy_rows).to_csv(OUTPUT_DIR / f"signals_{mode}_{as_of_str}_buy.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary["sell_rows"]).to_csv(OUTPUT_DIR / f"signals_{mode}_{as_of_str}_sell.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary["filtered_rows"]).to_csv(
        OUTPUT_DIR / f"signals_{mode}_{as_of_str}_blocked.csv", index=False, encoding="utf-8-sig"
    )

    lines = [f"# {briefing.report_titles(cfg)['short_title']} 신호 — 기준일 {as_of_str} ({summary['mode_label']})", ""]
    lines.append(f"보유 종목 수: {summary['held_tickers_count']} / 동시 보유 한도: {summary['max_concurrent']}")
    lines.append("")
    lines.append("## 매수 신호")
    lines.append(_to_markdown_table(all_buy_rows))
    lines.append("")
    lines.append("## 걸러진 신호 (매매 금지·한도 초과)")
    lines.append(_to_markdown_table(summary["filtered_rows"]))
    lines.append("")
    lines.append("## 매도·손절 신호")
    lines.append(_to_markdown_table(summary["sell_rows"]))
    lines.append("")
    lines.append("## 경고")
    lines.append(_to_markdown_table(summary["warn_rows"]))
    lines.append("")
    lines.append("## 단계별 종목 수")
    for stage, count in summary["stage_counts"].items():
        lines.append(f"- {stage}: {count}")
    lines.append("")
    lines.append(f"data_gap 종목: {', '.join(summary['data_gap_tickers']) or '없음'}")
    lines.append(f"실적일 확인불가 종목 수: {summary['earnings_unknown_count']}")
    (OUTPUT_DIR / f"signals_{mode}_{as_of_str}.md").write_text("\n".join(lines), encoding="utf-8")


def _resend_last(cfg: dict, mode: str, force_no_send: bool) -> None:
    """--resend(P3.4 3번): 상태를 다시 처리하지 않고 마지막 기준일의 보고서·글을
    다시 보낸다. 중복 발송 방지 기록(store.db의 notifications)은 확인하지 않는다.

    모의(paper) 모드는 텔레그램을 아예 보내지 않으므로(notify.telegram의 하드 가드)
    재발송 대상이 될 수 없다 — --mode로 무엇을 넘기든 항상 실전(live) 기록·파일만
    다시 보낸다.
    """
    if mode != "live":
        print(f"[daily] --resend: {_MODE_LABEL.get(mode, mode)} 모드는 텔레그램을 보내지 않으므로, --resend는 항상 실전(live) 기록을 재발송합니다.")
    conn = db.connect(db.db_path_for_mode("live"))
    last_date = db.get_meta(conn, "last_processed_date")
    conn.close()
    if last_date is None:
        print(f"[daily] --resend: {_MODE_LABEL['live']} 모드에 처리된 기준일 기록이 없습니다.")
        return

    report_path = OUTPUT_DIR / f"report_live_{last_date}.html"
    text_path = OUTPUT_DIR / f"telegram_live_{last_date}.txt"
    print(f"[daily] --resend: 기준일 {last_date} 재발송")
    ok = telegram.resend_last(report_path, text_path, last_date, force_no_send=force_no_send, cfg=cfg)
    print(f"[daily] 재발송 {'성공' if ok else '실패'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="나스닥 100 MACD 스크리너 — 일일 신호 판정")
    parser.add_argument("--mode", choices=["live", "paper"], default=None, help="config.yaml의 mode보다 우선")
    parser.add_argument("--replay", action="store_true", help="레거시(P2): state.db가 비어 있으면 되돌려 보기로 상태를 만든다")
    parser.add_argument("--dry-run", action="store_true", help="DB에 쓰지 않고 결과만 출력한다")
    parser.add_argument("--no-send", action="store_true", help="보고서만 만들고 텔레그램은 보내지 않는다")
    parser.add_argument(
        "--resend", action="store_true",
        help="상태를 다시 처리하지 않고, 마지막 기준일의 보고서·글을 다시 보낸다(중복 발송 방지 기록 무시)",
    )
    args = parser.parse_args()

    cfg = load_config()
    mode = resolve_mode(cfg, args.mode)

    if args.resend:
        _resend_last(cfg, mode, force_no_send=args.no_send)
        return

    summary = run(cfg, mode, do_replay=args.replay, dry_run=args.dry_run)

    if summary.get("skipped"):  # P3.2 3번: 이번 기준일이 이미 처리됨 — 아무것도 하지 않는다
        print(f"\n[daily] {summary['warnings'][0]}")
        return

    as_of = summary["as_of"]
    print(f"\n기준일: {as_of.date().isoformat() if as_of is not None else '알수없음'} / 모드: {summary['mode_label']}")
    print(f"단계별 종목 수: {summary['stage_counts']} (보유 {summary['held_tickers_count']} / 한도 {summary['max_concurrent']})")
    print(f"오늘 매수 신호 {summary['buy_count']}건, 걸러진 신호 {len(summary['filtered_rows'])}건, 매도·손절 신호 {len(summary['sell_rows'])}건")
    print(f"경고 {len(summary['warn_rows'])}건, data_gap 종목 {len(summary['data_gap_tickers'])}개")
    held_names = sorted({r["티커"] for r in summary.get("hold_rows", [])})
    print(f"보유 종목: {', '.join(held_names) if held_names else '없음'}")
    if summary.get("rebuilt_from"):
        print(f"체결 기록 변경 반영: {summary['rebuilt_from']}부터 재계산")
    if summary.get("input_errors"):
        print(f"시트 입력 오류 {len(summary['input_errors'])}건")
    print(f"보고서: {summary['report_path']}")

    if not args.dry_run:
        if summary.get("stale"):  # P3.2 2번: 데이터 지연 모드 — 지연 배너 보고서 한 통만 보낸다(지연 문구는 첨부 설명)
            sent_path = telegram.send_delay_notice(summary, cfg, force_no_send=args.no_send)
        else:
            text = briefing.build_briefing_text(summary, cfg)
            sent_path = telegram.send_briefing(text, summary, cfg, force_no_send=args.no_send)

            # 단체방 공개 발송 (live 전용) — 실패해도 개인 발송·실행에는 영향 없이
            # 개인 채팅에 경고 한 줄만 남긴다.
            if mode == "live":
                public_text = briefing.build_public_briefing_text(summary, cfg)
                group_result = telegram.send_group_briefing(
                    public_text, summary.get("public_report_path"), summary, cfg, force_no_send=args.no_send
                )
                warning = telegram.group_send_warning_line(group_result)
                if warning:
                    print(f"[daily] {warning}")
                    telegram.notify_ops_error(warning)
        print(f"텔레그램 글: {sent_path}")


if __name__ == "__main__":
    main()
