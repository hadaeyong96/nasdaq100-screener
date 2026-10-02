"""계획 탭 자동 기록 v2 — 순수 함수 (docs/design/auto_plan.md).

네트워크·파일·DB·현재 시각에 접근하지 않는다(core/ 원칙). 체결 기록은 호출부가 읽어 넘긴다.
시트 읽기·쓰기는 data/sheets.py, 흐름 연결은 engine/daily.py 몫이다.

체결 기록(fills_df)은 data/fills.py가 대기자금(QQQM) 줄을 cash_rows로 이미 따로 뺀 신호 종목
체결만 받는다 — 그래서 여기서는 대기자금을 따로 거르지 않는다.
"""

from __future__ import annotations

import pandas as pd

AUTO_MEMO_PREFIX = "자동 추가"


def auto_plan_settings(cfg: dict) -> dict:
    """config.yaml live.auto_plan을 기본값과 합쳐 돌려준다.

    입력: cfg 전체
    출력: {"enabled": bool} (섹션이 없으면 enabled=False — 이 기능이 생기기 전과 같은 동작)
    """
    section = ((cfg.get("live") or {}).get("auto_plan")) or {}
    return {"enabled": bool(section.get("enabled", False))}


def build_first_buy_memo(first_buy_date) -> str:
    """자동 추가 메모 문자열.

    입력: first_buy_date(첫 매수일 — date·Timestamp·문자열)
    출력: "자동 추가 · 첫 매수 YYYY-MM-DD"
    """
    return f"{AUTO_MEMO_PREFIX} · 첫 매수 {pd.Timestamp(first_buy_date).date().isoformat()}"


def _records(fills_df: pd.DataFrame | None) -> list[dict]:
    if fills_df is None or fills_df.empty:
        return []
    return fills_df.to_dict(orient="records")


def held_qty_by_ticker(fills_df: pd.DataFrame | None) -> dict[str, float]:
    """체결 기록의 종목별 순보유 수량.

    입력: fills_df(date, ticker, side "buy"|"sell", qty, ...)
    출력: {티커(대문자): 매수 수량 합 − 매도 수량 합}
    """
    held: dict[str, float] = {}
    for r in _records(fills_df):
        ticker = str(r["ticker"]).strip().upper()
        sign = 1 if r["side"] == "buy" else -1
        held[ticker] = held.get(ticker, 0) + sign * r["qty"]
    return held


def select_first_buy_plan_rows(
    fills_df: pd.DataFrame | None, existing_tickers, budget_krw: float
) -> list[dict]:
    """첫 매수 후 계획 탭에 아직 없는 종목의 계획 줄을 만든다 (v2 규칙 D).

    대상: 매수 기록이 있고 순보유 수량 > 0이고 existing_tickers에 없는 종목.
    등록일·기준가는 그 종목의 첫 매수(날짜가 가장 이른 매수, 같은 날이면 먼저 적힌 줄)에서 가져온다.

    입력: fills_df(대기자금 줄을 뺀 체결 기록), existing_tickers(계획 탭에 이미 있는 티커들),
         budget_krw(투자현황 탭 "종목당 계획금액")
    출력: [{"ticker","budget_krw","reg_date"(YYYY-MM-DD),"ref_price","memo"}] — 티커 오름차순
    """
    existing = {str(t).strip().upper() for t in existing_tickers if str(t).strip()}
    held = held_qty_by_ticker(fills_df)
    first_buy: dict[str, dict] = {}
    for r in _records(fills_df):
        if r["side"] != "buy":
            continue
        ticker = str(r["ticker"]).strip().upper()
        if ticker not in first_buy or pd.Timestamp(r["date"]) < pd.Timestamp(first_buy[ticker]["date"]):
            first_buy[ticker] = r
    rows = []
    for ticker in sorted(first_buy):
        if ticker in existing or (held.get(ticker) or 0) <= 0:
            continue
        r = first_buy[ticker]
        rows.append(
            {
                "ticker": ticker,
                "budget_krw": float(budget_krw),
                "reg_date": pd.Timestamp(r["date"]).date().isoformat(),
                "ref_price": round(float(r["price"]), 2),
                "memo": build_first_buy_memo(r["date"]),
            }
        )
    return rows


def invested_principal_krw(fills_df: pd.DataFrame | None) -> tuple[float, int]:
    """투자 원금 = Σ(매수 수량×체결가×환율) − Σ(매도 수량×체결가×환율) (v2 규칙 C).

    입력: fills_df(대기자금 줄을 뺀 체결 기록, fx_rate는 빈 칸 채우기까지 끝난 값)
    출력: (투자 원금(원), 환율이 없어 뺀 체결 줄 수)
    """
    total = 0.0
    missing = 0
    for r in _records(fills_df):
        fx = r.get("fx_rate")
        if fx is None or pd.isna(fx):
            missing += 1
            continue
        sign = 1 if r["side"] == "buy" else -1
        total += sign * float(r["qty"]) * float(r["price"]) * float(fx)
    return total, missing


def cash_shortfall(amounts_krw: list[float], remaining_cash_krw: float) -> dict:
    """오늘 추천 투입금액이 남은 현금을 넘는지 판정한다 (v2 규칙 C).

    입력: amounts_krw(추천 줄별 투입금액, 원), remaining_cash_krw(총 투자금 − 투자 원금)
    출력: {"total_krw": 합계, "total_exceeds": 합계 > 남은 현금,
          "row_exceeds": [줄별 투입금액 > 남은 현금]}
    """
    total = float(sum(a or 0 for a in amounts_krw))
    return {
        "total_krw": total,
        "total_exceeds": total > remaining_cash_krw,
        "row_exceeds": [(a or 0) > remaining_cash_krw for a in amounts_krw],
    }


def merge_plan_rows(plan_df: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    """새로 추가한 계획 줄을 기존 계획 DataFrame에 합친다(메모리 병합, 같은 실행의 수량 계산용).

    입력: plan_df(ticker, budget_krw, ref_price, memo), new_rows(select_first_buy_plan_rows 결과 중 실제 추가한 줄)
    출력: 새 DataFrame. 이미 있는 티커는 기존 줄을 그대로 둔다(덮어쓰지 않음).
    """
    columns = ["ticker", "budget_krw", "ref_price", "memo"]
    existing = set(plan_df["ticker"]) if plan_df is not None and not plan_df.empty else set()
    add = [{c: r[c] for c in columns} for r in new_rows if r["ticker"] not in existing]
    if not add:
        return plan_df
    base = plan_df if plan_df is not None and not plan_df.empty else pd.DataFrame(columns=columns)
    return pd.concat([base, pd.DataFrame(add, columns=columns)], ignore_index=True)
