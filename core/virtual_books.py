"""가상 장부 5개 — 순수 계산 (네트워크·파일·DB·현재 시각 접근 금지).

실제 돈은 QQQM만 굴리고, 레버리지 ETF는 가상 장부로만 관찰한다 (docs/design/virtual_books.md).

장부 종류:
- 그냥 보유(hold): 시작일 종가에 전액 매수하고 계속 들고 간다.
- 200일선 전환(sma_switch): 신호 종목(QQQ) t일 종가 > 200일 단순이동평균이면 레버리지 ETF,
  같거나 아래면 방어 ETF(SGOV). t일 신호는 t+1 거래일 종가에 체결한다 — sealed-a B1(l1b)과
  같은 방식이고, 위/아래 판정은 core.l1b.above_sma를 그대로 쓴다. 시작일 첫 매수는 시작일
  전 거래일 신호로 정한다(sealed-a "2021-12-31 신호 → 2022-01-03 종가 체결"과 같은 규칙).

장부 규칙:
- 가격: 분할 반영 종가(auto_adjust=False Close). 운용보수는 ETF 가격에 이미 들어 있다.
- 체결: 정수 주식만, 남는 돈은 현금. 매수가 = 종가 × (1 + 슬리피지), 매도가 = 종가 × (1 − 슬리피지),
  수수료 = 체결금액 × 편도 수수료율. 매수 원가(취득가)는 수수료 포함.
- 분배금: 배당락일 직전 거래일 종가까지 들고 있던 주식 수 × 주당 분배금 × (1 − 원천징수율)을
  배당락일 현금에 더한다(재투자 안 함).
- 세금: 그해 마지막 거래일(전환 체결 뒤) 실현 순이익(손실과 상계, 이월결손 없음, 공제 무시) × 세율.
  장부 현금에서 빼지 않고 "확정 세금 누적"으로 따로 쌓는다(연말 가상 계산).
  "청산 가정 세후 추정" = 오늘 종가로 전량 팔았다고 보고(매도 수수료 포함) 올해 실현 + 미실현
  이익에 세금을 매긴 값 — 그냥 보유 장부처럼 실현이 없는 장부를 비교하기 위한 값이다.

미래 데이터 금지: t일 결과는 t일까지의 종가·분배금과 t−1일 신호만 쓴다(양수 shift·과거 인덱스만).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.l1b import above_sma

HOLD = "hold"
SMA_SWITCH = "sma_switch"
SIDE_BUY = "매수"
SIDE_SELL = "매도"


@dataclass(frozen=True)
class BookSpec:
    """장부 하나의 정의. name: 표시 이름(한글 그대로 시트·텔레그램에), kind: hold|sma_switch, asset: 공격 종목."""

    name: str
    kind: str
    asset: str


@dataclass(frozen=True)
class LedgerParams:
    """장부 공통 설정 (퍼센트 값은 % 단위)."""

    initial_usd: float
    commission_pct: float = 0.07
    slippage_pct: float = 0.0
    tax_rate_pct: float = 22.0
    dividend_withholding_pct: float = 15.0
    sma_days: int = 200
    signal_ticker: str = "QQQ"
    defensive_ticker: str = "SGOV"


@dataclass
class BookResult:
    """simulate_book 결과. daily: 날짜 인덱스 DataFrame, trades: 체결 목록, taxes: {연도: {realized, tax}}."""

    spec: BookSpec
    daily: pd.DataFrame
    trades: list[dict] = field(default_factory=list)
    taxes: dict[int, dict] = field(default_factory=dict)


def books_from_config(vb_cfg: dict) -> list[BookSpec]:
    """config.yaml virtual_books.books → [BookSpec]. 알 수 없는 kind면 ValueError."""
    out = []
    for b in vb_cfg["books"]:
        if b["kind"] not in (HOLD, SMA_SWITCH):
            raise ValueError(f"알 수 없는 장부 종류: {b['kind']!r} ({b['name']})")
        out.append(BookSpec(name=str(b["name"]), kind=str(b["kind"]), asset=str(b["asset"])))
    return out


def params_from_config(vb_cfg: dict, initial_usd: float) -> LedgerParams:
    """config.yaml virtual_books + 시작 달러 금액 → LedgerParams."""
    return LedgerParams(
        initial_usd=float(initial_usd),
        commission_pct=float(vb_cfg["commission_pct"]),
        slippage_pct=float(vb_cfg["slippage_pct"]),
        tax_rate_pct=float(vb_cfg["capital_gains_rate_pct"]),
        dividend_withholding_pct=float(vb_cfg["dividend_withholding_pct"]),
        sma_days=int(vb_cfg["sma_days"]),
        signal_ticker=str(vb_cfg["signal_ticker"]),
        defensive_ticker=str(vb_cfg["defensive_ticker"]),
    )


def required_tickers(books: list[BookSpec], params: LedgerParams) -> list[str]:
    """장부 계산에 필요한 종목(신호·방어·각 장부 종목), 중복 없이 처음 나온 순서대로."""
    seq = [params.signal_ticker] + [b.asset for b in books]
    if any(b.kind == SMA_SWITCH for b in books):
        seq.append(params.defensive_ticker)
    return list(dict.fromkeys(seq))


def regime_frame(signal_close: pd.Series, sma_days: int) -> pd.DataFrame:
    """신호 종목 종가 → 국면 표.

    입력: signal_close(오름차순 종가), sma_days
    출력: DataFrame(close, sma, above(1.0 위 / 0.0 같거나 아래 / NaN 미산출), gap_pct(종가/SMA − 1, %))
    """
    sma = signal_close.rolling(sma_days).mean()
    return pd.DataFrame({
        "close": signal_close,
        "sma": sma,
        "above": above_sma(signal_close, sma_days),
        "gap_pct": (signal_close / sma - 1.0) * 100.0,
    })


def switch_signals(regime: pd.DataFrame, start: pd.Timestamp) -> list[dict]:
    """시작일 이후(시작일 포함) 국면이 바뀐 신호일 목록.

    t일 above가 t−1일 above와 다르면 t일 신호(체결은 t+1 거래일 종가, 아직 없으면 fill_date None).
    출력: [{signal_date, fill_date, above(bool), close, sma, gap_pct}]
    """
    idx = regime.index
    out = []
    for i in range(1, len(idx)):
        t = idx[i]
        if t < start:
            continue
        a, p = regime["above"].iat[i], regime["above"].iat[i - 1]
        if np.isnan(a) or np.isnan(p) or a == p:
            continue
        out.append({
            "signal_date": t,
            "fill_date": idx[i + 1] if i + 1 < len(idx) else None,
            "above": bool(a == 1.0),
            "close": float(regime["close"].iat[i]),
            "sma": float(regime["sma"].iat[i]),
            "gap_pct": float(regime["gap_pct"].iat[i]),
        })
    return out


def _regime_reason(regime: pd.DataFrame, signal_date: pd.Timestamp, signal_ticker: str, prefix: str) -> str:
    r = regime.loc[signal_date]
    sign = ">" if r["above"] == 1.0 else "≤"
    return f"{prefix} ({signal_ticker} {signal_date.date()} 종가 {r['close']:.2f} {sign} 200일선 {r['sma']:.2f})"


def simulate_book(
    spec: BookSpec,
    closes: pd.DataFrame,
    dividends: pd.DataFrame,
    regime: pd.DataFrame,
    start: pd.Timestamp,
    params: LedgerParams,
    year_end_dates: set,
) -> BookResult:
    """장부 하나를 시작일부터 마지막 행까지 하루씩 굴린다.

    입력: spec, closes(종목별 종가, 거래일 오름차순 인덱스), dividends(같은 모양, 주당 분배금·없으면 0),
         regime(regime_frame 결과, closes와 같은 인덱스), start(시작일 — closes 안의 거래일 이상 첫 날),
         params, year_end_dates(그해 마지막 거래일 Timestamp 집합)
    출력: BookResult. daily 열: book, holding, qty, close, cash, value, cum_return_pct, drawdown_pct,
         realized_ytd, tax_cum, value_after_tax, value_after_tax_liq, dividends_cum
    예외: 시작일 이후 행이 없거나, 전환 장부의 직전 거래일 신호가 없으면(200일선 미산출) ValueError
    """
    days = closes.index[closes.index >= start]
    if len(days) == 0:
        raise ValueError(f"{spec.name}: 시작일 {pd.Timestamp(start).date()} 이후 가격이 없습니다")
    first = int(closes.index.get_loc(days[0]))
    c = params.commission_pct / 100.0
    s = params.slippage_pct / 100.0
    tr = params.tax_rate_pct / 100.0
    w = params.dividend_withholding_pct / 100.0

    cash = float(params.initial_usd)
    qty = 0
    ticker: str | None = None
    basis = 0.0
    realized_ytd = 0.0
    tax_cum = 0.0
    div_cum = 0.0
    peak = -math.inf
    rows: list[dict] = []
    trades: list[dict] = []
    taxes: dict[int, dict] = {}

    for k, d in enumerate(days):
        i = first + k
        prev_d = closes.index[i - 1] if i > 0 else None

        # 1) 분배금 — 어제 종가까지 들고 있던 주식 수 기준(배당락일 당일 판 주식도 받는다)
        if ticker is not None and qty > 0:
            dps = float(dividends.at[d, ticker]) if ticker in dividends.columns else 0.0
            if dps > 0:
                net = qty * dps * (1.0 - w)
                cash += net
                div_cum += net

        # 2) 오늘 종가에 들고 있어야 할 종목
        if spec.kind == HOLD:
            target, signal_date = spec.asset, days[0]
        else:
            if prev_d is None or np.isnan(regime["above"].at[prev_d]):
                raise ValueError(
                    f"{spec.name}: {d.date()} 체결에 쓸 직전 거래일 200일선 신호가 없습니다 (시작일 이전 데이터 부족)"
                )
            target = spec.asset if regime["above"].at[prev_d] == 1.0 else params.defensive_ticker
            signal_date = prev_d

        if target != ticker:
            if ticker is None:
                reason = "시작 매수 (그냥 보유)" if spec.kind == HOLD else _regime_reason(
                    regime, signal_date, params.signal_ticker, "시작 매수")
            else:
                reason = _regime_reason(regime, signal_date, params.signal_ticker, "200일선 전환")
            if ticker is not None and qty > 0:
                px = float(closes.at[d, ticker]) * (1.0 - s)
                gross = qty * px
                fee = gross * c
                proceeds = gross - fee
                pnl = proceeds - basis
                realized_ytd += pnl
                cash += proceeds
                trades.append({
                    "book": spec.name, "signal_date": signal_date, "fill_date": d, "side": SIDE_SELL,
                    "ticker": ticker, "qty": qty, "price": px, "commission": fee, "amount": proceeds,
                    "realized_pnl": pnl, "reason": reason,
                })
                qty, basis = 0, 0.0
            px = float(closes.at[d, target]) * (1.0 + s)
            n = int(math.floor(cash / (px * (1.0 + c)) + 1e-9))
            fee = n * px * c
            cost = n * px + fee
            cash -= cost
            qty, basis, ticker = n, cost, target
            trades.append({
                "book": spec.name, "signal_date": signal_date, "fill_date": d, "side": SIDE_BUY,
                "ticker": target, "qty": n, "price": px, "commission": fee, "amount": cost,
                "realized_pnl": None, "reason": reason,
            })

        # 3) 평가
        close = float(closes.at[d, ticker])
        value = cash + qty * close

        # 4) 연말 세금 (전환 체결 뒤)
        if d in year_end_dates:
            tax = max(realized_ytd, 0.0) * tr
            taxes[d.year] = {"realized": realized_ytd, "tax": tax, "settled_on": d}
            tax_cum += tax
            realized_ytd = 0.0

        liq = qty * close * (1.0 - s) * (1.0 - c)
        est_tax = max(realized_ytd + (liq - basis), 0.0) * tr
        peak = max(peak, value)
        rows.append({
            "date": d, "book": spec.name, "holding": ticker, "qty": qty, "close": close, "cash": cash,
            "value": value,
            "cum_return_pct": (value / params.initial_usd - 1.0) * 100.0,
            "drawdown_pct": (value / peak - 1.0) * 100.0,
            "realized_ytd": realized_ytd, "tax_cum": tax_cum,
            "value_after_tax": value - tax_cum,
            "value_after_tax_liq": cash + liq - tax_cum - est_tax,
            "dividends_cum": div_cum,
        })

    daily = pd.DataFrame(rows).set_index("date")
    return BookResult(spec=spec, daily=daily, trades=trades, taxes=taxes)


def summarize_book(result: BookResult, params: LedgerParams, min_days_for_annualized: int) -> dict:
    """장부 요약: 누적 수익·연환산·최대 낙폭·전환 횟수 등.

    출력: {book, start, end, initial_usd, value, cum_return_pct, cagr_pct(짧으면 None), mdd_pct, mdd_date,
          switches(전환 체결 횟수 — 시작 매수 제외), holding, tax_cum, value_after_tax, value_after_tax_liq}
    """
    d = result.daily
    start, end = d.index[0], d.index[-1]
    value = float(d["value"].iat[-1])
    days = (end - start).days
    cagr = None
    if days >= min_days_for_annualized and days > 0:
        cagr = ((value / params.initial_usd) ** (365.25 / days) - 1.0) * 100.0
    mdd_date = d["drawdown_pct"].idxmin()
    return {
        "book": result.spec.name,
        "start": start, "end": end,
        "initial_usd": params.initial_usd,
        "value": value,
        "cum_return_pct": float(d["cum_return_pct"].iat[-1]),
        "cagr_pct": cagr,
        "mdd_pct": float(d["drawdown_pct"].min()),
        "mdd_date": mdd_date,
        "switches": sum(1 for t in result.trades if t["side"] == SIDE_SELL),
        "holding": d["holding"].iat[-1],
        "tax_cum": float(d["tax_cum"].iat[-1]),
        "value_after_tax": float(d["value_after_tax"].iat[-1]),
        "value_after_tax_liq": float(d["value_after_tax_liq"].iat[-1]),
    }


def first_drawdown_breach(daily: pd.DataFrame, threshold_pct: float) -> pd.Timestamp | None:
    """낙폭이 처음 threshold_pct(예: −50) 이하가 된 날. 없으면 None."""
    hit = daily.index[daily["drawdown_pct"] <= threshold_pct]
    return hit[0] if len(hit) else None


@dataclass
class LedgerResult:
    """run_ledger 결과. books: {장부 이름: BookResult}, summaries: [summarize_book], regime, switches."""

    start: pd.Timestamp
    base_date: pd.Timestamp
    books: dict[str, BookResult]
    summaries: list[dict]
    regime: pd.DataFrame
    switches: list[dict]


def run_ledger(
    books: list[BookSpec],
    closes: pd.DataFrame,
    dividends: pd.DataFrame,
    start,
    params: LedgerParams,
    year_end_dates: set,
    min_days_for_annualized: int = 60,
) -> LedgerResult:
    """다섯 장부를 같은 시작일·같은 가격으로 굴린다.

    입력: books, closes(필요 종목 종가 — required_tickers, 거래일 오름차순), dividends(같은 모양),
         start(시작일), params, year_end_dates, min_days_for_annualized
    출력: LedgerResult (base_date = closes 마지막 날)
    예외: 시작일 이후 필요한 종목 종가가 빈 날이 있으면 ValueError (조용히 채우지 않는다)
    """
    start = pd.Timestamp(start)
    need = required_tickers(books, params)
    missing_cols = [t for t in need if t not in closes.columns]
    if missing_cols:
        raise ValueError(f"가격 열이 없습니다: {missing_cols}")
    after = closes.loc[closes.index >= start, need]
    holes = {t: [str(x.date()) for x in after.index[after[t].isna()]] for t in need if after[t].isna().any()}
    if holes:
        raise ValueError(f"시작일 이후 종가가 빈 날이 있습니다: {holes}")

    regime = regime_frame(closes[params.signal_ticker], params.sma_days)
    divs = dividends.reindex(index=closes.index, columns=need).fillna(0.0)
    results = {b.name: simulate_book(b, closes, divs, regime, start, params, year_end_dates) for b in books}
    first_day = closes.index[closes.index >= start][0]
    return LedgerResult(
        start=first_day,
        base_date=closes.index[-1],
        books=results,
        summaries=[summarize_book(r, params, min_days_for_annualized) for r in results.values()],
        regime=regime,
        switches=switch_signals(regime, first_day) if any(b.kind == SMA_SWITCH for b in books) else [],
    )


def regime_today(ledger: LedgerResult, books: list[BookSpec]) -> dict:
    """기준일 국면 요약 (브리핑 맨 위 한 줄·알림용).

    출력: {date, close, sma, gap_pct, above(bool), switch_holdings([(장부, 지금 보유 종목)]),
          signal_today(오늘 신호일이면 switch_signals의 그 항목, 아니면 None)}
    """
    d = ledger.base_date
    r = ledger.regime.loc[d]
    holds = [(b.name, ledger.books[b.name].daily["holding"].iat[-1]) for b in books if b.kind == SMA_SWITCH]
    sig = next((s for s in ledger.switches if s["signal_date"] == d), None)
    return {
        "date": d,
        "close": float(r["close"]),
        "sma": float(r["sma"]),
        "gap_pct": float(r["gap_pct"]),
        "above": bool(r["above"] == 1.0),
        "switch_holdings": holds,
        "signal_today": sig,
    }
