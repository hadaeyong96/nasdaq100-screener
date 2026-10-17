"""가상 장부 브리핑 문구·시트 표 (순수 함수 — 네트워크·파일 없음).

- regime_line: 브리핑 맨 위 한 줄 "오늘 국면: QQQ 200일선 위/아래(+x.x%) · 전환 장부 … 보유 중"
- switch_alert_text: 200일선 전환 신호가 나온 날 알림
- drawdown_alert_text: 지정 장부 낙폭이 처음 기준(−50%)을 넘은 날 한 번 알림
- daily_table / trades_table / summary_table: 구글 시트 세 탭(전략비교_일일·매매기록·전략요약) 값

장부 이름은 config.yaml의 한글 이름 그대로 쓴다. 금액은 달러(장부 운용 통화).
"""

from __future__ import annotations

import pandas as pd

from core.virtual_books import SMA_SWITCH, BookSpec, LedgerResult
from notify.briefing import REGIME_LINE_PREFIX as REGIME_PREFIX

VIRTUAL_NOTE = "(가상 장부 · 실제 주문 아님)"
_KOREAN_WEEKDAY = ["월", "화", "수", "목", "금", "토", "일"]


def _md(ts) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.month}/{ts.day}({_KOREAN_WEEKDAY[ts.dayofweek]})"


def _side_word(above: bool) -> str:
    return "위" if above else "아래"


def regime_line(regime: dict, signal_ticker: str = "QQQ", defensive_ticker: str = "SGOV",
                switch_assets: list[str] | None = None) -> str:
    """core.virtual_books.regime_today 결과 → 브리핑 맨 위 한 줄.

    예: "오늘 국면: QQQ 200일선 위(+3.2%) · 전환 장부 QLD·TQQQ 보유 중"
    오늘이 전환 신호일이면 보유와 국면이 엇갈려 보이지 않게 다음 거래일 체결을 덧붙인다.
    """
    held = list(dict.fromkeys(h for _, h in regime["switch_holdings"]))
    line = (f"{REGIME_PREFIX} {signal_ticker} 200일선 {_side_word(regime['above'])}({regime['gap_pct']:+.1f}%)"
            f" · 전환 장부 {'·'.join(held)} 보유 중")
    sig = regime.get("signal_today")
    if sig is not None:
        to = "·".join(switch_assets or []) if sig["above"] else defensive_ticker
        line += f" → 오늘 전환 신호, 다음 거래일 종가에 {to}로"
    return line


def switch_alert_text(signal: dict, moves: list[tuple[str, str, str]], next_day=None,
                      signal_ticker: str = "QQQ") -> str:
    """200일선 전환 신호 알림.

    입력: signal(core.virtual_books.switch_signals 항목), moves([(장부, 지금 종목, 바꿀 종목)]),
         next_day(체결 예정 거래일, 모르면 None)
    """
    sign = ">" if signal["above"] else "≤"
    when = f"다음 거래일 {_md(next_day)}" if next_day is not None else "다음 거래일"
    lines = [
        f"🔀 200일선 전환 신호 · {_md(signal['signal_date'])} 마감",
        f"{signal_ticker} 종가 {signal['close']:,.2f} {sign} 200일선 {signal['sma']:,.2f} "
        f"({signal['gap_pct']:+.1f}%) → 200일선 {_side_word(signal['above'])}",
        f"{when} 종가에 가상 체결: " + ", ".join(f"{b} {a}→{z}" for b, a, z in moves),
        VIRTUAL_NOTE,
    ]
    return "\n".join(lines)


def drawdown_alert_text(book: str, daily: pd.DataFrame, breach_date, threshold_pct: float) -> str:
    """지정 장부 낙폭이 처음 threshold_pct 이하가 된 날 알림 (한 번만)."""
    upto = daily.loc[:breach_date]
    row = upto.iloc[-1]
    peak_date = upto["value"].idxmax()
    return "\n".join([
        f"⚠️ {book} 낙폭 {row['drawdown_pct']:.1f}% — 시작 후 처음 {threshold_pct:.0f}% 아래 ({_md(breach_date)} 마감)",
        f"고점 ${upto['value'].max():,.0f}({_md(peak_date)}) → ${row['value']:,.0f}",
        VIRTUAL_NOTE,
    ])


def _r(v, digits: int = 2):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return round(float(v), digits)


def _d(v) -> str:
    return pd.Timestamp(v).date().isoformat() if v is not None else ""


def _yn(v) -> str:
    return "" if v is None else ("예" if v else "아니오")


def daily_table(ledger: LedgerResult, books: list[BookSpec], benchmark: str, extra: list[dict] | None = None) -> list[list]:
    """전략비교_일일 탭: 날짜·QQQ 국면 + 장부별 총자산 / 누적 수익률 / QQQ 대비 / 고점 대비 낙폭 (오래된 날 → 최근).

    extra: 추천 따라하기 장부들 [{name, start, daily(DataFrame: value·cum_return_pct·drawdown_pct)}] — 시작일이
    다를 수 있어 "QQQ 대비"는 그 장부 시작일부터 잰 QQQ 그냥 보유 수익과 비교한다. 장부가 없는 날은 빈칸.
    """
    extra = extra or []
    names = [b.name for b in books]
    all_names = names + [x["name"] for x in extra]
    others = [n for n in all_names if n != benchmark]
    header = (["날짜", "QQQ 종가", "QQQ 200일선", "국면", "전환 장부 보유"]
              + [f"{n} 총자산($)" for n in all_names]
              + [f"{n} 누적(%)" for n in all_names]
              + [f"{n} QQQ 대비(%p)" for n in others]
              + [f"{n} 낙폭(%)" for n in all_names])
    switch_names = [b.name for b in books if b.kind == SMA_SWITCH]
    dailies = {n: ledger.books[n].daily for n in names}
    dailies.update({x["name"]: x["daily"] for x in extra})
    bench_value = dailies[benchmark]["value"]
    starts = {n: dailies[n].index[0] for n in names}
    starts.update({x["name"]: pd.Timestamp(x["start"]) for x in extra})

    def cell(n, d, col):
        df = dailies[n]
        return _r(df.at[d, col]) if d in df.index else ""

    def vs_bench(n, d):
        df = dailies[n]
        if d not in df.index or starts[n] not in bench_value.index:
            return ""
        if n in names:  # 기존 5개 장부는 시작일이 같다 — 예전 계산 그대로
            bench_cum = dailies[benchmark].at[d, "cum_return_pct"]
        else:
            bench_cum = (bench_value.at[d] / bench_value.at[starts[n]] - 1.0) * 100.0
        return _r(df.at[d, "cum_return_pct"] - bench_cum)

    dates = sorted(set().union(*[set(df.index) for df in dailies.values()]))
    rows = [header]
    for d in dates:
        r = ledger.regime.loc[d]
        held = "·".join(dict.fromkeys(dailies[n].at[d, "holding"] for n in switch_names if d in dailies[n].index))
        rows.append(
            [_d(d), _r(r["close"]), _r(r["sma"]), _side_word(r["above"] == 1.0), held]
            + [cell(n, d, "value") for n in all_names]
            + [cell(n, d, "cum_return_pct") for n in all_names]
            + [vs_bench(n, d) for n in others]
            + [cell(n, d, "drawdown_pct") for n in all_names]
        )
    return rows


def trades_table(ledger: LedgerResult, books: list[BookSpec], extra: list[dict] | None = None) -> list[list]:
    """매매기록 탭: 장부 이름, 신호일, 체결일, 구분, 종목, 수량, 가격, 수수료, 실현손익, 사유, 브리핑 추천에도 있었음.

    extra(추천 따라하기 장부)의 체결·"자리 없음" 행도 함께 넣는다. SGOV(남는 현금) 매매는 넣지 않는다.
    "브리핑 추천에도 있었음"은 추천 따라하기 장부만 예/아니오(그날 브리핑 기록이 없으면 빈칸).
    """
    extra = extra or []
    order = {b.name: i for i, b in enumerate(books)}
    order.update({x["name"]: len(books) + i for i, x in enumerate(extra)})
    trades = [t for b in books for t in ledger.books[b.name].trades] + [t for x in extra for t in x["trades"]]
    trades.sort(key=lambda t: (pd.Timestamp(t["fill_date"]), order[t["book"]], 0 if t["side"] == "매도" else 1))
    rows = [["장부", "신호일", "체결일", "구분", "종목", "수량", "가격($)", "수수료($)", "실현손익($)", "사유",
             "브리핑 추천에도 있었음"]]
    for t in trades:
        rows.append([t["book"], _d(t["signal_date"]), _d(t["fill_date"]), t["side"], t["ticker"], int(t["qty"]),
                     _r(t["price"], 4), _r(t["commission"]), _r(t["realized_pnl"]), t["reason"], _yn(t.get("in_briefing"))])
    return rows


# 정해진 신호일마다 상위 10을 같은 비중으로 맞추는 장부 kind → (시작 조건 줄 이름, 시작일 설명)
REBALANCE_KINDS = {"momentum": ("6개월 강세", "첫 월말 신호일"), "accel": ("한 달 가속", "처음 돈 기준일 = 첫 신호일")}


def summary_table(ledger: LedgerResult, meta: dict, extra: list[dict] | None = None) -> list[list]:
    """전략요약 탭: 장부별 누적 수익·연환산·최대 낙폭·전환 횟수 + 세후 값, 아래에 시작 조건·가정.

    meta: {start_krw, start_fx, fx_date, initial_usd, updated_at, min_days_for_annualized, notes([str])}
    extra: 추천 따라하기 장부 [{name, start, summary, meta}] — 이 장부들만 완료 거래 수·승률·평균 보유일·
    SGOV 보유액 열을 채운다(전환 횟수는 비움). kind == "momentum"(6개월 강세 따라하기)은 이 장부만
    교체 횟수·현재 보유 10종목 열을 채운다.
    """
    extra = extra or []
    rows = [["장부", "시작일", "기준일", "시작 자금($)", "총자산($)", "누적 수익(%)", "연환산(%)", "최대 낙폭(%)",
             "최대 낙폭일", "전환 횟수", "현재 보유", "납부 세금 누적($)",
             "청산 가정 세후 추정($)", "청산 가정 세후 수익(%)", "완료 거래 수", "승률(%)", "평균 보유일", "SGOV 보유액($)",
             "교체 횟수", "현재 보유 10종목"]]

    def base(s, switches):
        return [s["book"], _d(s["start"]), _d(s["end"]), _r(s["initial_usd"]), _r(s["value"]), _r(s["cum_return_pct"]),
                _r(s["cagr_pct"]), _r(s["mdd_pct"]), _d(s["mdd_date"]), switches, s["holding"],
                _r(s["tax_cum"]), _r(s["value_after_tax_liq"]),
                _r((s["value_after_tax_liq"] / s["initial_usd"] - 1.0) * 100.0)]

    for s in ledger.summaries:
        rows.append(base(s, int(s["switches"])) + ["", "", "", "", "", ""])
    momentum = [x for x in extra if x.get("kind") in REBALANCE_KINDS]
    picks = [x for x in extra if x.get("kind") not in REBALANCE_KINDS]
    for x in picks:
        s = x["summary"]
        rows.append(base(s, "") + [int(s["completed"]), _r(s["win_rate_pct"], 1), _r(s["avg_hold_days"], 1), _r(s["sgov_value"]),
                                   "", ""])
    for x in momentum:
        s = x["summary"]
        rows.append(base(s, "") + ["", "", "", "", int(s["rebalances"]), s["holdings_list"]])
    rows.append([])
    rows.append(["시작 조건", f"장부당 {meta['start_krw']:,.0f}원 ÷ {meta['start_fx']:,.2f}원/$ ({meta['fx_date']}) "
                 f"= ${meta['initial_usd']:,.2f}, 이후 달러 운용"])
    rows.append(["config 고정값", f"start_date: \"{_d(ledger.start)}\"  start_fx: {meta['start_fx']!r}"])
    for x in picks[:1]:  # 추천 따라하기 장부들은 시작일·환율을 함께 쓴다
        m = x["meta"]
        rows.append(["시작 조건(추천 따라하기)", f"장부당 {m['start_krw']:,.0f}원 ÷ {m['start_fx']:,.2f}원/$ ({m['fx_date']}) "
                     f"= ${m['initial_usd']:,.2f}, 이후 달러 운용"])
        rows.append(["config 고정값(picks)", f"start_date: \"{_d(x['start'])}\"  start_fx: {m['start_fx']!r}"])
    for x in momentum:  # 8·9번째 장부는 각자 첫 신호일에 따로 시작한다
        m = x["meta"]
        label, first = REBALANCE_KINDS[x["kind"]]
        rows.append([f"시작 조건({label})", f"장부당 {m['start_krw']:,.0f}원 ÷ {m['start_fx']:,.2f}원/$ ({m['fx_date']}) "
                     f"= ${m['initial_usd']:,.2f}, 이후 달러 운용 · 시작일 {_d(x['start'])}({first})"])
        rows.append([f"config 고정값({x['kind']})", f"start_date: \"{_d(x['start'])}\"  start_fx: {m['start_fx']!r}"])
    rows.append(["갱신", meta["updated_at"]])
    rows.append(["연환산", f"시작 후 {meta['min_days_for_annualized']}일 미만이면 비움"])
    for n in meta.get("notes", []):
        rows.append(["가정", n])
    return rows


# ── 보여 주는 이름 (2026-10-17) — 아침 편지·시트 4개 탭에만. 장부 id·DB·계산은 config의 name 그대로 ──


def display_map(cfg: dict) -> dict[str, str]:
    """config virtual_books.display_names → {장부 id: 보여 주는 이름}. 없으면 빈 dict."""
    return dict(((cfg or {}).get("virtual_books") or {}).get("display_names") or {})


def display_name(cfg: dict, name: str) -> str:
    """장부 id → 보여 주는 이름(없으면 id 그대로)."""
    return display_map(cfg).get(name, name)


def _swap_cell(v, mapping: dict[str, str]):
    """셀 하나: 장부 이름과 정확히 같으면 바꾸고, "장부 이름 " 로 시작하는 머리글(예: "… 총자산($)")은 앞부분만 바꾼다."""
    if not isinstance(v, str):
        return v
    if v in mapping:
        return mapping[v]
    for old, new in mapping.items():
        if v.startswith(old + " "):
            return new + v[len(old):]
    return v


def rename_tables(tables: dict[str, list[list]], mapping: dict[str, str]) -> dict[str, list[list]]:
    """시트 탭 값의 장부 이름을 보여 주는 이름으로 (새 dict, 원본은 그대로)."""
    if not mapping:
        return tables
    long_first = dict(sorted(mapping.items(), key=lambda kv: -len(kv[0])))  # 긴 이름부터(앞부분이 겹치는 이름 대비)
    return {tab: [[_swap_cell(v, long_first) for v in row] for row in rows] for tab, rows in tables.items()}


def unrename_tables(tables: dict[str, list[list]], mapping: dict[str, str]) -> dict[str, list[list]]:
    """rename_tables의 반대 — 시트에서 읽은 값을 장부 id로 되돌린다(DB 복구용). 옛 이름 그대로인 값은 그대로 둔다."""
    return rename_tables(tables, {v: k for k, v in mapping.items()})
