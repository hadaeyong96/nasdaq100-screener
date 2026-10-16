"""폭락 연구 관문 2 — 실시간 신호 재구성 (docs/prereg/shock_dip.md v2). 전략 수익률은 계산하지 않는다.

각 지수를 판정 구간 끝에서 자른 뒤(그 뒤 데이터는 읽지 않음), 그날까지 공개된 값만으로 신호를 만든다.
캐시는 scripts/shock_dip_gate01.py와 같은 data/cache/shock_dip/ (BAA·GS10·^NDX는 여기서 추가).

실행: python scripts/shock_dip_gate2.py
출력: docs/research/shock_dip_signals.md, shock_dip_trades.csv, shock_dip_qqq_corrections.csv,
      docs/research/shock_dip_daily/<지수>.csv.gz (일별 신호 표)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import l1d  # noqa: E402
from core import shock_dip as sd  # noqa: E402
from scripts.shock_dip_gate01 import (  # noqa: E402
    load_alfred_vintages, load_cape, load_fred, load_yf, md_table,
)

OUT = ROOT / "docs" / "research"
DAILY = OUT / "shock_dip_daily"
B3_START = pd.Timestamp("1986-01-01")
VIX_FROM = pd.Timestamp("1990-01-01")

# (이름, 티커, 판정 시작, 판정 끝, 해외 여부)
PERIODS = [
    ("S&P500", "^GSPC", "1960-04-01", "1998-12-31", False),
    ("나스닥 종합", "^IXIC", "1972-01-01", "1998-12-31", False),
    ("QQQ(보정)", "QQQ", "1999-01-01", "2021-12-31", False),
    ("닛케이225", "^N225", None, "2021-12-31", True),
    ("DAX", "^GDAXI", None, "2021-12-31", True),
    ("FTSE100", "^FTSE", None, "2021-12-31", True),
    ("항셍", "^HSI", None, "2021-12-31", True),
    ("코스피", "^KS11", None, "2021-12-31", True),
]
CHECK_MONTHS = ["1987-10", "1998-08", "2000-04", "2001-09", "2008-01", "2008-10", "2011-08", "2018-10", "2018-12", "2020-03"]


def load_close(ticker: str) -> pd.Series:
    c, _ = l1d.clean_index(load_yf(ticker)["Close"])
    return c


def macro_inputs():
    """월별·일별 거시 입력과 공개일. 출력: dict."""
    sahm = load_fred("SAHMREALTIME").dropna()
    sahm_rel = sd.release_after_month_end(sahm.index, load_alfred_vintages("UNRATE"))
    spread = (load_fred("BAA") - load_fred("GS10")).dropna()
    spread = spread[spread.index >= "1953-04-01"]
    cape = load_cape().dropna()
    vxo = load_fred("VXOCLS").dropna()
    vix = load_yf("^VIX")["Close"].dropna()
    fear = pd.concat([vxo[(vxo.index >= B3_START) & (vxo.index < VIX_FROM)], vix[vix.index >= VIX_FROM]]).sort_index()
    return {
        "sahm": sahm, "sahm_rel": sahm_rel,
        "spread": spread, "spread_av": sd.month_start_available(spread.index, 1),
        "cape": cape, "cape_av": sd.month_start_available(cape.index, 2),
        "fear": fear, "fear_av": pd.Series(fear.index, index=fear.index),
    }


def build(name, close, start, end, foreign, mi):
    """한 지수의 일별 신호 표와 변형별 거래. close는 이미 판정 끝에서 잘린 것."""
    d = close.index
    f = sd.price_features(close)
    sahm = sd.asof_daily(mi["sahm"], mi["sahm_rel"], d, strict=foreign)
    baa_pct = sd.asof_percentile(mi["spread"], mi["spread_av"], d, strict=False)
    cape_pct = sd.asof_percentile(mi["cape"], mi["cape_av"], d, strict=False)
    fear = sd.asof_daily(mi["fear"], mi["fear_av"], d, strict=foreign)
    conds = sd.variant_conditions(f, sahm, baa_pct, cape_pct, fear)
    in_period = pd.Series((d >= (pd.Timestamp(start) if start else d[0])) & (d <= pd.Timestamp(end)), index=d)
    base_ready = f["ret20"].notna() & f["sma200"].shift(20).notna() & sahm.notna()
    ready = {
        "A": base_ready, "B4": base_ready,
        "B1": base_ready & baa_pct.notna(),
        "B2": base_ready & cape_pct.notna(),
        "B3": fear.notna() & (d >= B3_START),
        "D": f["high252"].notna() & f["sma200"].notna(),
    }
    table = pd.DataFrame({
        "close": f["close"].round(4), "ret20": f["ret20"].round(5), "above200_20ago": f["above200_20ago"],
        "sahm_asof": sahm, "baa_gs10_pct": baa_pct.round(4), "cape_pct": cape_pct.round(4), "vix_vxo": fear,
        "sma20_cross_up": f["cross20_up"], "high252": f["high252"].round(4), "above200": f["above200"],
        "in_period": in_period,
    })
    trades, starts = [], {}
    for v in sd.VARIANTS:
        allowed = in_period & ready[v]
        starts[v] = allowed.idxmax().date() if allowed.any() else None
        table[f"cond_{v}"] = conds[v] & allowed
        tr = sd.simulate(v, f, conds[v], sahm, allowed)
        sig = pd.Series(False, index=d)
        hold = pd.Series(False, index=d)
        for t in tr:
            sig.iloc[t.signal_pos] = True
            hold.iloc[t.entry_pos: (t.exit_pos if t.exit_pos is not None else len(d) - 1) + 1] = True
            trades.append({
                "지수": name, "변형": v, "신호일": d[t.signal_pos].date(), "체결일": d[t.entry_pos].date(),
                "청산 판단일": d[t.exit_signal_pos].date() if t.exit_signal_pos is not None else "",
                "청산 체결일": d[t.exit_pos].date() if t.exit_pos is not None else "보유 중(데이터 끝)",
                "보유 거래일": (t.exit_pos - t.entry_pos) if t.exit_pos is not None else "",
                "청산 사유": t.reason,
            })
        table[f"entry_signal_{v}"] = sig
        table[f"holding_{v}"] = hold
    return table, trades, starts


def check_cell(tr: pd.DataFrame, table: pd.DataFrame, v: str, month: str) -> str:
    """사건 달에 그 변형이 무엇을 했는지 한 칸 문자열."""
    p = pd.Period(month)
    m0, m1 = p.start_time, p.end_time
    t = tr[tr["변형"] == v]
    sig = t[(pd.to_datetime(t["신호일"]) >= m0) & (pd.to_datetime(t["신호일"]) <= m1)]
    if len(sig):
        return "신호 " + ", ".join(pd.to_datetime(sig["신호일"]).dt.strftime("%m-%d"))
    for _, r in t.iterrows():
        s = pd.Timestamp(r["신호일"])
        e = pd.Timestamp(r["청산 체결일"]) if r["청산 체결일"] != "보유 중(데이터 끝)" else pd.Timestamp.max
        if s < m0 and e >= m0:
            return f"보유 중({s:%Y-%m-%d} 신호)"
    seg = table.loc[m0:m1]
    if len(seg) == 0 or not seg["in_period"].any():
        return "구간 밖"
    if seg[f"cond_{v}"].any():
        return "조건○ 진입✕"
    return "—"


def ks11_check() -> list[str]:
    """^KS11 2026-06 고점 이후 하락이 원자료에서도 보이는지 — 다른 출처(KODEX 200, EWY)와 대조."""
    import yfinance as yf

    lines = []
    k = load_yf("^KS11")["Close"].dropna()
    seg = k.loc["2026-05-01":]
    pk = seg.idxmax()
    lines.append(f"- ^KS11(야후, 캐시): 2026-05 이후 최고 {pk.date()} {seg.max():.2f}, 마지막 {seg.index[-1].date()} {seg.iloc[-1]:.2f} "
                 f"→ {seg.iloc[-1] / seg.max() - 1:+.1%}. 최저 {seg.idxmin().date()} {seg.min():.2f} ({seg.min() / seg.max() - 1:+.1%})")
    r = seg.pct_change().dropna()
    lines.append(f"- 같은 기간 ^KS11 하루 변동 최대 상승 {r.max():+.2%}({r.idxmax().date()}), 최대 하락 {r.min():+.2%}({r.idxmin().date()}), "
                 f"±8% 넘는 날 {int((r.abs() > 0.08).sum())}일, 같은 종가 연속 5일 이상 구간 {int(((seg.diff() == 0).astype(int).groupby((seg.diff() != 0).cumsum()).sum() >= 4).sum())}개")
    fresh = yf.download("^KS11", start="2026-05-01", end="2026-10-17", auto_adjust=False, progress=False, threads=False)["Close"]
    fresh = (fresh.iloc[:, 0] if isinstance(fresh, pd.DataFrame) else fresh).dropna()
    fresh.index = pd.DatetimeIndex(fresh.index).tz_localize(None)
    j = pd.concat([seg, fresh], axis=1).dropna()
    lines.append(f"- 오늘 다시 받은 ^KS11과 캐시 비교: 겹친 {len(j)}일 최대 차이 {(j.iloc[:, 0] - j.iloc[:, 1]).abs().max():.4f}")
    for tk, label in [("069500.KS", "KODEX 200(코스피200 ETF, 원화)"), ("EWY", "EWY(미국 상장 한국 ETF, 달러)")]:
        x = yf.download(tk, start="2026-05-01", end="2026-10-17", auto_adjust=False, progress=False, threads=False)["Close"]
        x = (x.iloc[:, 0] if isinstance(x, pd.DataFrame) else x).dropna()
        if x.empty:
            lines.append(f"- {label}: 받기 실패(빈 결과)")
            continue
        x.index = pd.DatetimeIndex(x.index).tz_localize(None)
        xp = x.loc[: pk + pd.Timedelta(days=3)]
        base = xp.iloc[-1] if len(xp) else x.iloc[0]
        lines.append(f"- {label}: {pk.date()} 무렵 {base:.2f} → 마지막 {x.index[-1].date()} {x.iloc[-1]:.2f} ({x.iloc[-1] / base - 1:+.1%}), "
                     f"구간 최고 {x.max():.2f}({x.idxmax().date()}) 대비 {x.iloc[-1] / x.max() - 1:+.1%}")
    return lines


def main() -> None:
    DAILY.mkdir(parents=True, exist_ok=True)
    mi = macro_inputs()
    qqq_raw = load_close("QQQ")
    ndx = load_close("^NDX")
    all_trades, starts_rows, tables, raw_rows = [], [], {}, []
    corrections = None
    for name, tk, start, end, foreign in PERIODS:
        close = load_close(tk)
        if tk == "QQQ":
            close, bad = sd.correct_with_reference(qqq_raw, ndx, 0.02)
            corrections = bad
        close = close[close.index <= pd.Timestamp(end)]
        table, trades, starts = build(name, close, start, end, foreign, mi)
        table.to_csv(DAILY / f"{tk.replace('^', '')}.csv.gz", compression="gzip")
        tables[name] = table
        all_trades += trades
        starts_rows.append({"지수": name, "판정 구간": f"{start or close.index[0].date()} ~ {end}",
                            **{v: str(starts[v]) if starts[v] else "없음" for v in sd.VARIANTS}})
        raw_rows.append({"지수": name, **{v: int(table[f"cond_{v}"].sum()) for v in sd.VARIANTS}})

    tr = pd.DataFrame(all_trades)
    tr.to_csv(OUT / "shock_dip_trades.csv", index=False, encoding="utf-8-sig")
    pd.Series([x.date() for x in corrections], name="대체한 날짜(QQQ 일간 수익률 → ^NDX)").to_csv(
        OUT / "shock_dip_qqq_corrections.csv", index=False, encoding="utf-8-sig")

    counts = tr.pivot_table(index="지수", columns="변형", values="신호일", aggfunc="size", fill_value=0)
    counts = counts.reindex(index=[p[0] for p in PERIODS], columns=sd.VARIANTS, fill_value=0).reset_index()

    check_rows = []
    for month in CHECK_MONTHS:
        names = ["S&P500", "나스닥 종합"] if month < "1999" else ["QQQ(보정)"]
        for nm in names:
            check_rows.append({"달": month, "지수": nm,
                               **{v: check_cell(tr[tr["지수"] == nm], tables[nm], v, month) for v in sd.VARIANTS}})

    lists = []
    for name, *_ in PERIODS:
        lists.append(f"### {name}")
        lists.append("")
        for v in sd.VARIANTS:
            t = tr[(tr["지수"] == name) & (tr["변형"] == v)]
            dates = ", ".join(str(x) for x in t["신호일"]) or "없음"
            lists.append(f"- **{v}** ({len(t)}회): {dates}")
        lists.append("")

    md = [
        "# 폭락 연구 관문 2 — 실시간 신호 재구성 (전략 수익률 계산 없음)", "",
        "> 규칙: `docs/prereg/shock_dip.md` v2(태그 `shock-dip-prereg-v2`). 코드 `core/shock_dip.py`, 실행 `scripts/shock_dip_gate2.py`.",
        "> 각 지수는 판정 구간 끝에서 잘라 그 뒤 데이터를 쓰지 않는다. 각 날짜의 값은 그날까지 공개된 것만 쓴다.",
        "> 일별 표: `docs/research/shock_dip_daily/<지수>.csv.gz`, 거래(신호·청산 날짜만): `shock_dip_trades.csv`.", "",
        "## 구현 해석 (사전 등록 원문에 명시 없음 — 관문 3 전 확인 필요)", "",
        "1. 미국 외 구간·지수에서는 「QQQ」를 그 지수로 읽는다(신호·청산 모두 그 지수 종가).",
        "2. 보유 중(진입 신호일 ~ 청산 체결일)에는 새 신호를 무시하고, 청산 체결일 다음 거래일부터 다시 신호를 찾는다.",
        "3. A·B1·B2·B4 청산 판단은 진입 체결일 종가부터 매일 한다. 「진입 직전 252거래일 최고가」 = 신호일까지 252거래일 최고 종가. "
        "「60거래일 경과」 = 체결일 + 60거래일 종가에 판단 → 다음 거래일 체결.",
        "4. B3: 체결일 + 20거래일 종가에 QQQ 100%로 복귀(이날까지 보유 중).",
        "5. B4: ①②③ 성립일(셋업) 다음 날부터 60거래일 안에서 「종가가 20일선 위로 올라온 날」 = 오늘 종가 > 20일선이고 어제 종가 ≤ 어제 20일선. "
        "없으면 셋업 취소, 61거래일째부터 다시 찾는다. 셋업 대기 중 새 셋업은 무시.",
        "6. D: 진입 조건의 252일 고점 = 오늘 포함 252거래일 최고 종가. 청산 목표 = 신호일의 그 값(고정). 「즉시」 = 다른 변형처럼 판단 다음 거래일 종가.",
        "7. 해외 지수는 미국 발표값(샴·VIX/VXO)을 공개일 다음 날부터 쓴다(미국 발표가 그 시장 마감 뒤). 월별 BAA−GS10·CAPE는 v2대로 월 1일 기준.",
        "8. 샴 공개일 = 월 말일 뒤 첫 UNRATE 발표일(ALFRED 빈티지) 근사. 1960-03-15 발표가 첫 값.",
        "9. 판정 구간 첫날부터 신호를 찾되, 지표 계산에는 구간 앞 데이터도 쓴다(200일선 등). 필요한 값이 없는 날은 그 변형에서 제외.",
        "", "## 변형별 진입 신호 횟수 (보유 중 무시 반영)", "", md_table(counts), "",
        "## 참고: 진입 조건이 성립한 날 수 (보유 중 무시 전, B4는 셋업일)", "", md_table(pd.DataFrame(raw_rows)), "",
        "## 변형별 실제 첫 판정 가능일", "", md_table(pd.DataFrame(starts_rows)), "",
        "## 확인할 사건 달", "",
        "> 칸: 「신호 월-일」 = 그 달 진입 신호일 / 「보유 중(…)」 = 앞 신호로 이미 보유 / 「조건○ 진입✕」 = 조건은 성립했으나 보유 중·B4 대기 등으로 진입 안 함 / — = 없음", "",
        md_table(pd.DataFrame(check_rows)), "",
        "## QQQ 보정 (v2-5)", "",
        f"- |QQQ 일간 수익률 − ^NDX 일간 수익률| > 2%인 날 **{len(corrections)}일**을 ^NDX 일간 수익률로 대체(전체 기간 1999-03-10 ~ 2026-10-16 기준, 판정에는 2021-12-31까지만 사용). "
        f"판정 구간 안 {sum(1 for x in corrections if x <= pd.Timestamp('2021-12-31'))}일. 목록: `shock_dip_qqq_corrections.csv`",
        "- 날짜: " + ", ".join(str(x.date()) for x in corrections), "",
        "## 진입 신호일 목록", "", *lists,
        "## 코스피 2026-06 고점 이후 하락 확인 (보고만, 판정 구간 밖)", "", *ks11_check(), "",
    ]
    (OUT / "shock_dip_signals.md").write_text("\n".join(md), encoding="utf-8")
    print("done", len(tr))


if __name__ == "__main__":
    main()
