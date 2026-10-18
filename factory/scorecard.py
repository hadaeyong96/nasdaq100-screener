"""가상 장부 월간 성적표 — 순수 함수 (전략공장 F2). 네트워크·파일·DB·현재 시각에 접근하지 않는다.

입력은 장부별 일별 기록(DataFrame)·매매 기록·QQQ 종가뿐이다. 가상 장부 계산 로직은 건드리지 않고 기록만 읽는다.

정의 (작업 F2 지시, 2026-10-16):
- 비교 곡선: 전략과 QQQ 모두 "그날 전부 팔았다면 세후 가치"(value_after_tax_liq) — 공용 규칙 2
- QQQ = 1번 장부 「QQQ 그냥 보유」의 같은 날 기록. 늦게 시작한 장부는 자기 시작일부터 같은 날짜끼리 비교
- 월 수익 = 그달 마지막 기록 ÷ 전달 마지막 기록(첫 달은 첫 기록) − 1. 이번 달·누적 QQQ 대비 = 수익률 차(%p)
- 이긴 달 = 그달 수익이 QQQ보다 높은 달. 최대 낙폭 = 세전 총자산(value) 기준
- 국면: factory.regimes.classify(QQQ 종가). 지난달 국면은 일수 비율, 국면별 누적은 그날 국면 → 다음 날 수익.
  국면별 일수 60거래일 미만이면 "참고(INCONCLUSIVE)"
- 분기 조기 탈락 제안(분기 마지막 달 성적표에서만): 기록 6개월 이상 + 누적 QQQ 대비 −5%p 이하 → "중단 제안"
  (1·2번은 기준선이라 제외). 제안만 하고 장부를 멈추지 않는다
- 12개월 실전 검토 제안: 기록 12개월 이상 + 연환산 QQQ 대비 +2%p 이상 + 이긴 달 60% 이상 + 최대 낙폭 차 10%p 이내
  → "실제 돈 일부 검토" 제안만
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from factory import regimes as rg

MIN_REGIME_DAYS = 60
STOP_MONTHS, STOP_GAP_PP = 6, -5.0
REVIEW_MONTHS, REVIEW_ANN_PP, REVIEW_WIN_RATIO, REVIEW_MDD_GAP_PP = 12, 2.0, 0.60, 10.0
BASELINE_NOS = (1, 2)  # 조기 탈락 제안에서 빼는 기준선 장부
QUARTER_END_MONTHS = (3, 6, 9, 12)


@dataclass(frozen=True)
class Strategy:
    no: int
    name: str
    label: str = ""  # 보여 주는 이름(2026-10-17, config virtual_books.display_names) — 비면 name. DB 조회는 name


def strategies_from_cfg(cfg: dict) -> list[Strategy]:
    """config.yaml virtual_books → 전략 1~9번 (1~5 기존 장부, 6·7 추천 따라하기, 8 가치평가(2026-10-18~, 전 6개월 강세), 9 한 달 가속)."""
    vb = cfg["virtual_books"]
    names = [b["name"] for b in vb["books"]]
    names += [b["name"] for b in (vb.get("picks") or {}).get("books", [])]
    for key in ("momentum", "value_score", "accel"):  # 8번 자리: 2026-10-18부터 가치평가(꺼 둔 장부는 뺀다)
        c = vb.get(key) or {}
        if c.get("name") and c.get("enabled", True):
            names.append(c["name"])
    labels = vb.get("display_names") or {}
    return [Strategy(i + 1, n, labels.get(n, n)) for i, n in enumerate(names)]


def is_first_trading_day(day, trading_days: pd.DatetimeIndex) -> bool:
    """day가 거래일이고, 같은 달에 그보다 이른 거래일이 없으면 True (주말·휴장 반영은 trading_days가 한다)."""
    d = pd.Timestamp(day).normalize()
    td = pd.DatetimeIndex(trading_days).normalize()
    if d not in td:
        return False
    same = td[(td.year == d.year) & (td.month == d.month)]
    return bool(len(same)) and same.min() == d


def _month_ends(s: pd.Series) -> pd.Series:
    """일별 Series → 달마다 마지막 값 (인덱스 Period[M])."""
    return s.groupby(s.index.to_period("M")).last()


def _mdd(values: pd.Series) -> float:
    v = values.astype(float)
    return float((v / v.cummax() - 1.0).min() * 100.0) if len(v) else float("nan")


def _months_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    """a에서 b까지 꽉 찬 달 수 (a + n달 ≤ b인 가장 큰 n)."""
    n = (b.year - a.year) * 12 + (b.month - a.month)
    if a + pd.DateOffset(months=n) > b:
        n -= 1
    return max(n, 0)


def regime_month_ratio(qqq_close: pd.Series, month: pd.Period) -> dict[str, float]:
    """그달 거래일 중 국면별 비율(0~1). 판단불가 포함."""
    reg = rg.classify(qqq_close)
    m = reg[reg.index.to_period("M") == month]
    if m.empty:
        return {}
    vc = m.value_counts(normalize=True)
    return {k: float(vc.get(k, 0.0)) for k in (rg.UP, rg.DOWN, rg.CRASH, rg.UNKNOWN) if vc.get(k, 0.0) > 0}


def regime_perf(values: pd.Series, bench: pd.Series, qqq_close: pd.Series) -> dict[str, dict]:
    """국면별 누적 수익(%) — 그날 국면(QQQ 그날 종가까지)에 그날 → 다음 날 수익을 넣는다. 60일 미만은 참고."""
    reg = rg.classify(qqq_close)
    idx = values.index
    r_s = values.pct_change().iloc[1:]
    r_b = bench.reindex(idx).pct_change().iloc[1:]
    g = reg.reindex(idx).shift(1).iloc[1:]  # 전날 국면(양수 shift — 미래 정보 아님)
    out = {}
    for k in (rg.UP, rg.DOWN, rg.CRASH):
        m = g == k
        n = int(m.sum())
        if n == 0:
            continue
        out[k] = {"days": n, "cum_pct": float((np.prod(1 + r_s[m]) - 1) * 100),
                  "qqq_cum_pct": float((np.prod(1 + r_b[m].fillna(0.0)) - 1) * 100),
                  "inconclusive": n < MIN_REGIME_DAYS}
    return out


def strategy_row(st: Strategy, daily: pd.DataFrame | None, bench_daily: pd.DataFrame, trades: pd.DataFrame | None,
                 qqq_close: pd.Series, month: pd.Period, quarter_end: bool) -> dict:
    """전략 하나의 성적 (month 말까지 기록). daily·bench_daily: 인덱스 날짜, 열 value·value_after_tax_liq."""
    row = {"no": st.no, "name": st.label or st.name, "status": "ok", "notes": [], "proposal": None}
    end = month.to_timestamp(how="end")
    if daily is None or daily.empty or daily.index.min() > end:
        row["status"] = "기록 없음"
        return row
    d = daily[daily.index <= end]
    b = bench_daily.reindex(d.index)
    if b["value_after_tax_liq"].isna().any():
        row["status"] = "QQQ 기록 없는 날 있음"
        return row
    vs, vq = d["value_after_tax_liq"].astype(float), b["value_after_tax_liq"].astype(float)
    ms, mq = _month_ends(vs), _month_ends(vq)
    base_s, base_q = float(vs.iloc[0]), float(vq.iloc[0])
    prev_s = ms.shift(1).fillna(base_s)
    prev_q = mq.shift(1).fillna(base_q)
    rs, rq = ms / prev_s - 1.0, mq / prev_q - 1.0
    first, last = d.index.min(), d.index.max()
    row.update({
        "start": first.date().isoformat(), "last": last.date().isoformat(),
        "month_pct": float(rs.iloc[-1] * 100) if ms.index[-1] == month else None,
        "month_vs_qqq_pp": float((rs.iloc[-1] - rq.iloc[-1]) * 100) if ms.index[-1] == month else None,
        "cum_pct": float((vs.iloc[-1] / base_s - 1) * 100),
        "cum_vs_qqq_pp": float((vs.iloc[-1] / base_s - vq.iloc[-1] / base_q) * 100),
        "months_won": int((rs > rq).sum()), "months_total": int(len(rs)),
        "mdd_pct": _mdd(d["value"]), "qqq_mdd_pct": _mdd(b["value"]),
        "record_months": _months_between(first, last),
        "short_record": first.to_period("M") == month,
    })
    days = (last - first).days
    if days > 0:
        ann_s = (vs.iloc[-1] / base_s) ** (365.25 / days) - 1
        ann_q = (vq.iloc[-1] / base_q) ** (365.25 / days) - 1
        row["ann_vs_qqq_pp"] = float((ann_s - ann_q) * 100)
    else:
        row["ann_vs_qqq_pp"] = None
    t = trades if trades is not None else pd.DataFrame(columns=["fill_date", "commission"])
    t = t[pd.to_datetime(t["fill_date"]) <= end] if len(t) else t
    in_month = t[pd.to_datetime(t["fill_date"]).dt.to_period("M") == month] if len(t) else t
    row.update({"trades_month": int(len(in_month)), "cost_month": float(in_month["commission"].sum()) if len(in_month) else 0.0,
                "trades_total": int(len(t)), "cost_total": float(t["commission"].sum()) if len(t) else 0.0})
    row["regimes"] = regime_perf(d["value"].astype(float), b["value"].astype(float), qqq_close)
    if row["short_record"]:
        row["notes"].append("기록 짧음")
    row["proposal"] = proposal(row, quarter_end)
    return row


def proposal(row: dict, quarter_end: bool) -> str | None:
    """조기 탈락·실전 검토 제안 (제안만). 경계값은 포함(이하·이상)."""
    if row.get("status") != "ok":
        return None
    if (quarter_end and row["no"] not in BASELINE_NOS and row["record_months"] >= STOP_MONTHS
            and row["cum_vs_qqq_pp"] <= STOP_GAP_PP):
        return "중단 제안"
    if (row["no"] != 1 and row["record_months"] >= REVIEW_MONTHS and row["ann_vs_qqq_pp"] is not None
            and row["ann_vs_qqq_pp"] >= REVIEW_ANN_PP and row["months_total"] > 0
            and row["months_won"] / row["months_total"] >= REVIEW_WIN_RATIO
            and row["mdd_pct"] - row["qqq_mdd_pct"] >= -REVIEW_MDD_GAP_PP):
        return "실제 돈 일부 검토"
    return None


def build(strats: list[Strategy], daily: dict[str, pd.DataFrame], trades: dict[str, pd.DataFrame],
          qqq_close: pd.Series, month: pd.Period) -> dict:
    """성적표 한 장. month: 성적을 매기는 달(보통 지난달). 1번 장부가 QQQ 기준선."""
    bench = daily.get(strats[0].name)
    if bench is None or bench.empty:
        raise ValueError(f"기준선 장부 「{strats[0].name}」 기록이 없습니다")
    qe = month.month in QUARTER_END_MONTHS
    rows = [strategy_row(s, daily.get(s.name), bench, trades.get(s.name), qqq_close, month, qe) for s in strats]
    return {"month": str(month), "quarter_end": qe, "regime_ratio": regime_month_ratio(qqq_close, month), "rows": rows,
            "proposals": [(r["no"], r["name"], r["proposal"]) for r in rows if r.get("proposal")]}


# ── 글 ───────────────────────────────────────────────────────────────────


def _pp(x, d=1) -> str:
    return "-" if x is None else f"{x:+.{d}f}%p"


def briefing_lines(sc: dict) -> list[str]:
    """브리핑 접기용 짧은 줄: 전략별 「이름 · 누적 QQQ 대비 %p · 이긴 달 수」. 이모지 없음."""
    out = []
    for r in sc["rows"]:
        if r["status"] != "ok":
            out.append(f"{r['no']}. {r['name']} · {r['status']}")
            continue
        tail = "".join(f" · {n}" for n in r["notes"]) + (f" · {r['proposal']}" if r["proposal"] else "")
        out.append(f"{r['no']}. {r['name']} · 누적 QQQ 대비 {_pp(r['cum_vs_qqq_pp'])} · 이긴 달 {r['months_won']}/{r['months_total']}{tail}")
    return out


def render_markdown(sc: dict) -> str:
    """성적표 전체(md) — 저장용."""
    reg = " · ".join(f"{k} {v * 100:.0f}%" for k, v in sc["regime_ratio"].items()) or "-"
    L = [f"# 가상 장부 월간 성적표 {sc['month']}", "",
         "- 관찰 전용 가상 장부(실제 주문 아님). 비교 곡선: 그날 전부 팔았다면 세후 가치(전략·QQQ 모두). QQQ = 1번 장부",
         f"- {sc['month']} 국면(QQQ 거래일 비율): {reg}", "",
         "| # | 전략 | 이번 달 QQQ 대비 | 누적 QQQ 대비 | 이긴 달 | 최대 낙폭(QQQ) | 매매(이번 달/누적) | 비용 $(이번 달/누적) | 비고 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in sc["rows"]:
        if r["status"] != "ok":
            L.append(f"| {r['no']} | {r['name']} | - | - | - | - | - | - | {r['status']} |")
            continue
        note = ", ".join(r["notes"] + ([r["proposal"]] if r["proposal"] else []))
        L.append(f"| {r['no']} | {r['name']} | {_pp(r['month_vs_qqq_pp'], 2)} | {_pp(r['cum_vs_qqq_pp'], 2)} | "
                 f"{r['months_won']}/{r['months_total']} | {r['mdd_pct']:.1f}% ({r['qqq_mdd_pct']:.1f}%) | "
                 f"{r['trades_month']}/{r['trades_total']} | {r['cost_month']:,.0f}/{r['cost_total']:,.0f} | {note} |")
    L += ["", "## 국면별 누적 (그날 국면 → 다음 날 수익, 60거래일 미만은 참고(INCONCLUSIVE))", "",
          "| # | 전략 | 국면 | 일수 | 전략 누적 | QQQ 누적 | |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in sc["rows"]:
        for k, v in (r.get("regimes") or {}).items():
            L.append(f"| {r['no']} | {r['name']} | {k} | {v['days']} | {v['cum_pct']:+.2f}% | {v['qqq_cum_pct']:+.2f}% | "
                     f"{'참고(INCONCLUSIVE)' if v['inconclusive'] else ''} |")
    L += ["", "## 제안 (제안만 — 장부를 멈추거나 실제 주문을 내지 않는다)", ""]
    L += [f"- {no}. {name}: {p}" for no, name, p in sc["proposals"]] or ["- 없음"]
    return "\n".join(L) + "\n"


def sheet_rows(sc_: dict, recorded_at: str) -> list[list]:
    """월간성적표 탭에 덧붙일 행 (data.virtual_books_sheet.SCORECARD_HEADER 순서, 전략마다 한 줄)."""
    reg = " · ".join(f"{k} {v * 100:.0f}%" for k, v in sc_["regime_ratio"].items())
    out = []
    for r in sc_["rows"]:
        if r["status"] != "ok":
            out.append([sc_["month"], r["no"], r["name"], "", "", "", "", "", "", "", "", "", "", reg, r["status"], "", recorded_at])
            continue
        rnd = lambda x, d=2: "" if x is None else round(float(x), d)  # noqa: E731
        out.append([sc_["month"], r["no"], r["name"], rnd(r["month_vs_qqq_pp"]), rnd(r["cum_vs_qqq_pp"]), r["months_won"],
                    r["months_total"], rnd(r["mdd_pct"]), rnd(r["qqq_mdd_pct"]), r["trades_month"], r["trades_total"],
                    rnd(r["cost_month"]), rnd(r["cost_total"]), reg, ", ".join(r["notes"]), r["proposal"] or "", recorded_at])
    return out
