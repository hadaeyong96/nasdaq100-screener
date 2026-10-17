"""아침 편지 짧은 판 (2026-10-17) — 할 일·시장 한눈에·실험 중인 전략 문구 (순수 함수, 네트워크·파일 없음).

보여 주는 순서·문구만 만든다. 장부·신호·추천 계산은 하지 않고 이미 계산된 결과(summary, 가상 장부 결과)를 읽기만 한다.
판(edition): "personal"(개인 채팅) | "group"(투자클럽 단체방 — 개인 항목·티커 없음).
이모지는 쓰지 않는다.
"""

from __future__ import annotations

import pandas as pd

from notify import virtual_books as vbtext

PERSONAL, GROUP = "personal", "group"
TODO_NONE = "없음 — QQQM 적립만 그대로"
STRATEGY_HEADER = "[실험 중인 전략] 가상 1억 · {start} 시작 · 매수 추천 아님"
GLANCE_HEADER = "[시장 한눈에]"
FAMILIES = ("hold", "trend", "picks", "rebalance")


def _md(ts) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.month}/{ts.day}"


# ── ② 오늘 할 일 ────────────────────────────────────────────────────────────


DEC_TAX_TODO = "올해 세금 없는 매도 한도 확인"


def december_items(summary: dict, cfg: dict) -> list[str]:
    """12월 공제 안내(개인판만): 표시일(12월 첫 거래일·15일 이후 첫 거래일)에 engine.daily가 summary["dec_tax"]를 둔 날만."""
    return [DEC_TAX_TODO] if summary.get("dec_tax") else []


def todo_items(summary: dict, cfg: dict, edition: str = PERSONAL) -> list[str]:
    """오늘 할 일 항목(해당 있는 것만, 짧게). 단체방판은 개인 항목(월간 성적표·12월 공제)을 뺀다."""
    items = []
    vb = summary.get("vbooks") or {}
    if vb.get("switch_today"):
        items.append(f"추세 전략 전환 신호(가상) — {_md(vb['switch_next_day'])} 체결" if vb.get("switch_next_day")
                     else "추세 전략 전환 신호(가상)")
    if edition == PERSONAL:
        sc = summary.get("scorecard")
        if sc:
            items.append(f"월간 성적표 보기({int(str(sc['month'])[-2:])}월)")
        items.extend(december_items(summary, cfg))
    return items


def todo_line(summary: dict, cfg: dict, edition: str = PERSONAL) -> str:
    """"오늘 할 일: …" 한 줄. 없으면 "오늘 할 일: 없음 — QQQM 적립만 그대로"."""
    items = todo_items(summary, cfg, edition)
    return "오늘 할 일: " + (" · ".join(items) if items else TODO_NONE)


# ── ③ 시장 한눈에 ───────────────────────────────────────────────────────────


def glance_lines(qqq_close: pd.Series | None, macro_rows: list[dict]) -> list[str]:
    """3줄 이내. 1줄 QQQ 하루·1주·200일선 대비·국면, 2줄 공포·탐욕·10년 금리·원/달러, 3줄 주의·위험 지표 이름(없으면 생략).

    입력: qqq_close(QQQ 확정 종가, 날짜 오름차순 — 없으면 1줄 생략), macro_rows(engine.daily._build_macro_rows 행)
    """
    lines = []
    if qqq_close is not None and len(qqq_close.dropna()) >= 6:
        from factory import regimes

        q = qqq_close.dropna().astype(float)
        parts = [f"QQQ 하루 {(q.iloc[-1] / q.iloc[-2] - 1) * 100:+.1f}%", f"1주 {(q.iloc[-1] / q.iloc[-6] - 1) * 100:+.1f}%"]
        if len(q) >= 200:
            sma = q.iloc[-200:].mean()
            parts.append(f"200일선 대비 {(q.iloc[-1] / sma - 1) * 100:+.1f}%")
        regime = regimes.classify(q).iloc[-1]
        if regime != regimes.UNKNOWN:
            parts.append(regime)
        lines.append(" · ".join(parts))
    by = {r.get("slug"): r for r in macro_rows or []}
    second = []
    fg = by.get("fear-greed")
    if fg:
        zone = (fg.get("badge") or {}).get("zone")
        second.append(f"공포·탐욕 {fg['value']}" + (f"({zone})" if zone else ""))
    elif by.get("vix"):
        second.append(f"VIX {by['vix']['value']:.1f}")
    if by.get("dgs10"):
        second.append(f"미 10년 금리 {by['dgs10']['value']:.2f}%")
    if by.get("fx"):
        second.append(f"원/달러 {by['fx']['value']:,.0f}원")
    if second:
        lines.append(" · ".join(second))
    danger = [_danger_text(r) for r in macro_rows or [] if (r.get("badge") or {}).get("label") == "위험"]
    if danger:  # 위험(빨강)만 짧게. 주의(노랑)는 그래프 접기에서 본다(2026-10-17)
        lines.append("위험: " + " · ".join(danger))
    return lines


def _danger_text(r: dict) -> str:
    """위험 단계 지표 하나 → 짧은 이름 + 값. 예: "공포 극단", "10년 금리 5.1%", "기준금리 인상"."""
    slug, v = r.get("slug"), r.get("value")
    zone = (r.get("badge") or {}).get("zone") or ""
    if slug == "fear-greed":
        return {"극단적 공포": "공포 극단", "극단적 탐욕": "탐욕 극단"}.get(zone, f"공포·탐욕 {v}")
    if slug == "vix":
        return f"VIX {v:.1f}"
    if slug == "dgs10":
        return f"10년 금리 {v:.1f}%"
    if slug == "t10y2y":
        return f"금리차 {v:+.2f}%p"
    if slug == "hy":
        return f"하이일드 스프레드 {v:.2f}%"
    if slug == "fed":
        return "기준금리 인상"
    if slug == "fx":
        return f"원/달러 {v:,.0f}원"
    return str(r.get("name", slug))


# ── ④ 실험 중인 전략 ─────────────────────────────────────────────────────────


def _family_of(cfg: dict) -> dict[str, str]:
    vb = cfg.get("virtual_books") or {}
    fam = {}
    for b in vb.get("books", []):
        fam[b["name"]] = "trend" if b.get("kind") == "sma_switch" else "hold"
    for b in (vb.get("picks") or {}).get("books", []):
        fam[b["name"]] = "picks"
    for key in ("momentum", "accel"):
        if (vb.get(key) or {}).get("name"):
            fam[vb[key]["name"]] = "rebalance"
    return fam


def letter_books(cfg: dict, base_date, summaries: list[dict], book_trades: dict[str, list[dict]],
                 extra: list[dict], bench_daily: pd.DataFrame | None, waiting: dict[str, pd.Timestamp] | None = None,
                 enabled: list[str] | None = None) -> list[dict]:
    """가상 장부 결과 → 편지·표용 장부 목록(config 순서). 계산은 하지 않고 이미 나온 값만 옮긴다.

    입력: summaries(1~5번 core.virtual_books.summarize_book 결과), book_trades({1~5번 이름: 매매 dict 목록}),
         extra(6~9번 engine.picks_books extra), bench_daily(QQQ 전략 일별 — 늦게 시작한 장부의 QQQ 대비용),
         waiting({이름: 시작 예정일} — 아직 시작 전 장부), enabled(편지에 낼 장부 이름 순서, 없으면 config 순서 전체)
    출력: [{name, label, family, status(ok|waiting|missing), cum_return_pct, vs_qqq_pp, holding, n_positions,
            trades_today[(구분, 티커)], start, value, mdd_pct, switches}]
    """
    base = pd.Timestamp(base_date).normalize()
    fam = _family_of(cfg)
    order = enabled or list(fam)
    vb = cfg.get("virtual_books") or {}
    bench_name = vb.get("benchmark_book")
    by_sum = {s["book"]: s for s in summaries or []}
    by_extra = {x["name"]: x for x in extra or []}
    bench_cum = by_sum.get(bench_name, {}).get("cum_return_pct")
    out = []
    for name in order:
        row = {"name": name, "label": vbtext.display_name(cfg, name), "family": fam.get(name, "hold"),
               "status": "missing", "cum_return_pct": None, "vs_qqq_pp": None, "holding": "", "n_positions": None,
               "trades_today": [], "start": None, "value": None, "mdd_pct": None, "switches": None}
        if name in by_sum:
            s = by_sum[name]
            row.update(status="ok", cum_return_pct=s["cum_return_pct"], holding=s.get("holding", ""),
                       start=pd.Timestamp(s["start"]) if s.get("start") is not None else None, value=s.get("value"),
                       mdd_pct=s.get("mdd_pct"), switches=s.get("switches"))
            if name != bench_name and bench_cum is not None:
                row["vs_qqq_pp"] = s["cum_return_pct"] - bench_cum
            trades = book_trades.get(name, [])
        elif name in by_extra:
            x = by_extra[name]
            s = x["summary"]
            row.update(status="ok", cum_return_pct=s["cum_return_pct"], start=pd.Timestamp(x["start"]), value=s["value"],
                       mdd_pct=s["mdd_pct"], n_positions=len(x.get("positions") or {}),
                       holding=", ".join(sorted(x.get("positions") or {})))
            if bench_daily is not None and not bench_daily.empty:
                bv = bench_daily["value"]
                at = bv[bv.index <= row["start"]]
                if len(at):
                    row["vs_qqq_pp"] = s["cum_return_pct"] - (float(bv.iloc[-1]) / float(at.iloc[-1]) - 1) * 100
            trades = x.get("trades") or []
        elif waiting and name in waiting:
            row.update(status="waiting", start=pd.Timestamp(waiting[name]))
            trades = []
        else:
            trades = []
        row["trades_today"] = [(t["side"], t["ticker"]) for t in trades
                               if t.get("fill_date") is not None and pd.Timestamp(t["fill_date"]).normalize() == base
                               and t.get("side") in ("매수", "매도")]
        out.append(row)
    return out


def _status(b: dict, cfg: dict) -> tuple[str, str]:
    """(묶는 데 쓰는 상태 키, 상태 문구). 같은 키끼리 한 줄로 묶는다(2026-10-17 다듬기)."""
    if b["status"] == "waiting":
        return f"wait:{b['start'].date()}", f"{_md(b['start'])} 시작"
    if b["status"] == "missing":
        return "missing", "오늘 계산 못함"
    if b["family"] == "trend":
        defensive = (cfg.get("virtual_books") or {}).get("defensive_ticker", "SGOV")
        if b["holding"] == defensive:
            return "down", "하락 추세라 단기 국채로 피신"
        return "up", "상승 추세라 {holdings} 보유"
    if b["family"] in ("picks", "rebalance"):
        n = b["n_positions"] or 0
        return ("cash", "살 종목이 없어 현금 보관") if n == 0 else (f"n:{n}", f"{n}종목 보유")
    return "hold", "그냥 보유"


def _ret(b: dict) -> str:
    """누적 수익률 한 칸: +1.2% / −0.1% / 0.0% (음수는 긴 빼기표)."""
    v = round(b["cum_return_pct"], 1) + 0.0
    return "0.0%" if v == 0 else (f"+{v:.1f}%" if v > 0 else f"−{abs(v):.1f}%")


def join_names(labels: list[str]) -> str:
    """보여 주는 이름 묶기: "QLD 추세 전략"+"TQQQ 추세 전략" → "QLD·TQQQ 추세 전략",
    "기술지표 전략"+"볼린저 전략"+"한 달 가속 전략" → "기술지표·볼린저·한 달 가속 전략"."""
    if len(labels) == 1:
        return labels[0]
    words = [x[:-3].split() if x.endswith(" 전략") else x.split() for x in labels]
    common = []
    while all(len(w) > len(common) + 1 for w in words) and len({w[-1 - len(common)] for w in words}) == 1:
        common.insert(0, words[0][-1 - len(common)])
    heads = [" ".join(w[: len(w) - len(common)]) for w in words]
    return "·".join(heads) + (" " + " ".join(common) if common else "") + " 전략"


def _trades_line(books: list[dict], edition: str) -> str:
    """칸 맨 끝 한 줄: 개인판 "오늘 가상 매수: 전략명 티커 · …", 단체방판 "오늘 가상 거래 n건". 없으면 "오늘 거래 없음"."""
    if edition == GROUP:
        n = sum(len(b["trades_today"]) for b in books)
        return f"오늘 가상 거래 {n}건" if n else "오늘 거래 없음"
    parts = []
    for side in ("매수", "매도"):
        per = [(b["label"], list(dict.fromkeys(t for s, t in b["trades_today"] if s == side))) for b in books]
        per = [f"{label} {', '.join(ts)}" for label, ts in per if ts]
        if per:
            parts.append(f"오늘 가상 {side}: " + " · ".join(per))
    return " / ".join(parts) if parts else "오늘 거래 없음"


def strategy_lines(books: list[dict], cfg: dict, edition: str = PERSONAL) -> list[str]:
    """같은 상태인 전략을 한 줄로: "QLD·TQQQ 추세 전략: 상승 추세라 QLD·TQQQ 보유 (−5.3% · −8.0%)".
    QQQ 대비는 넣지 않는다(표로 보기 접기에만). 마지막 줄은 그날 가상 매매 한 줄."""
    groups: list[dict] = []
    for b in books:
        key, text = _status(b, cfg)
        g = next((x for x in groups if x["key"] == key), None)
        if g is None:
            groups.append({"key": key, "text": text, "books": [b]})
        else:
            g["books"].append(b)
    lines = []
    for g in groups:
        bs = g["books"]
        text = g["text"].format(holdings="·".join(dict.fromkeys(b["holding"] for b in bs)))
        line = f"{join_names([b['label'] for b in bs])}: {text}"
        if all(b["status"] == "ok" and b["cum_return_pct"] is not None for b in bs):
            line += " (" + " · ".join(_ret(b) for b in bs) + ")"
        lines.append(line)
    lines.append(_trades_line(books, edition))
    return lines


def strategy_header(books: list[dict], start=None) -> str:
    """"[실험 중인 전략] 가상 1억 · 10/14 시작 · 매수 추천 아님"."""
    first = start or min((b["start"] for b in books if b.get("start") is not None and b["status"] == "ok"), default=None)
    return STRATEGY_HEADER.format(start=_md(first) if first is not None else "-")


def _holding_cell(b: dict, cfg: dict, edition: str) -> str:
    """보유 칸. 단체방판은 종목 티커 대신 "n종목"·"단기 국채"(전략 이름에 있는 QQQ·QLD·TQQQ만 그대로)."""
    ok = b["status"] == "ok"
    defensive = (cfg.get("virtual_books") or {}).get("defensive_ticker", "SGOV")
    if b["family"] in ("picks", "rebalance"):
        n = b["n_positions"] or 0
        if not ok:
            return ""
        if n == 0:
            return "현금" if edition == GROUP else f"현금({defensive})"
        return f"{n}종목" if edition == GROUP else b["holding"]
    if edition == GROUP and b["holding"] == defensive:
        return "단기 국채"
    return b["holding"]


def table_rows(books: list[dict], cfg: dict | None = None, edition: str = PERSONAL) -> tuple[list[str], list[list[str]]]:
    """접은 표(9줄): (머리글, 행). 모든 값이 0(또는 비어 있음)인 열은 뺀다(0만 있는 칸 숨김)."""
    cfg = cfg or {}
    head = ["전략", "총자산($)", "누적 수익", "QQQ 대비", "최대 낙폭", "전환 횟수", "보유"]
    rows = []
    for b in books:
        ok = b["status"] == "ok"
        rows.append([
            b["label"], f"{b['value']:,.0f}" if ok and b["value"] is not None else "",
            f"{b['cum_return_pct']:+.2f}%" if ok else ("" if b["status"] == "missing" else f"{_md(b['start'])} 시작"),
            f"{b['vs_qqq_pp']:+.2f}%p" if ok and b["vs_qqq_pp"] is not None else "",
            f"{b['mdd_pct']:.1f}%" if ok and b["mdd_pct"] is not None else "",
            str(b["switches"]) if ok and isinstance(b["switches"], int) else "",
            _holding_cell(b, cfg, edition),
        ])
    keep = [i for i in range(len(head)) if i == 0 or any(_nonzero(r[i]) for r in rows)]
    return [head[i] for i in keep], [[r[i] for i in keep] for r in rows]


def _nonzero(cell: str) -> bool:
    s = cell.replace("%p", "").replace("%", "").replace(",", "").replace("+", "").replace("-", "").strip()
    if s == "":
        return False
    try:
        return float(s) != 0.0
    except ValueError:
        return True
