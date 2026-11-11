"""실적급등 추천 편지 v2 그래프 (개인판 보고서 전용, 2026-11-11 사용자 요청 — 그래프 중심).

render       ① 한눈에 비교: 종목마다 매수일부터 "QQQ보다 얼마나 앞섰나"(QQQ 대비 초과 %p) 선. 가운데 0선 = QQQ.
render_card  ② 종목 카드: QQQ 캔들(왼쪽 축, 달러) + 종목 종가 선(오른쪽 축, 달러). 두 축은 매수일 시가에서 같은 높이.
입력은 engine.pead_watch.chart_data 결과(lines·cards·qqq_pct). 계산 없음 — 그리기만. 단체방판에는 넣지 않는다(종목 이름).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

WIDTH, HEIGHT, DPI = 9, 5.6, 110
CARD_W, CARD_H = 9, 4.3
INK, SUB, QQQ_C, GRID = "#1F2328", "#6B7280", "#8C95A1", "#EEF0F3"
UP, DOWN, DONE, PURPLE = "#E5484D", "#3B82F6", "#B8BEC6", "#7C3AED"
BG_UP, BG_DOWN = "#FDF1F1", "#EEF3FD"
DOT_EARN, DOT_BUY, DOT_SELL = "#DC2626", "#2563EB", "#FACC15"  # 카드 점: 실적 발표(반응일)·매수일·매도 예정일 (2026-11-12)
NAME_MAX = 8  # 이름표 글자 수 — 넘으면 줄인다
LABEL_GAP = 0.105  # 이름표(두 줄) 세로 최소 간격, 축 높이 비율
LABEL_SPACE = 0.92  # 이름표가 쓸 수 있는 세로 폭(축 높이 비율) — 넘치면 한 줄 이름표
AXES_PT = HEIGHT * 72 * (0.80 - 0.08)  # 그림 ①의 축 높이(포인트) — 한 줄 이름표 글자 크기 계산용


def short_name(name: str, limit: int = NAME_MAX) -> str:
    """긴 이름 줄이기: 띄어쓰기가 있으면 첫 낱말("어플라이드 머티어리얼즈" → "어플라이드"), 없으면 앞 limit−1글자 + "…"."""
    if len(name) <= limit:
        return name
    head = name.split(" ")[0]
    return head if " " in name and len(head) >= 2 else name[:limit - 1] + "…"


def spread_labels(ys: list[float], gap: float, lo: float, hi: float) -> list[float]:
    """세로 이름표 위치: 원래 값 순서를 지키며 서로 gap 이상 떨어지게 위로 민다. 위 끝을 넘으면 전체를 아래로. 출력 순서 = 입력 순서."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    pos = [max(ys[i], lo) for i in order]
    for k in range(1, len(pos)):
        pos[k] = max(pos[k], pos[k - 1] + gap)
    over = pos[-1] - hi if pos else 0
    if over > 0:
        pos = [p - over for p in pos]
        for k in range(len(pos) - 2, -1, -1):
            pos[k] = min(pos[k], pos[k + 1] - gap)
    out = [0.0] * len(ys)
    for k, i in enumerate(order):
        out[i] = pos[k]
    return out


def _setup():
    import matplotlib

    from notify.curve_chart import korean_font

    font = korean_font()
    matplotlib.rcParams["axes.unicode_minus"] = False
    if font:
        matplotlib.rcParams["font.family"] = [font, "DejaVu Sans"]


def _month_ticks(ax, dates: pd.DatetimeIndex, xs=None) -> None:
    """x축 달마다 "7월"(같은 달 한 번). xs가 있으면 정수 위치(카드), 없으면 날짜."""
    seen, ticks, labels = set(), [], []
    for k, d in enumerate(dates):
        if (d.year, d.month) in seen:
            continue
        seen.add((d.year, d.month))
        if k == 0 and d.day > 7:  # 달 중간에 시작하면 첫 눈금은 다음 달부터
            continue
        ticks.append(xs[k] if xs is not None else d)
        labels.append(f"{d.month}월")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def _week_ticks(ax, dates: pd.DatetimeIndex) -> None:
    """x축 주마다 "11/2"(그 주 첫 거래일)."""
    ticks, labels, last = [], [], None
    for k, d in enumerate(dates):
        wk = d.isocalendar()[:2]
        if wk != last:
            last = wk
            ticks.append(k)
            labels.append(f"{d.month}/{d.day}")
    if len(ticks) > 1 and ticks[1] - ticks[0] < 3:  # 짧은 첫 주는 글자가 겹친다
        ticks, labels = ticks[1:], labels[1:]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def _save(fig, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    return out_path


def render(chart: dict | None, out_path: Path, as_of: str | None = None) -> Path | None:
    """① 한눈에 비교 PNG. 선이 없으면 None."""
    if not chart or not chart.get("lines"):
        return None
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    _setup()
    fig = Figure(figsize=(WIDTH, HEIGHT), dpi=DPI, facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.74, top=0.80, bottom=0.08)
    ax = fig.add_subplot()
    fig.text(0.02, 0.955, "QQQ보다 얼마나 앞섰나", fontsize=17, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.885, "가운데 선 = QQQ · 위 = 앞섬 · 아래 = 뒤짐  ○ 매수 ◇ 60일 점검 ● 오늘", fontsize=11, color=SUB, va="top")
    if chart.get("qqq_pct") is not None:
        fig.text(0.98, 0.955, f"같은 기간 QQQ {chart['qqq_pct']:+.1f}%", fontsize=11, color=QQQ_C, fontweight="bold",
                 ha="right", va="top")
    lines = sorted(chart["lines"], key=lambda ln: ln["status"] == "open")  # 끝난 종목을 먼저(아래에) 그린다
    allv, alld = [0.0], []
    for ln in lines:
        allv += [v for v in ln["excess"] if v is not None and v == v]
        alld += list(pd.to_datetime(ln["dates"]))
    lo, hi = min(allv), max(allv)
    pad = max((hi - lo) * 0.12, 1.0)
    lo, hi = lo - pad, hi + pad
    ax.set_ylim(lo, hi)
    ax.axhspan(0, hi, color=BG_UP, zorder=0, lw=0)
    ax.axhspan(lo, 0, color=BG_DOWN, zorder=0, lw=0)
    ax.axhline(0, color=QQQ_C, lw=2.4, zorder=2)
    ends = []
    for ln in lines:
        open_ = ln["status"] == "open"
        last = ln["excess"][-1]
        col = (UP if (last or 0) >= 0 else DOWN) if open_ else DONE
        x = pd.to_datetime(ln["dates"])
        y = ln["excess"]
        z = 4 if open_ else 3
        ax.plot(x, y, color=col, lw=2.3 if open_ else 1.3, zorder=z, solid_capstyle="round")
        ax.scatter([x[0]], [y[0]], s=40, facecolor="white", edgecolor=col, linewidth=1.6, zorder=z + 1)
        if ln.get("check_at") is not None and ln["check_at"] < len(x):
            k = ln["check_at"]
            ax.scatter([x[k]], [y[k]], s=46, marker="D", facecolor="white", edgecolor=col, linewidth=1.5, zorder=z + 1)
        ax.scatter([x[-1]], [y[-1]], s=46, color=col, zorder=z + 1)
        ends.append({"y": last, "col": col if open_ else SUB, "open": open_,
                     "l1": f"{short_name(ln['name'])} {last:+.1f}%p" + ("" if open_ else " (끝)"),
                     "l2": f"주가 {ln['values'][-1]:+.1f}% · {ln['held']}/{ln['hold']}일"})
    # 이름표가 많아 두 줄로 다 못 넣으면 한 줄("회사명 +x%p")로, 그래도 많으면 글자를 줄인다(보유 20~30개도 잘리지 않게)
    two = len(ends) * LABEL_GAP <= LABEL_SPACE
    gap = LABEL_GAP if two else LABEL_SPACE / len(ends)
    fs = 11.5 if two else max(6.5, min(11.5, gap * AXES_PT * 0.8))
    ys = spread_labels([e["y"] for e in ends], (hi - lo) * gap, lo + (hi - lo) * 0.04, hi - (hi - lo) * 0.04)
    for e, yy in zip(ends, ys):
        ax.annotate(e["l1"], xy=(1.02, yy), xycoords=("axes fraction", "data"), fontsize=fs, color=e["col"],
                    va="bottom" if two else "center", fontweight="bold" if e["open"] else "normal", annotation_clip=False)
        if two:
            ax.annotate(e["l2"], xy=(1.02, yy), xycoords=("axes fraction", "data"), fontsize=9.5, color=SUB, va="top",
                        annotation_clip=False)
    ticks = [t for t in MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi) if lo <= t <= hi]
    if 0 not in ticks:
        ticks.append(0.0)
    ticks = sorted(ticks)
    ax.set_yticks(ticks)
    ax.set_yticklabels(["QQQ" if t == 0 else f"{t:+g}%p" for t in ticks])
    for lab in ax.get_yticklabels():
        if lab.get_text() == "QQQ":
            lab.set_fontweight("bold")
            lab.set_color(QQQ_C)
        else:
            lab.set_color(SUB)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, labelsize=10, colors=SUB)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    dmin, dmax = min(alld), max(alld)
    ax.set_xlim(dmin - pd.Timedelta(days=2), dmax + pd.Timedelta(days=2))
    _month_ticks(ax, pd.bdate_range(dmin, dmax))
    return _save(fig, out_path)


def sell_note(card: dict, next_day: str | None = None, today_kst=None) -> str:
    """카드 오른쪽 위 둘째 줄: "n/90일 · 매도 예정 YYYY-MM-DD". 매도일이 다음 거래일이면 "오늘 밤 장 마감 때 매도"
    (편지 받는 날 ≠ 다음 거래일이면 "M/D(요일) 밤 …"), 그다음 거래일이면 "내일 밤 장 마감 때 매도"."""
    from notify.pead_letter import md

    head = f"{card['held']}/{card['hold']}일"
    same = next_day is None or today_kst is None or pd.Timestamp(next_day).normalize() == pd.Timestamp(today_kst).normalize()
    if card["sell_in"] == 1:
        return f"{head} · {'오늘 밤' if same else md(next_day) + ' 밤'} 장 마감 때 매도"
    if card["sell_in"] == 2 and same:
        return f"{head} · 내일 밤 장 마감 때 매도"
    return f"{head} · 매도 예정 {card['plan_day']}"


def card_tag(card: dict) -> str:
    """카드 제목 아래 태그: 새 추천이면 반응일 숫자, 아니면 "보유 중 · 소속"."""
    if card["kind"] == "new" and card.get("sig_ret") is not None:
        return (f"새 추천 · 반응일 {card['sig_ret'] * 100:+.1f}% · QQQ 대비 {card['sig_excess'] * 100:+.1f}%p"
                f" · 거래량 {card['sig_vol']:.1f}배")
    if card["kind"] == "new":
        return f"새 추천 · {card.get('label') or ''}".rstrip(" ·")
    return f"보유 중 · {card.get('label') or ''}".rstrip(" ·")


def render_card(card: dict, out_path: Path, next_day: str | None = None, today_kst=None) -> Path:
    """② 종목 카드 PNG 한 장. 둥근 점(2026-11-12): 빨강 = 실적 발표(반응일 d0 종가), 파랑 = 매수(매수일 시가),
    노랑 = 매도 예정일(아직 가격이 없어 바닥선, 가로축을 그날까지 늘림). 점 설명은 그래프 왼쪽 위 색인."""
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    _setup()
    fig = Figure(figsize=(CARD_W, CARD_H), dpi=DPI, facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.90, top=0.72, bottom=0.10)
    ax = fig.add_subplot()
    ax2 = ax.twinx()
    dates = pd.to_datetime(card["dates"])
    n = len(dates)
    xs = np.arange(n)
    fig.text(0.02, 0.96, f"{card['name']} ({card['ticker']})", fontsize=16, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.865, card_tag(card), fontsize=10.5, color=SUB, va="top")
    vq = card.get("vs_qqq")
    if vq is not None:
        fig.text(0.98, 0.96, f"QQQ 대비 {vq:+.1f}%p", fontsize=14, fontweight="bold", color=PURPLE if vq >= 0 else SUB,
                 ha="right", va="top")
    fig.text(0.98, 0.865, sell_note(card, next_day, today_kst), fontsize=10.5, color=SUB, ha="right", va="top")

    q = card["qqq"]
    k = card["buy_px"] / card["q_buy"]  # 종목 달러 = QQQ 달러 × k (매수일 시가에서 같은 높이)
    vals = [v for v in q["low"] + q["high"] if v is not None]
    vals += [c / k for c in card["close"] if c is not None]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08 or hi * 0.02
    lo, hi = lo - pad, hi + pad
    ax.set_ylim(lo, hi)
    ax2.set_ylim(lo * k, hi * k)
    wbar = 0.6
    for i in range(n):
        o, h, l_, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
        if None in (o, h, l_, c):
            continue
        col = UP if c >= o else DOWN
        ax.vlines(i, l_, h, color=col, lw=0.8, zorder=2)
        ax.add_patch(Rectangle((i - wbar / 2, min(o, c)), wbar, max(abs(c - o), (hi - lo) * 0.002), facecolor=col,
                               edgecolor=col, lw=0.5, zorder=2))
    qc = np.array([np.nan if v is None else v for v in q["close"]], dtype=float)
    ax.plot(xs, qc, color=SUB, lw=1.0, zorder=3)
    sc = np.array([np.nan if v is None else v for v in card["close"]], dtype=float)
    ax2.plot(xs, sc, color=PURPLE, lw=2.4, zorder=4, solid_capstyle="round")
    b = card["buy_at"]
    x_sell = n - 1 + max(int(card.get("sell_in") or 0), 0)  # 매도 예정일 = 기준일 + sell_in 거래일
    if x_sell > n - 1:
        ax.axvspan(n - 1, x_sell, color=GRID, alpha=0.35, lw=0, zorder=0)  # 아직 오지 않은 기간
    dot = {"s": 120, "edgecolor": "white", "linewidth": 1.6, "zorder": 9}
    d0 = pd.Timestamp(card["d0"]) if card.get("d0") else None
    if d0 is not None and d0 in dates:
        i0 = int(dates.get_loc(d0))
        if not np.isnan(sc[i0]):
            ax.axvline(i0, color=DOT_EARN, lw=1.0, ls=(0, (3, 3)), alpha=0.6, zorder=1)
            ax2.scatter([i0], [sc[i0]], color=DOT_EARN, **dot)
    ax.axvline(b, color=DOT_BUY, lw=1.0, ls=(0, (3, 3)), alpha=0.6, zorder=1)
    ax2.scatter([b], [card["buy_px"]], color=DOT_BUY, **dot)
    bd = pd.Timestamp(card["buy_day"])
    right = b < n / 2
    ax2.annotate(f"매수 {bd.month}/{bd.day} ${card['buy_px']:,.0f}", (b, card["buy_px"]), xytext=(8 if right else -8, 10),
                 textcoords="offset points", ha="left" if right else "right", fontsize=10, color=DOT_BUY,
                 bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none"}, zorder=7)
    ax.axvline(x_sell, color="#CA8A04", lw=1.0, ls=(0, (3, 3)), alpha=0.7, zorder=1)
    ax.scatter([x_sell], [lo], color=DOT_SELL, clip_on=False, **{**dot, "edgecolor": INK, "linewidth": 1.0})
    # 점 색인 — 그래프 바깥(태그 줄과 축 사이), 매도 예정 날짜는 오른쪽 위 글자(sell_note)에 있다
    keys = [("실적 발표(반응일)", DOT_EARN, "white"), ("매수일", DOT_BUY, "white"), ("매도 예정일", DOT_SELL, INK)]
    fig.legend(handles=[Line2D([], [], ls="none", marker="o", ms=9, mfc=c, mec=e, mew=1.0, label=t) for t, c, e in keys],
               loc="lower left", bbox_to_anchor=(0.075, 0.745), ncol=3, fontsize=10, frameon=False, handletextpad=0.3,
               columnspacing=1.2, borderaxespad=0)
    # 끝점 이름표 — 흰 상자, 더 높은 쪽이 위
    ys = card["last_close"] / k, card["last_qqq"]
    gap = (hi - lo) * 0.13
    y_s, y_q = ys
    if abs(y_s - y_q) < gap:
        mid = (y_s + y_q) / 2
        y_s, y_q = (mid + gap / 2, mid - gap / 2) if ys[0] >= ys[1] else (mid - gap / 2, mid + gap / 2)
    box = {"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.95}
    ax.annotate(f"${card['last_close']:,.0f} {card['ret']:+.1f}%", (n - 1, ys[0]), xytext=(n - 1 + n * 0.02, y_s),
                textcoords="data", va="center", fontsize=11, fontweight="bold", color=PURPLE, bbox=box, zorder=8,
                annotation_clip=False)
    ax.annotate(f"QQQ ${card['last_qqq']:,.0f} {card['qqq_ret']:+.1f}%", (n - 1, ys[1]), xytext=(n - 1 + n * 0.02, y_q),
                textcoords="data", va="center", fontsize=10, fontweight="bold", color=SUB, bbox=box, zorder=8,
                annotation_clip=False)
    ax.set_xlim(-1, max(n - 1 + n * 0.30, x_sell + (x_sell + 1) * 0.04))
    # 눈금은 매도 예정일까지 늘린 축 전체 기준(앞날은 평일로 근사 — 눈금 글자에만 쓴다)
    ext = dates.append(pd.bdate_range(dates[-1] + pd.offsets.BDay(1), periods=x_sell - (n - 1))) if x_sell > n - 1 else dates
    (_month_ticks(ax, ext, np.arange(len(ext))) if len(ext) > 40 else _week_ticks(ax, ext))
    ax.tick_params(axis="y", colors=SUB, labelsize=9.5, length=0)
    ax2.tick_params(axis="y", colors=PURPLE, labelsize=9.5, length=0)
    ax.tick_params(axis="x", colors=SUB, labelsize=9.5, length=0)
    ax.yaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
    ax2.yaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
    for a in (ax, ax2):
        a.spines["top"].set_visible(False)
        for side in ("left", "right", "bottom"):
            a.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    return _save(fig, out_path)
