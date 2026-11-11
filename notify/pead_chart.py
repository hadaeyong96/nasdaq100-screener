"""실적급등 추천 종목 비교 그래프 (개인판 보고서 전용, 2026-11-11 사용자 요청).

추천 종목마다 매수일(추천 다음 날 시가)부터 매수가 대비 % 선을 그리고, 선 끝에 회사명·수익률을 적는다.
QQQ는 가장 이른 매수일 시가 기준 점선, 이름표의 "QQQ 대비"는 그 종목 매수일 시가 기준. 끝난 종목은 흐리게. 단체방판에는 넣지 않는다(종목 이름 없음 규칙).
입력은 engine.pead_watch.chart_data 결과({lines, qqq}). 계산 없음.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

WIDTH, HEIGHT, DPI = 10, 5.6, 120
PALETTE = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2", "#bcbd22", "#7f7f7f")


def render(chart: dict | None, out_path: Path, as_of: str | None = None) -> Path | None:
    """그래프 PNG 저장. 선이 없으면 None."""
    if not chart or not chart.get("lines"):
        return None
    import matplotlib
    from matplotlib.figure import Figure

    from notify.curve_chart import korean_font

    font = korean_font()
    matplotlib.rcParams["axes.unicode_minus"] = False
    if font:
        matplotlib.rcParams["font.family"] = [font, "DejaVu Sans"]
    fig = Figure(figsize=(WIDTH, HEIGHT), dpi=DPI)
    ax = fig.subplots()
    q = chart["qqq"]
    ax.plot(pd.to_datetime(q["dates"]), q["values"], color="#555", lw=1.6, ls="--", label="QQQ (가장 이른 매수일 기준)")
    ends = []
    for k, ln in enumerate(chart["lines"]):
        col = PALETTE[k % len(PALETTE)]
        open_ = ln["status"] == "open"
        x = pd.to_datetime(ln["dates"])
        ax.plot(x, ln["values"], color=col, lw=2.2 if open_ else 1.3, alpha=1.0 if open_ else 0.5)
        ax.scatter([x[0]], [ln["values"][0]], color=col, s=18, zorder=3, alpha=1.0 if open_ else 0.5)
        vq = ln.get("vs_qqq")
        tag = (f"{ln['name']} {ln['values'][-1]:+.1f}%" + (f" · QQQ 대비 {vq:+.1f}%p" if vq is not None else "")
               + ("" if open_ else " (끝)"))
        ends.append([x[-1], ln["values"][-1], tag, col, open_])
    ends.sort(key=lambda e: e[1])  # 끝 이름표 겹침을 줄인다: 세로로 최소 간격
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.045
    for i in range(1, len(ends)):
        if ends[i][1] - ends[i - 1][1] < gap and ends[i][0] == ends[i - 1][0]:
            ends[i][1] = ends[i - 1][1] + gap
    for x, y, tag, col, open_ in ends:
        ax.annotate(tag, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=11, color=col,
                    alpha=1.0 if open_ else 0.6)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_ylabel("매수가 대비 (%)")
    ax.set_title(f"실적급등 추천 종목 — 추천 다음 날 시가부터{' · 기준일 ' + as_of if as_of else ''}")
    ax.grid(axis="y", color="#eee")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.margins(x=0.30)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    return out_path
