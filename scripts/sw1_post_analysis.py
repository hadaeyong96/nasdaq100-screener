"""SW1 결과 후 추가 분석 (판정 불변, 규칙 변경 없음).

입력: outputs/sw1_trades.csv (sw1_experiments.py 주 판정 매매 601건), QQQ 거래일.
출력: docs/results/sw1.md 끝에 "결과 후 추가 분석" 장을 붙인다 (이미 있으면 그 장만 바꾼다). 기존 판정·숫자는 건드리지 않는다.

1) 세후 초과 분포  2) 분기 묶음 부트스트랩 95% 구간  3) 상위 5·10건 뺀 평균
4) 동시 보유 종목 수  5) 신호 20% 무작위 제외 1,000회
세금은 해석 H2대로 매매 집합(장부)이 바뀔 때마다 sw1.ledger_tax로 다시 계산한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from core import sw1  # noqa: E402
from scripts import sw1_data_check as dc  # noqa: E402

TRADES_CSV = ROOT / "outputs" / "sw1_trades.csv"
RESULT_MD = ROOT / "docs" / "results" / "sw1.md"
HEADER = "## 결과 후 추가 분석 (판정 불변)"
SEED = 20261009  # 사전 등록 무작위 시드와 같은 값
RATE = 0.22
BOOT_ITERS = 10_000
DROP_ITERS = 1_000
DROP_FRAC = 0.20


def after_tax_excess(t: pd.DataFrame) -> np.ndarray:
    """매매 표 → 그 집합을 한 장부로 본 매매별 세후 초과 수익 (신호 − QQQ). 입력 열: ret_krw, bench_krw, sell_year."""
    yr = t["sell_year"].values
    ra = t["ret_krw"].values - sw1.ledger_tax(t["ret_krw"].values, yr, RATE)
    ba = t["bench_krw"].values - sw1.ledger_tax(t["bench_krw"].values, yr, RATE)
    return ra - ba


def p(x: float, d: int = 2) -> str:
    return f"{x * 100:+.{d}f}%"


def distribution(t: pd.DataFrame) -> list[str]:
    """1) 매매별 세후 초과(주 장부 값) 분포 → 표 줄."""
    x = t["ex_after"].values
    q10, q90 = np.quantile(x, [0.1, 0.9])
    k = int(round(len(x) * 0.1))
    s = np.sort(x)
    rows = [("평균", p(x.mean())), ("중앙값", p(np.median(x))), ("표준편차", f"{x.std(ddof=1) * 100:.2f}%"),
            ("평균의 표준오차 (σ/√n, 매매 독립 가정)", f"{x.std(ddof=1) / np.sqrt(len(x)) * 100:.2f}%"),
            ("하위 10% 경계 (10백분위)", p(q10)), ("상위 10% 경계 (90백분위)", p(q90)),
            (f"하위 10% ({k}건) 평균", p(s[:k].mean())), (f"상위 10% ({k}건) 평균", p(s[-k:].mean())),
            ("최소 · 최대", f"{p(s[0])} · {p(s[-1])}")]
    return ["| 항목 | 값 |", "| --- | --- |"] + [f"| {a} | {b} |" for a, b in rows]


def quarter_bootstrap(t: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """2) 분기 묶음 부트스트랩: 39분기를 복원 추출, 회차마다 장부 세금 다시 계산 → 평균 초과의 분포."""
    rng = np.random.default_rng(SEED)
    groups = [g.index.values for _, g in t.groupby("quarter")]
    means = np.empty(BOOT_ITERS)
    for i in range(BOOT_ITERS):
        pick = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[j] for j in pick])
        means[i] = after_tax_excess(t.loc[idx]).mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    lines = ["| 항목 | 값 |", "| --- | --- |",
             f"| 분기 수 · 회차 · 시드 | {len(groups)} · {BOOT_ITERS:,} · {SEED} |",
             f"| 부트스트랩 평균 초과 중앙값 | {p(np.median(means))} |",
             f"| 95% 신뢰구간 (2.5~97.5백분위) | {p(lo)} ~ {p(hi)} |",
             f"| 0 이하 회차 비율 | {(means <= 0).mean() * 100:.1f}% |"]
    return lines, means


def drop_top(t: pd.DataFrame) -> list[str]:
    """3) 세후 초과 상위 k건을 뺀 평균: 주 장부 값 그대로 / 남은 매매로 세금 다시 계산."""
    order = t.sort_values("ex_after", ascending=False)
    lines = ["| 뺀 건수 | 남은 매매 | 평균 초과 (주 장부 세금 그대로) | 평균 초과 (세금 다시 계산) | 뺀 매매 |",
             "| --- | --- | --- | --- | --- |"]
    for k in (0, 5, 10):
        rest = order.iloc[k:]
        names = ", ".join(f"{r.ticker} {r.d0[:7]} {p(r.ex_after, 0)}" for r in order.iloc[:k].itertuples()) if k else "—"
        lines.append(f"| {k} | {len(rest)} | {p(rest['ex_after'].mean())} | {p(after_tax_excess(rest).mean())} | {names} |")
    return lines


def concurrency(t: pd.DataFrame, days: pd.DatetimeIndex) -> list[str]:
    """4) 거래일마다 보유 중인 매매 수 (매수일~매도일 포함). 기간 = 첫 매수일 ~ 마지막 매도일."""
    b, s = pd.to_datetime(t["buy_day"]), pd.to_datetime(t["sell_day"])
    cal = days[(days >= b.min()) & (days <= s.max())]
    bi, si = cal.get_indexer(b), cal.get_indexer(s)
    if (bi < 0).any() or (si < 0).any():
        raise SystemExit("[sw1-post] 매수·매도일이 거래일 달력에 없음 — 멈춤")
    diff = np.zeros(len(cal) + 1, dtype=int)
    np.add.at(diff, bi, 1)
    np.add.at(diff, si + 1, -1)
    n = pd.Series(np.cumsum(diff[:-1]), index=cal)
    q50, q90, q99 = np.quantile(n.values, [0.5, 0.9, 0.99])
    lines = ["| 항목 | 값 |", "| --- | --- |",
             f"| 기간 (거래일) | {cal[0].date()} ~ {cal[-1].date()} ({len(cal):,}일) |",
             f"| 평균 · 중앙값 | {n.mean():.1f} · {q50:.0f} |",
             f"| 90 · 99백분위 · 최대 | {q90:.0f} · {q99:.0f} · {n.max()} ({n.idxmax().date()}) |",
             f"| 보유 0종목인 거래일 | {(n == 0).mean() * 100:.1f}% |",
             "", "| 연도 | 평균 | 최대 |", "| --- | --- | --- |"]
    for y, g in n.groupby(n.index.year):
        lines.append(f"| {y} | {g.mean():.1f} | {g.max()} |")
    return lines


def drop_signals(t: pd.DataFrame) -> list[str]:
    """5) 회차마다 신호 20%를 비복원으로 빼고 남은 매매로 장부 세금 다시 계산 → 평균 초과 1,000개."""
    rng = np.random.default_rng(SEED)
    n = len(t)
    keep = n - int(round(n * DROP_FRAC))
    y0 = int(t["sell_year"].min())
    ny = int(t["sell_year"].max()) - y0 + 1
    sel = np.stack([np.sort(rng.choice(n, size=keep, replace=False)) for _ in range(DROP_ITERS)])
    R, B = t["ret_krw"].values[sel], t["bench_krw"].values[sel]
    Y = t["sell_year"].values[sel] - y0
    ex = (R - sw1.ledger_tax_many(R, Y, ny, RATE)) - (B - sw1.ledger_tax_many(B, Y, ny, RATE))
    m = ex.mean(axis=1)
    q = np.quantile(m, [0.025, 0.05, 0.5, 0.95, 0.975])
    return ["| 항목 | 값 |", "| --- | --- |",
            f"| 회차 · 남은 매매 · 시드 | {DROP_ITERS:,} · {keep} (뺀 {n - keep}) · {SEED} |",
            f"| 평균 · 중앙값 | {p(m.mean())} · {p(q[2])} |",
            f"| 5 · 95백분위 | {p(q[1])} · {p(q[3])} |",
            f"| 2.5 · 97.5백분위 | {p(q[0])} · {p(q[4])} |",
            f"| 최소 · 최대 | {p(m.min())} · {p(m.max())} |",
            f"| 0 이하 회차 | {(m <= 0).sum()} / {DROP_ITERS:,} |"]


def main() -> None:
    t = pd.read_csv(TRADES_CSV, encoding="utf-8-sig")
    t["sell_year"] = pd.to_datetime(t["sell_day"]).dt.year
    # 입력 확인: 주 판정 표와 같은 매매 집합인지, 세금 재계산이 저장값과 같은지
    if len(t) != 601 or round(t["ex_after"].mean() * 100, 2) != 0.53:
        raise SystemExit(f"[sw1-post] 매매 표가 주 판정과 다름 ({len(t)}건, {t['ex_after'].mean():+.4%}) — 멈춤")
    if not np.allclose(after_tax_excess(t), t["ex_after"].values, atol=1e-12):
        raise SystemExit("[sw1-post] 장부 세금 재계산이 저장값과 다름 — 멈춤")
    days = pd.read_parquet(dc.BT_CACHE / "QQQ.parquet").index

    boot_lines, _ = quarter_bootstrap(t)
    body = [
        HEADER, "",
        "> 결과(시도 1)를 본 뒤 추가한 기술 통계다. **판정·규칙·위 숫자는 바꾸지 않는다.** 판정에 쓰지 않는다.",
        f"> 스크립트 `scripts/sw1_post_analysis.py`, 입력 `outputs/sw1_trades.csv`(601건). "
        "세금은 매매 집합이 바뀌면 그 집합을 한 장부로 다시 계산(H2). 단위: 매매당 세후 초과 수익(QQQ 대비, 원화).",
        "", "### 1) 매매별 세후 초과 수익 분포", "", *distribution(t),
        "", "### 2) 분기 묶음 부트스트랩 (같은 분기 반응일 매매를 한 묶음)", "", *boot_lines,
        "", "### 3) 상위 매매를 뺀 평균", "", *drop_top(t),
        "", "### 4) 동시 보유 종목 수 (매매 1건 = 1종목, 같은 종목 겹침 없음)", "", *concurrency(t, days),
        "", f"### 5) 신호 {DROP_FRAC:.0%}를 무작위로 놓쳤을 때", "", *drop_signals(t), "",
    ]
    text = RESULT_MD.read_text(encoding="utf-8")
    if HEADER in text:
        text = text[: text.index(HEADER)].rstrip("\n") + "\n"
    RESULT_MD.write_text(text.rstrip("\n") + "\n\n" + "\n".join(body), encoding="utf-8")
    print("\n".join(body))


if __name__ == "__main__":
    main()
