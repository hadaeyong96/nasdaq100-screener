"""KJB-1 함께 보고할 것(계획서 5장) 계산 + 최종 완료 보고서 작성.

scripts/kjb1_experiments.py가 만든 outputs/backtest/kjb1/<run_id>/의 trades_kjb.csv·
admission_log.csv·random_entries.csv(있으면)를 읽어 5장 항목을 계산하고 report.md를
같은 폴더에 쓴다. 무거운 시뮬레이션은 다시 하지 않는다(캐시된 시세만 다시 읽어
섹터 확산도·RSI A1 겹침 계산에만 쓴다).

실행:
    python -u -m scripts.kjb1_report_extra <run_id>
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from core import kjb  # noqa: E402
from data import sectors  # noqa: E402
from engine import backtest as bt  # noqa: E402
from scripts.kjb1_experiments import OUT_ROOT, load_preregistration  # noqa: E402


def rsi_a1_dates(df: pd.DataFrame) -> set:
    """core.signals.check_a1과 같은 조건(RSI 30 상향 돌파)을 벡터로 계산한다."""
    a1 = (df["rsi"].shift(1) < 30) & (df["rsi"] >= 30)
    return set(df.index[a1.fillna(False)])


def main() -> None:
    if len(sys.argv) != 2:
        print("사용법: python -u -m scripts.kjb1_report_extra <run_id>")
        sys.exit(2)
    run_id = sys.argv[1]
    out_dir = OUT_ROOT / run_id
    trades = pd.read_csv(out_dir / "trades_kjb.csv")
    admission_log_df = pd.read_csv(out_dir / "admission_log.csv")

    cfg = bt.load_config()
    prereg = load_preregistration()
    start = date.fromisoformat(prereg["main_settings"]["period"]["start"])
    end = date.fromisoformat(prereg["main_settings"]["period"]["end"])
    warmup_start = bt._parse_date(cfg["backtest"]["warmup_start"])

    print("[kjb1-extra] 캐시된 시세 다시 읽는 중...")
    data = bt.prepare_data(cfg, warmup_start, end)
    sector_by_ticker, _ = sectors.get_sectors(sorted(data.all_needed_tickers))

    entries = trades[trades["side"] == "진입"].copy()
    exits = trades[trades["side"] == "청산"].copy()
    entries["year"] = pd.to_datetime(entries["date"]).dt.year
    exits["year"] = pd.to_datetime(exits["date"]).dt.year

    # ── 연도별 신호 수 / 체결 수 ──
    admission_log_df["tickers_list"] = admission_log_df["tickers"].apply(eval)
    admission_log_df["year"] = pd.to_datetime(admission_log_df["date"]).dt.year
    signal_counts_by_year = admission_log_df.groupby("year")["tickers_list"].apply(lambda s: sum(len(x) for x in s)).to_dict()
    fill_counts_by_year = entries.groupby("year").size().to_dict()

    # ── 섹터별 신호 분포 ──
    sector_signal_counts: dict[str, int] = defaultdict(int)
    for tickers_list in admission_log_df["tickers_list"]:
        for t in tickers_list:
            sector_signal_counts[sector_by_ticker.get(t, "UNKNOWN")] += 1

    # ── position별 R (aggregate_positions) + first_score/섹터 join ──
    trades_records = trades.to_dict("records")
    still_open = set()  # 이미 다 청산됐거나 report에서는 회귀 방지용 재계산이 아니므로 빈 집합으로 둔다
    positions = bt.aggregate_positions(trades_records, still_open)
    entry_meta = {int(r["position_id"]): r for r in entries.to_dict("records") if pd.notna(r.get("position_id"))}

    score_buckets = {"0~10": [], "11~30": [], "31~60": []}
    dispersion_high_r, dispersion_low_r = [], []
    for p in positions:
        meta = entry_meta.get(int(p["position_id"]))
        if meta is None:
            continue
        fs = meta.get("first_score")
        if pd.notna(fs):
            fs = float(fs)
            bucket = "0~10" if fs <= 10 else ("11~30" if fs <= 30 else "31~60")
            score_buckets[bucket].append(p["r"])

        sector = meta.get("sector")
        entry_ts = pd.Timestamp(meta["date"])
        close_by_ticker = {t: df["close"] for t, df in data.indicator_map.items()}
        disp = kjb.sector_dispersion(close_by_ticker, sector_by_ticker, entry_ts)
        if sector in disp:
            (dispersion_high_r if disp[sector] >= 0.5 else dispersion_low_r).append(p["r"])

    def _avg(xs):
        return round(sum(xs) / len(xs), 2) if xs else None

    score_bucket_avg_r = {k: {"n": len(v), "avg_r": _avg(v)} for k, v in score_buckets.items()}
    dispersion_avg_r = {
        "high (>=50%)": {"n": len(dispersion_high_r), "avg_r": _avg(dispersion_high_r)},
        "low (<50%)": {"n": len(dispersion_low_r), "avg_r": _avg(dispersion_low_r)},
    }

    # ── 우리 신호(RSI A1)와 겹친 종목 수 ──
    overlap_count = 0
    total_kjb_signals = 0
    for _, row in admission_log_df.iterrows():
        d = pd.Timestamp(row["date"])
        for t in row["tickers_list"]:
            total_kjb_signals += 1
            df = data.indicator_map.get(t)
            if df is not None and "rsi" in df.columns and d in rsi_a1_dates(df):
                overlap_count += 1

    # ── 대장주 선행 신호 vs 이후 20일 QQQ 수익률 ──
    qqq_close = data.qqq_df["close"]
    close_by_ticker = {t: df["close"] for t, df in data.indicator_map.items()}
    volume_by_ticker = {t: df["volume"] for t, df in data.indicator_map.items()}
    leading_rows = []
    for d in sorted(set(pd.Timestamp(x) for x in admission_log_df["date"])):
        lead = kjb.leading_stock_signal(close_by_ticker, volume_by_ticker, qqq_close, d)
        if lead is None or d not in qqq_close.index:
            continue
        loc = qqq_close.index.get_loc(d)
        if loc + 20 >= len(qqq_close):
            continue
        fwd_ret = qqq_close.iloc[loc + 20] / qqq_close.iloc[loc] - 1
        leading_rows.append({"date": d.date().isoformat(), "leading_signal": round(lead, 4), "qqq_fwd_20d_return": round(float(fwd_ret), 4)})

    extra = {
        "signal_counts_by_year": signal_counts_by_year,
        "fill_counts_by_year": fill_counts_by_year,
        "sector_signal_counts": dict(sector_signal_counts),
        "score_bucket_avg_r": score_bucket_avg_r,
        "dispersion_avg_r": dispersion_avg_r,
        "rsi_a1_overlap": {"overlap": overlap_count, "total_kjb_signals": total_kjb_signals},
        "leading_stock_vs_qqq_fwd20d": leading_rows,
    }

    pd.DataFrame(leading_rows).to_csv(out_dir / "leading_stock_vs_qqq.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "reporting_extra.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(extra, f, allow_unicode=True, sort_keys=False)
    print(f"[kjb1-extra] 저장: {out_dir / 'reporting_extra.yaml'}")
    print(yaml.safe_dump(extra, allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()
