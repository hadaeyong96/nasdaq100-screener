"""D1 적립식 매수 방식 비교 (사전 등록: docs/d1_plan.md, configs/d1_preregistration.yaml, tag d1-prereg).

일회성 리포트 스크립트. 규칙·장부는 core/dca.py(순수 함수). 데이터는 L1 체크포인트(QQQ·환율·DTB3, C2와 같은
engine.portfolio.prepare_data 결과)와 L1b 로더(프렌치·^IXIC)를 그대로 쓴다. 시작할 때 사전 등록 두 파일의
sha256(LF 정규화)을 확인하고 다르면 멈춘다.

실행: python -u -m scripts.d1_experiments
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from core import dca  # noqa: E402

PREREG_FILES = {
    "docs/d1_plan.md": "3b2362f6c09988d54934e774711a3b25c7f994e9a367fbd7dbf8df97c8e8b1af",
    "configs/d1_preregistration.yaml": "78674468e854d6cf19aca57aa6d3ecaa7d17cdba89db0509d57ecd5338d849f0",
}
OUT_DIR = ROOT / "outputs"
RESULT_MD = ROOT / "docs" / "results" / "d1.md"
DATASETS = ["QQQ", "NASDAQ", "US"]
DS_KO = {"QQQ": "QQQ (원화·세후)", "NASDAQ": "나스닥 ^IXIC (달러·세전)", "US": "미국 시장 프렌치 (달러·세전)"}
CAND_KO = {"D0": "매달 그대로", "D1": "하락한 달에 더", "D2": "많이 빠질수록 더", "D3": "가치 평균법", "D4": "200일선 아래에서 더", "D5": "계절(11~4월)"}
HORIZON = 120


def normalized_sha256(path: Path) -> str:
    """줄바꿈을 LF로 맞춘 내용의 sha256 (Windows 체크아웃의 CRLF 변환에 흔들리지 않게)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_preregistration() -> dict:
    for rel, expected in PREREG_FILES.items():
        got = normalized_sha256(ROOT / rel)
        if got != expected:
            raise SystemExit(f"[d1] 사전 등록 파일이 바뀌었습니다: {rel} (기대 {expected[:12]}…, 실제 {got[:12]}…) — 멈춤")
    print("[d1] 사전 등록 해시 확인 통과", flush=True)
    with open(ROOT / "configs" / "d1_preregistration.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 데이터 ───────────────────────────────────────────────────────────────────


def load_datasets(prereg: dict) -> dict:
    """세 데이터: days, price, fx, rate, div, facts, seal, window_starts. 모두 봉인 검사를 거친 값."""
    from core import l1b
    from engine import backtest as bt
    from scripts import l1_experiments as l1x
    from scripts import l1b_experiments as l1bx

    cfg = bt.load_config()
    ctx = l1x.load_all(cfg, {"periods": {"full": {"start": "1999-03-10", "end": "2021-12-31"}}})  # seal_portfolio_data 포함
    data = ctx["data"]
    days = data.core_df.index
    fx = pd.Series({pd.Timestamp(k): v for k, v in data.fx_by_date.items()}, dtype=float).reindex(days)
    if fx.isna().any():
        raise SystemExit(f"[d1] QQQ 환율 결측일 {int(fx.isna().sum())}개 — 멈춤")
    sig = data.sma_source_df["close"]
    out = {"QQQ": {
        "days": days, "price": data.core_df["close"], "fx": fx, "rate": data.reserve_daily_rate.reindex(days).fillna(0.0),
        "div": data.core_dividends, "facts": dca.signal_facts(sig).reindex(days), "krw": True,
        "seal": pd.Timestamp(prereg["datasets"]["QQQ"]["seal"]), "signal_start": sig.index[0],
    }}
    with open(ROOT / "configs" / "l1b_preregistration.yaml", encoding="utf-8") as f:
        pre_b = yaml.safe_load(f)
    ctx_b = l1bx.load_markets(pre_b)  # 1998-12-31 봉인 포함
    for key, mk in (("NASDAQ", "N"), ("US", "U")):
        m = ctx_b[mk]
        lv = m["level"]
        out[key] = {"days": lv.index, "price": lv, "fx": None, "rate": m["rf"].reindex(lv.index).fillna(0.0),
                    "div": pd.Series(dtype=float), "facts": dca.signal_facts(lv), "krw": False,
                    "seal": pd.Timestamp(prereg["datasets"][key]["seal"]), "signal_start": lv.index[0]}
        l1b.seal(lv, key)
    for key, d in out.items():
        d["base"] = float(prereg["base_amount"][key])
        a, b = prereg["datasets"][key]["window_starts"]
        d["starts"] = list(pd.period_range(a, b, freq="M"))
        if d["days"][-1] > d["seal"]:
            raise SystemExit(f"[d1] {key} 봉인일 뒤 행 — 멈춤")
    return out


def window(d: dict, start: pd.Period, rule: str = "first"):
    months = [start + i for i in range(HORIZON)]
    dates = dca.buy_dates(d["days"], months, rule)
    valuation = dca.buy_dates(d["days"], [start + HORIZON], "first")[0]
    if valuation > d["seal"]:
        raise SystemExit(f"[d1] 평가일 {valuation.date()}이 봉인일 뒤 — 멈춤")
    return dca.build_steps(d["days"], d["price"], d["fx"], d["rate"], d["div"], d["facts"], dates, valuation)


def sim(d: dict, steps, cand: str) -> dict:
    return dca.simulate(steps, cand, d["base"], krw=d["krw"])


def run_all_windows(ds: dict) -> dict:
    res = {}
    for key, d in ds.items():
        print(f"[d1] {key}: {len(d['starts'])}개 구간", flush=True)
        rows = []
        for s in d["starts"]:
            steps = window(d, s)
            row = {"start": str(s), "valuation": steps[-1].date.date()}
            for c in dca.CANDIDATES:
                r = sim(d, steps, c)
                if r["min_cash"] < -1e-6:
                    raise SystemExit(f"[d1] 현금 음수 — {key} {s} {c}")
                row[c] = r["final"]
                row[f"{c}_cash"] = r["cash_ratio_avg"]
                row[f"{c}_contrib"] = r["contributed"]
            for rule in ("mid", "last"):
                row[f"D6_{rule}"] = sim(d, window(d, s, rule), "D0")["final"]
            if len({row[f"{c}_contrib"] for c in dca.CANDIDATES}) != 1:
                raise SystemExit(f"[d1] 총 납입이 다름 — {key} {s}")
            rows.append(row)
        res[key] = pd.DataFrame(rows)
    return res


# ── 측정·판정 ───────────────────────────────────────────────────────────────


def stats(df: pd.DataFrame) -> dict:
    out = {}
    for c in dca.CANDIDATES:
        v = df[c].to_numpy()
        out[c] = {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
                  "win": float((df[c] > df["D0"]).mean() * 100), "cash": float(df[f"{c}_cash"].mean() * 100)}
    for c in dca.CANDIDATES:
        out[c]["median_vs_D0"] = (out[c]["median"] / out["D0"]["median"] - 1) * 100
        out[c]["p10_vs_D0"] = (out[c]["p10"] / out["D0"]["p10"] - 1) * 100
    return out


def verdict(st: dict) -> dict:
    out = {}
    for c in dca.CANDIDATES[1:]:
        per = {k: {1: st[k][c]["median_vs_D0"] >= 1.0, 2: st[k][c]["p10"] >= st[k]["D0"]["p10"], 3: st[k][c]["win"] >= 60.0} for k in DATASETS}
        out[c] = {"per": per, "pass": all(all(v.values()) for v in per.values()),
                  "p10_score": float(np.mean([st[k][c]["p10_vs_D0"] for k in DATASETS]))}
    return out


# ── 출력 ─────────────────────────────────────────────────────────────────────


def _f(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return "예" if x else "아니오"
    if isinstance(x, (float, np.floating)):
        return f"{x:,.{nd}f}"
    return str(x)


def md_table(rows: list[dict]) -> str:
    if not rows:
        return "(없음)\n"
    cols = list(rows[0])
    return "\n".join(["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
                     + ["| " + " | ".join(_f(r[c]) for c in cols) + " |" for r in rows]) + "\n"


def make_plots(res: dict, ds: dict) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    paths = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, key in zip(axes, DATASETS):
        df = res[key]
        contrib = df["D0_contrib"].iloc[0]
        ax.boxplot([df[c] / contrib for c in dca.CANDIDATES], tick_labels=dca.CANDIDATES, whis=(10, 90), showfliers=True)
        ax.axhline(1.0, color="#999", lw=0.8, ls="--")
        ax.set_title(f"{DS_KO[key]}\n최종 자산 ÷ 총 납입 (수염 = 10~90%)", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = OUT_DIR / "d1_final_boxplot.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    colors = {"D1": "#1f77b4", "D2": "#d62728", "D3": "#2ca02c", "D4": "#9467bd", "D5": "#ff7f0e"}
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    for ax, key in zip(axes, DATASETS):
        df = res[key]
        x = pd.PeriodIndex(df["start"], freq="M").to_timestamp()
        for c in dca.CANDIDATES[1:]:
            ax.plot(x, (df[c] / df["D0"] - 1) * 100, lw=0.9, color=colors[c], label=f"{c} {CAND_KO[c]}")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"{DS_KO[key]} — 시작 월별 D0 대비 최종 자산 차이 %")
        ax.legend(fontsize=8, ncol=5, loc="upper left")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p = OUT_DIR / "d1_vs_d0_by_window.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)
    return paths


def write_outputs(ds: dict, res: dict, st: dict, vd: dict) -> None:
    import subprocess

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    tag = subprocess.run(["git", "rev-parse", "d1-prereg"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    for key in DATASETS:
        res[key].to_csv(OUT_DIR / f"d1_windows_{key.lower()}.csv", index=False, encoding="utf-8-sig")

    period_rows = []
    for key in DATASETS:
        d, df = ds[key], res[key]
        period_rows.append({"데이터": DS_KO[key], "데이터 기간": f"{d['days'][0].date()} ~ {d['days'][-1].date()}",
                            "신호 지수 시작": d["signal_start"].date(), "시작 월": f"{df['start'].iloc[0]} ~ {df['start'].iloc[-1]}",
                            "구간 수": len(df), "마지막 평가일": df["valuation"].iloc[-1], "매달 납입": f"{d['base']:,.0f}" + ("원" if d["krw"] else "달러"),
                            "총 납입(구간당)": f"{df['D0_contrib'].iloc[0]:,.0f}"})
    method_rows = []
    for key in DATASETS:
        for c in dca.CANDIDATES:
            s = st[key][c]
            method_rows.append({"데이터": key, "방식": f"{c} {CAND_KO[c]}", "중앙값": s["median"], "하위 10%": s["p10"], "상위 10%": s["p90"],
                                "중앙값 D0 대비 %": s["median_vs_D0"], "하위10% D0 대비 %": s["p10_vs_D0"],
                                "D0보다 많이 남은 구간 %": s["win"] if c != "D0" else None, "현금 대기 %": s["cash"]})
    pd.DataFrame(method_rows).to_csv(OUT_DIR / "d1_summary.csv", index=False, encoding="utf-8-sig")

    worst_rows = []
    for key in DATASETS:
        df = res[key]
        if key == "QQQ":
            pick, why = "2000-03", "2000년 고점 직전 시작"
        else:
            pick = df.loc[(df["D0"] / df["D0_contrib"]).idxmin(), "start"]
            why = "D0 최종/총납입 최저"
        r = df[df["start"] == pick].iloc[0]
        row = {"데이터": key, "시작 월": pick, "이유": why, "총 납입": r["D0_contrib"]}
        for c in dca.CANDIDATES:
            row[c] = r[c]
        for c in dca.CANDIDATES[1:]:
            row[f"{c} D0 대비 %"] = (r[c] / r["D0"] - 1) * 100
        worst_rows.append(row)
    pd.DataFrame(worst_rows).to_csv(OUT_DIR / "d1_worst.csv", index=False, encoding="utf-8-sig")

    d6_rows, d6_same = [], {}
    for key in DATASETS:
        df = res[key]
        base = float(np.median(df["D0"]))
        mid, last = float(np.median(df["D6_mid"])), float(np.median(df["D6_last"]))
        dm, dl = (mid / base - 1) * 100, (last / base - 1) * 100
        d6_same[key] = abs(dm) < 0.3 and abs(dl) < 0.3
        d6_rows.append({"데이터": key, "첫 거래일 중앙값": base, "15일 이후 첫 거래일": mid, "마지막 거래일": last,
                        "15일 차이 %": dm, "마지막 차이 %": dl,
                        "15일이 이긴 구간 %": float((df["D6_mid"] > df["D0"]).mean() * 100),
                        "마지막이 이긴 구간 %": float((df["D6_last"] > df["D0"]).mean() * 100)})
    pd.DataFrame(d6_rows).to_csv(OUT_DIR / "d1_buyday.csv", index=False, encoding="utf-8-sig")
    plots = make_plots(res, ds)

    v_rows = []
    for c in dca.CANDIDATES[1:]:
        row = {"방식": f"{c} {CAND_KO[c]}"}
        for key in DATASETS:
            p = vd[c]["per"][key]
            row[key] = " ".join(("✓" if p[i] else "✗") for i in (1, 2, 3))
        row["판정"] = "D0보다 낫다" if vd[c]["pass"] else "아니다"
        v_rows.append(row)
    passed = [c for c in dca.CANDIDATES[1:] if vd[c]["pass"]]
    if not passed:
        conclusion = "**매달 그대로 사기(D0)가 가장 좋다** — D1~D5 중 세 데이터 모두에서 판정 1·2·3을 충족한 방식이 없다."
    else:
        best = max(passed, key=lambda c: vd[c]["p10_score"])
        conclusion = f"**{best} {CAND_KO[best]}가 D0보다 낫다** (해당 {len(passed)}개 중 하위 10% 평균 비율 최고)"
    d6_line = "아무 날이나 같음" if all(d6_same.values()) else "차이가 0.3% 이상인 데이터가 있음: " + ", ".join(k for k, v in d6_same.items() if not v)

    md = f"""# D1 적립식 매수 방식 비교 — 결과 (D1 시도 1)

- 사전 등록: `docs/d1_plan.md`, `configs/d1_preregistration.yaml` (tag `d1-prereg` = `{tag}`), 시작 시 sha256 확인 통과
  - `docs/d1_plan.md` `{PREREG_FILES['docs/d1_plan.md']}`
  - `configs/d1_preregistration.yaml` `{PREREG_FILES['configs/d1_preregistration.yaml']}`
- 실행 코드 커밋: `{head}` · 봉인 QQQ 2021-12-31 · 달러 데이터 1998-12-31
- 점검: 모든 구간에서 방식별 총 납입이 같고(120 × B), 현금이 음수가 된 달 없음

## 판정

결론: {conclusion}

판정 칸 = 판정 1(중앙값 +1% 이상) · 2(하위 10% ≥ D0) · 3(이긴 구간 60% 이상):

{md_table(v_rows)}
D6 매수일(참고): **{d6_line}**

## 데이터 기간

{md_table(period_rows)}
## 방식별 결과 (최종 자산 = 현금 포함, QQQ는 원화 세후, 달러 데이터는 달러 세전)

{md_table(method_rows)}
## 최악 구간

{md_table(worst_rows)}
## D6 매수일 비교 (참고, D0 규칙)

{md_table(d6_rows)}
## 결과를 본 뒤 바꾼 점

- 없음 (사전 등록과 구현 해석 그대로 1회 실행).

## 한계

- 현금 이자에 세금을 매기지 않았다(계획이 양도세만 정함) — 현금을 많이 드는 방식(D1~D5)에 약간 유리.
- QQQ 배당은 달러 현금으로 들어와 규칙이 더 사라고 할 때만 쓰인다(구현 해석 11번) — D0은 배당을 끝까지 현금으로 들고 있어 D0에 약간 불리.
- 배당은 "직전 매수일 다음 날 ~ 이번 매수일 전날" 배당락분만 받는다(구현 해석 11번 문구 그대로) — 배당락일이 매수일과 겹치면 그 배당은 빠진다.
- 나스닥은 가격 지수(배당 없음), 미국 시장은 프렌치 총수익 지수. 달러 데이터는 세전·매도 비용 없음.
- 같은 시작 월을 한 달씩 옮긴 10년 구간은 서로 크게 겹친다 — 구간 수만큼 독립 표본이 아니다.

## 그래프

{chr(10).join(f'- `{p.relative_to(ROOT).as_posix()}`' for p in plots)}
"""
    RESULT_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULT_MD.write_text(md, encoding="utf-8")
    print(f"[d1] 결과: {RESULT_MD.relative_to(ROOT)}")
    print(conclusion)
    print("D6:", d6_line)


def main() -> None:
    verify_preregistration()
    prereg = yaml.safe_load((ROOT / "configs" / "d1_preregistration.yaml").read_text(encoding="utf-8"))
    ds = load_datasets(prereg)
    res = run_all_windows(ds)
    st = {k: stats(res[k]) for k in DATASETS}
    vd = verdict(st)
    write_outputs(ds, res, st, vd)


if __name__ == "__main__":
    main()
