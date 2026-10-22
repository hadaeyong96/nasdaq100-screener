"""실패 구간 분석기 5단계 — 「RSI·MACD·일목 신호 전략」 매매 하나하나 (sim_rsi_macd_ichimoku.py 결과 그대로).

매매: 매수 줄(A1·B)과 같은 종목·같은 진입일(entry_date)의 매도 줄을 잇는다. 끝까지 열린 매매는 2021-12-31 종가로 평가(청산 이유 "기타(끝까지 보유)").
진입 계기: A1 = "RSI 30 탈출". B = 세 조건 묶음(일목: 종가>구름 위·미래 구름 양·후행스팬 / MACD: 골든크로스·S등급 / RSI: 50≤RSI<70)
          중 신호일까지 이어진 날 수가 가장 짧은 것(= 마지막으로 켜진 것).
청산 이유: STOP → 손절, E1 → MACD 데드크로스, E2 → RSI 50 이탈, E3·A1 만료·세금 매도 → 기타.
수익: 매도 금액 ÷ 매수 금액 − 1(수수료 포함). QQQ 대비 초과 = 수익 − 같은 체결일 사이 QQQ(분배금 재투자) 수익.
MAE·MFE: 보유 중 종가 기준 매수가 대비 최저·최고. 국면: factory.regimes(QQQ 종가, 신호일). 업종: data/cache/sectors.parquet(현재 GICS, 없으면 미상).
지표 백분위: 신호일 전 달 말의 3단계 지표 백분위(analyze.py).
무작위 대조: 매매마다 같은 체결일 명단(전 달 말 나스닥100)에서 체결일·같은 보유일 뒤 종가가 있는 종목 하나를 무작위로 — 1,000회(시드 20261022),
           평균 초과(종가 기준, 수수료 없음)를 실제 매매의 종가 기준 평균 초과와 비교.
청산 대조: 같은 진입에 고정 20·60거래일 보유(체결일 종가 → n거래일 뒤 종가, 끝을 넘으면 마지막 날).
출력: data/cache/failure/deep_rsi.json, exports/failure/RSI_MACD_일목_매매.xlsx
실행: python scripts/failure/deep_rsi.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
for p in (ROOT, ROOT / "scripts", ROOT / "scripts" / "failure"):
    sys.path.insert(0, str(p))

import analyze as A  # noqa: E402
import sim_rsi_macd_ichimoku as sim  # noqa: E402  (read_guarded·members·PX — 모듈 수준 실행 없음)
from core.indicators import compute_indicators  # noqa: E402
from engine.daily import load_config  # noqa: E402
from factory import regimes  # noqa: E402

CUR = A.CURVES
SEED = 20261022
N_RANDOM = 1000
EXIT = {"STOP": "손절", "E1": "MACD 데드크로스", "E2": "RSI 50 이탈", "E3": "기타(일목 구름 이탈)", "A1_EXPIRE": "기타(A1 만료)",
        "TAX": "기타(세금 매도)", "OPEN": "기타(끝까지 보유)"}


def run_len(b: pd.Series, d) -> int:
    """d까지 참이 이어진 날 수(그날 거짓이면 0)."""
    s = b.loc[:d]
    n = 0
    for x in s.to_numpy()[::-1]:
        if not x:
            break
        n += 1
    return n


def trigger_b(df: pd.DataFrame, d, cfg) -> str:
    s_min = cfg["assumptions"]["s_grade_macd_norm_min_pct"]
    ichi = (df["close"] > df["cloud_top"]) & df["future_yang"].astype(bool) & df["chikou_ok"].astype(bool)
    macd = df["gc"].astype(bool) & (df["macd_norm"] >= s_min)
    rsi = (df["rsi"] >= 50) & (df["rsi"] < 70)
    lens = {"일목 조건": run_len(ichi, d), "MACD 골든크로스": run_len(macd, d), "RSI 50~70": run_len(rsi, d)}
    lens = {k: v for k, v in lens.items() if v > 0}
    return min(lens, key=lambda k: (lens[k], k)) if lens else "판단 불가"


def main() -> None:
    cfg = load_config()
    raw = pd.read_csv(CUR / "rsi_trades_raw.csv", parse_dates=["signal_date", "fill_date", "entry_date"])
    buys = raw[raw["kind"].isin(["A1", "B"]) & (raw["qty"] > 0)].copy()
    sells = raw[raw["kind"].isin(["STOP", "E1", "E2", "E3", "A1_EXPIRE", "TAX"]) & (raw["qty"] > 0)].copy()
    files = {p.stem: p for p in sim.PX.glob("*.csv")}
    px = {t: sim.read_guarded(files[t]) for t in sorted(set(buys["ticker"])) if t in files}
    q = sim.read_guarded(files["QQQ"])
    days = q.index[q.index >= pd.Timestamp("2011-01-01")]
    qtr = ((q["close"] + q["div"]) / q["close"].shift(1)).fillna(1.0).cumprod()
    reg = regimes.classify(q["close"])
    ind, shapes = A.indicators()
    pcs = A.pct_rank(ind.join(shapes["NDX"], how="outer"))
    key_cols = ["VIX", "10년 금리 6개월 변화", "동일비중−QQQ 6개월(QQQE)", "vol20", "dd", "신용 스프레드(BAA−10년)"]
    try:
        sect = pd.read_parquet(Path(r"C:\projects\nasdaq100-screener\data\cache\sectors.parquet"))
        sect = sect.iloc[:, 0] if isinstance(sect, pd.DataFrame) else sect
        sect = sect.to_dict()
    except Exception:  # noqa: BLE001
        sect = {}
    frames = {}
    rows = []
    for _, b in buys.iterrows():
        t = b["ticker"]
        s = sells[(sells["ticker"] == t) & (sells["entry_date"] == b["fill_date"])]
        if t not in px:
            continue
        p = px[t]["close"]
        if len(s):  # 세금 매도로 일부를 먼저 판 경우도 있어 그 진입의 매도 금액을 모두 더한다
            exit_amt = float(s["amount"].sum())
            s = s.sort_values("fill_date").iloc[-1]
            exit_d, kind, sell_px = s["fill_date"], s["kind"], float(s["price"])
        else:
            exit_d = p.index[p.index <= sim.LAST][-1]
            sell_px = float(p.loc[exit_d])
            exit_amt = float(b["qty"]) * sell_px * (1 - 0.0007)
            kind = "OPEN"
        ret = exit_amt / float(b["amount"]) - 1.0
        gross = sell_px / float(b["price"]) - 1.0
        qret = float(qtr.loc[exit_d] / qtr.loc[b["fill_date"]] - 1.0)
        seg = p[(p.index >= b["fill_date"]) & (p.index <= exit_d)]
        if b["kind"] == "A1":
            trig = "RSI 30 탈출"
        else:
            if t not in frames:
                frames[t] = compute_indicators(px[t][["open", "high", "low", "close", "volume"]], cfg)
            trig = trigger_b(frames[t], b["signal_date"], cfg)
        prev_m = b["signal_date"].to_period("M") - 1
        r = reg.loc[:b["signal_date"]]
        rows.append({"종목": t, "진입 신호일": b["signal_date"].date(), "진입일(체결)": b["fill_date"].date(), "진입 종류": b["kind"],
                     "진입 계기": trig, "청산일": exit_d.date(), "청산 이유": EXIT.get(kind, kind),
                     "보유일(거래일)": int(((q.index > b["fill_date"]) & (q.index <= exit_d)).sum()),
                     "수익": ret, "수익(종가 기준)": gross, "QQQ 같은 기간": qret, "QQQ 대비 초과": ret - qret, "초과(종가 기준)": gross - qret,
                     "MAE": float(seg.min() / b["price"] - 1) if len(seg) else np.nan, "MFE": float(seg.max() / b["price"] - 1) if len(seg) else np.nan,
                     "국면": r.iloc[-1] if len(r) else "판단불가", "업종": sect.get(t) or "미상", "연도": b["signal_date"].year,
                     **{f"백분위:{c}": (float(pcs.at[prev_m, c]) if prev_m in pcs.index and not pd.isna(pcs.at[prev_m, c]) else np.nan)
                        for c in key_cols}})
    tr = pd.DataFrame(rows)

    def grp(col):
        g = tr.groupby(col).agg(매매=("QQQ 대비 초과", "size"), 평균_초과=("QQQ 대비 초과", "mean"), 중앙_초과=("QQQ 대비 초과", "median"),
                                승률=("수익", lambda x: float((x > 0).mean())), 초과_승률=("QQQ 대비 초과", lambda x: float((x > 0).mean())),
                                평균_보유일=("보유일(거래일)", "mean"), 평균_MFE=("MFE", "mean"), 평균_MAE=("MAE", "mean"))
        return g.reset_index()

    groups = {k: grp(k) for k in ("진입 계기", "청산 이유", "국면", "업종", "연도", "진입 종류")}
    # 무작위 대조(같은 날 같은 명단, 같은 보유일)
    mem = sim.members()
    me_keys = sorted(mem)
    tick = sorted({x for v in mem.values() for x in v if x in files})
    close = pd.DataFrame({t: sim.read_guarded(files[t])["close"] for t in tick}).reindex(q.index)
    qi = {d: i for i, d in enumerate(q.index)}
    cm = close.to_numpy()
    qv = qtr.reindex(q.index).to_numpy()
    cand_idx, act_ex, fi_list, xi_list = [], [], [], []
    col = {t: j for j, t in enumerate(tick)}
    for _, r in tr.iterrows():
        f = qi[pd.Timestamp(r["진입일(체결)"])]
        x = min(f + int(r["보유일(거래일)"]), len(q.index) - 1)
        me = max(k for k in me_keys if k < pd.Timestamp(r["진입일(체결)"]).replace(day=1))
        js = [col[t] for t in mem[me] if t in col and not np.isnan(cm[f, col[t]]) and not np.isnan(cm[x, col[t]])]
        if not js:
            continue
        cand_idx.append(np.array(js)); fi_list.append(f); xi_list.append(x); act_ex.append(r["초과(종가 기준)"])
    rng = np.random.default_rng(SEED)
    fi, xi = np.array(fi_list), np.array(xi_list)
    qret = qv[xi] / qv[fi] - 1
    rand_means = []
    for _ in range(N_RANDOM):
        pick = np.array([c[rng.integers(len(c))] for c in cand_idx])
        rr = cm[xi, pick] / cm[fi, pick] - 1
        rand_means.append(float(np.mean(rr - qret)))
    rand_means = np.array(rand_means)
    actual_mean = float(np.mean(act_ex))
    pct = float(100 * ((rand_means < actual_mean).sum() + 0.5 * (rand_means == actual_mean).sum()) / N_RANDOM)
    # 청산 대조(고정 20·60거래일)
    fixed = {}
    for n in (20, 60):
        ex, rt = [], []
        for _, r in tr.iterrows():
            f = qi[pd.Timestamp(r["진입일(체결)"])]
            x = min(f + n, len(q.index) - 1)
            c = cm[:, col[r["종목"]]] if r["종목"] in col else None
            if c is None or np.isnan(c[f]):
                continue
            last = x
            while last > f and np.isnan(c[last]):
                last -= 1
            g = c[last] / c[f] - 1
            rt.append(g); ex.append(g - (qv[last] / qv[f] - 1))
        fixed[f"고정 {n}거래일"] = {"평균 수익": float(np.mean(rt)), "평균 초과": float(np.mean(ex)), "초과 승률": float((np.array(ex) > 0).mean()), "매매": len(ex)}
    actual = {"평균 수익(종가 기준)": float(tr["수익(종가 기준)"].mean()), "평균 초과(종가 기준)": actual_mean,
              "초과 승률": float((tr["초과(종가 기준)"] > 0).mean()), "평균 보유일": float(tr["보유일(거래일)"].mean()), "매매": len(tr)}
    # 이익을 잘랐나: MFE가 +10% 넘었는데 끝 수익이 MFE의 1/3 미만이었던 비율
    big = tr[tr["MFE"] >= 0.10]
    cut = float((big["수익(종가 기준)"] < big["MFE"] / 3).mean()) if len(big) else np.nan
    out = {"actual": actual, "random": {"평균 초과 백분위": pct, "무작위 평균 초과 중앙값": float(np.median(rand_means)),
                                       "무작위 95백분위": float(np.percentile(rand_means, 95)), "회수": N_RANDOM, "비교 매매 수": len(act_ex)},
           "fixed": fixed, "mfe_cut": {"MFE ≥ +10% 매매": int(len(big)), "그중 끝 수익 < MFE의 1/3 비율": cut,
                                        "평균 MFE": float(tr["MFE"].mean()), "평균 MAE": float(tr["MAE"].mean())},
           "groups": {k: v.to_dict(orient="records") for k, v in groups.items()},
           "sector_unknown_share": float((tr["업종"] == "미상").mean())}
    (A.C / "deep_rsi.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    with pd.ExcelWriter(A.EXP / "RSI_MACD_일목_매매.xlsx") as w:
        tr.to_excel(w, sheet_name="매매 한 줄씩", index=False)
        for k, v in groups.items():
            v.to_excel(w, sheet_name=f"{k}별", index=False)
        pd.DataFrame([{"구분": "실제 청산", **actual}] + [{"구분": k, **v} for k, v in fixed.items()]).to_excel(w, sheet_name="청산 대조", index=False)
        pd.DataFrame({"무작위 평균 초과": rand_means}).to_excel(w, sheet_name="무작위 대조", index=False)
    print("done", json.dumps({k: out[k] for k in ("actual", "random", "fixed", "mfe_cut")}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
