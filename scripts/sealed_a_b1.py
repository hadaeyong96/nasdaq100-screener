"""봉인 ⓐ 개봉 — 후보 ① 200일선 2배 전환(B1) 실행 (사전 등록: docs/sealed_a_plan.md, configs/sealed_a.yaml, tag sealed-a-prereg).

일회성 리포트 스크립트. 신호·합성 2배·장부는 core/l1b.py(l1b-prereg 그대로), 금리 환산·판정은 core/sealed_a.py.
시작할 때 사전 등록 파일(봉인 ⓐ 두 파일 + l1b·sw1 근거 파일)의 sha256(LF 정규화)이 태그 내용·스크립트 기록값과
같은지 확인하고 다르면 멈춘다. 가격·금리에 2026-09-30 뒤 행이 있으면 잘라내지 않고 멈춘다.

실행: python -u -m scripts.sealed_a_b1 --check-data   (데이터 검사표만)
      python -u -m scripts.sealed_a_b1                (전체)
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
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

from core import l1b  # noqa: E402
from core import sealed_a as sa  # noqa: E402

PREREG = {  # 태그: {파일: LF 정규화 sha256}
    "sealed-a-prereg": {
        "docs/sealed_a_plan.md": "4660cb88358dab4461e69903d53363c99e504f633e12a7c2d76623263f876db9",
        "configs/sealed_a.yaml": "98a4f5fae84f77b62fb4321090e80c4bb6be41fa1ef9fffdc5b0792caec5e80e",
    },
    "l1b-prereg": {
        "docs/l1b_plan.md": "dd8b8cd6fd17b3f0e0a27a5fdc035ad48a6b3e0331b41e14c3e8d53cf3c7fa71",
        "configs/l1b_preregistration.yaml": "091dbc7c73cf0cd33aefa5a25d966e84f82be8210ebdaa7570c9f2588a0f3816",
    },
    "sw1-prereg": {
        "docs/sw1_plan.md": "8b864411c51104aef01888ff2b99228b6688d4e34a89fcb13f7fbeda140ebfe9",
        "configs/sw1_preregistration.yaml": "c5ad820f50993b91f14642a9ab58aa48f4adf88731dc5f8285d95246d90c730e",
    },
}
CACHE = ROOT / "data" / "cache" / "sealed_a"
RESULT_MD = ROOT / "docs" / "results" / "sealed_a_b1.md"
DATA_START = "2021-01-01"
FETCH_END = "2026-10-01"  # yfinance end는 그날 제외
ASSET_KO = {"1x": "1배", "2x": "2배", "cash": "국채"}


def lf_sha256(b: bytes) -> str:
    """줄바꿈을 LF로 맞춘 내용의 sha256."""
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def verify_preregistration() -> dict:
    """태그 내용 = 작업 폴더 = 기록값 확인. 다르면 멈춤. 출력: configs/sealed_a.yaml 내용"""
    for tag, files in PREREG.items():
        for rel, want in files.items():
            work = lf_sha256((ROOT / rel).read_bytes())
            committed = lf_sha256(subprocess.run(["git", "show", f"{tag}:{rel}"], cwd=ROOT, check=True,
                                                 capture_output=True).stdout)
            if not (work == committed == want):
                raise SystemExit(f"[sealed-a] 사전 등록 해시 불일치: {tag}:{rel} — 멈춤")
    print("[sealed-a] 사전 등록 해시 확인 통과 (sealed-a·l1b·sw1 태그 = 작업 폴더 = 기록값)", flush=True)
    with open(ROOT / "configs" / "sealed_a.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 데이터 ───────────────────────────────────────────────────────────────────


def load_close(ticker: str) -> pd.Series:
    """yfinance auto_adjust=False Close (2021-01-01 ~ 2026-09-30). data/cache/sealed_a/<티커>.csv 캐시. 봉인 검사."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{ticker}.csv"
    if not path.exists():
        import yfinance as yf

        df = yf.download(ticker, start=DATA_START, end=FETCH_END, auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            raise SystemExit(f"[sealed-a] {ticker} 받기 실패 — 멈춤")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].to_csv(path)
    d = pd.read_csv(path, index_col=0, parse_dates=True)
    sa.check_seal(d, ticker)
    return d["Close"].astype(float)


def load_dtb3() -> pd.Series:
    """FRED DTB3 연율(%) 2020-12-01 ~ 2026-09-30. data/cache/sealed_a/DTB3.csv 캐시. 봉인 검사."""
    path = CACHE / "DTB3.csv"
    if not path.exists():
        from data import fx as fxmod

        raw = fxmod.fetch_fred_series_range("DTB3", "2020-12-01", "2026-09-30")
        s = pd.Series(raw, dtype=float)
        s.index = pd.to_datetime(s.index)
        s.sort_index().rename("DTB3").to_csv(path)
    s = pd.read_csv(path, index_col=0, parse_dates=True)["DTB3"].astype(float)
    return sa.check_seal(s, "DTB3")


def gaps(idx: pd.DatetimeIndex) -> str:
    d = pd.Series(idx[1:] - idx[:-1], index=idx[1:])
    big = d[d > pd.Timedelta(days=5)]
    return ", ".join(f"{i.date()}({v.days}일)" for i, v in big.items()) or "없음"


def build(prereg: dict) -> dict:
    """QQQ·QLD 종가, DTB3 → 일별 수익 표(1x·2x·cash)와 참고용 QLD 표. 데이터 검사표도 만든다."""
    qqq, qld, dtb3 = load_close("QQQ"), load_close("QLD"), load_dtb3()
    days = qqq.index
    if qqq.isna().any():
        raise SystemExit("[sealed-a] QQQ 종가 결측 — 멈춤")
    rf = sa.dtb3_daily_rf(dtb3, days)
    if rf.isna().any():
        raise SystemExit("[sealed-a] DTB3로 금리를 못 구한 거래일 — 멈춤")
    r1 = qqq.pct_change().fillna(0.0)
    r2 = l1b.synthetic_2x_returns(r1, rf)
    qld_r = qld.reindex(days).pct_change()
    returns = pd.DataFrame({"1x": r1, "2x": r2, "cash": rf})
    returns_qld = pd.DataFrame({"1x": r1, "2x": qld_r, "cash": rf})
    n2021 = int((days.year == 2021).sum())
    if n2021 < 210:
        raise SystemExit(f"[sealed-a] 2021년 QQQ 행 {n2021}개 < 210 — 멈춤")
    series = [
        {"시리즈": "QQQ Close (auto_adjust=False)", "시작": days[0].date(), "끝": days[-1].date(), "행 수": len(qqq),
         "결측": int(qqq.isna().sum()), "5일 넘는 공백": gaps(days)},
        {"시리즈": "QLD Close (참고)", "시작": qld.index[0].date(), "끝": qld.index[-1].date(), "행 수": len(qld),
         "결측": int(qld.reindex(days).isna().sum()), "5일 넘는 공백": gaps(qld.index)},
        {"시리즈": "FRED DTB3 (연율 %)", "시작": dtb3.index[0].date(), "끝": dtb3.index[-1].date(), "행 수": len(dtb3),
         "결측": int(dtb3.isna().sum()), "5일 넘는 공백": gaps(dtb3.index)},
    ]
    extra = [
        {"항목": "2021년 QQQ 행(워밍업)", "값": n2021, "비고": "≥ 210 필요"},
        {"항목": "봉인 검사(2026-09-30 뒤 행)", "값": 0, "비고": "QQQ·QLD·DTB3 통과"},
        {"항목": "QQQ 거래일 중 DTB3 발표 없는 날", "값": int((~days.isin(dtb3.index)).sum()), "비고": "직전 값 사용"},
        {"항목": "DTB3 범위 2022~2026-09 (연율 %)", "값": f"{dtb3.loc['2022':].min():.2f} ~ {dtb3.loc['2022':].max():.2f}", "비고": ""},
    ]
    return {"qqq": qqq, "returns": returns, "returns_qld": returns_qld, "series": series, "extra": extra}


# ── 실행 ─────────────────────────────────────────────────────────────────────


def run(level: pd.Series, returns: pd.DataFrame, cand: str, start, end, sma_days: int = 200, pretax: bool = False) -> dict:
    """후보(B0/B1) 장부 → 보고 지표. 구간 첫날 보유 = 전날 신호 목표(워밍업부터 이어진 신호)."""
    target = l1b.target_b0(level) if cand == "B0" else l1b.target_b1(level, sma_days)
    held = l1b.held_after_close_daily(target).loc[pd.Timestamp(start):pd.Timestamp(end)]
    sim = l1b.simulate(held, returns)
    pre = l1b.simulate(held, returns, apply_tax=False) if pretax else None
    v = sim["values"]
    dd = l1b.max_drawdown(v)
    rec = l1b.longest_recovery(v)
    yr = l1b.yearly_returns(v)
    return {"posttax": sim["cagr_pct"], "pretax": pre["cagr_pct"] if pre else None, "mdd": dd["mdd_pct"],
            "mdd_span": f"{dd['peak'].date()} → {dd['trough'].date()}", "recovery_rows": rec["rows"],
            "recovery_span": f"{rec['start'].date()} → {rec['end'].date()}" + (" (진행 중)" if rec["open_ended"] else ""),
            "worst_year": f"{yr.idxmin()} {yr.min():.1f}%", "switches": sim["switches"], "tax": sum(sim["taxes"].values()),
            "lev2": float((held == "2x").mean() * 100), "cash": float((held == "cash").mean() * 100),
            "yearly": yr, "sim": sim, "final": sim["after_tax_final"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-data", action="store_true")
    args = ap.parse_args()
    prereg = verify_preregistration()
    b = prereg["B1"]
    ctx = build(prereg)
    print(pd.DataFrame(ctx["series"]).to_string(index=False))
    print(pd.DataFrame(ctx["extra"]).to_string(index=False))
    if args.check_data:
        return
    level, R = ctx["qqq"], ctx["returns"]
    days = level.index
    full_s, full_e = pd.Timestamp(b["periods"]["full"]["start"]), days[days <= pd.Timestamp(b["periods"]["full"]["end"])][-1]
    if days[days >= "2022-01-01"][0] != full_s:
        raise SystemExit("[sealed-a] 첫 거래일이 2022-01-03이 아님 — 멈춤")
    full = {c: run(level, R, c, full_s, full_e, pretax=True) for c in ("B0", "B1")}
    subs = {}
    for sp in b["periods"]["sub"]:
        e = days[days <= pd.Timestamp(sp["end"])][-1]
        subs[sp["name"]] = {c: run(level, R, c, sp["start"], e, pretax=True) for c in ("B0", "B1")}
    nb = {n: run(level, R, "B1", full_s, full_e, sma_days=n, pretax=True) for n in b["criteria"][3]["sma_days"]}
    verdict = sa.b1_verdict(full, subs, {n: r["posttax"] for n, r in nb.items()})
    qld_ref = run(level, ctx["returns_qld"], "B1", full_s, full_e, pretax=True)
    bear = sa.window_report(full["B1"]["sim"]["values"], full["B1"]["sim"]["held"], full["B0"]["sim"]["values"],
                            "2022-01-03", "2022-12-30")
    print("판정:", "PASS" if verdict["pass"] else "FAIL", verdict["ok"])
    write_report(ctx, full, subs, nb, verdict, qld_ref, bear, full_s, full_e)


# ── 보고서 ───────────────────────────────────────────────────────────────────


def _f(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    if isinstance(x, (float, np.floating)):
        return f"{x:,.{nd}f}"
    return str(x)


def md_table(rows: list[dict]) -> str:
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(_f(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def cand_row(name: str, r: dict, b0: dict | None) -> dict:
    return {"후보": name, "세후 연복리 %": r["posttax"], "세전 연복리 %": r["pretax"],
            "B0 대비 세후 %p": (r["posttax"] - b0["posttax"]) if b0 else None, "최대 낙폭 %": r["mdd"],
            "낙폭 고점→저점": r["mdd_span"], "최장 회복(거래일)": r["recovery_rows"], "회복 구간": r["recovery_span"],
            "최악의 해": r["worst_year"], "전환": r["switches"], "세금 합계 $": r["tax"], "2배 보유 %": r["lev2"],
            "최종 세후 $": r["final"]}


def write_report(ctx, full, subs, nb, v, qld_ref, bear, s, e) -> None:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    ok = lambda k: "충족" if v["ok"][k] else "미충족"  # noqa: E731
    sub_names = list(subs)
    L = [
        "# 봉인 ⓐ 개봉 — 200일선 2배 전환(B1) 결과 (sealed-a)", "",
        "> 해시 기준 (LF 정규화 sha256, 태그 = 작업 폴더 = 스크립트 기록값 확인 후 실행)",
        f"> - `docs/sealed_a_plan.md` `{PREREG['sealed-a-prereg']['docs/sealed_a_plan.md']}`",
        f"> - `configs/sealed_a.yaml` `{PREREG['sealed-a-prereg']['configs/sealed_a.yaml']}`",
        "> - 근거 태그 `l1b-prereg`(B1 규칙·비용·세금)·`sw1-prereg` 파일 해시도 같은 확인 통과",
        f"> 실행 코드 커밋 `{head}` · 개봉 2026-10-13 · 봉인 2022-01-01 ~ 2026-09-30 (뒤 행 검사 통과)", "",
        f"- 기간: {s.date()} ~ {e.date()} (시작일 종가 매수 = 2021-12-31 신호), 워밍업 2021년 QQQ",
        "- 시장: QQQ 하나. 신호·2배·B0 모두 QQQ 분할 반영 종가(배당 미포함). 2배 = 2 × QQQ 수익 − RF − 0.95%/252, 매일 재조정",
        "- 국채 RF: FRED DTB3, 직전 거래일까지 알려진 값 × 달력 일수/360",
        "- 체결: t일 종가 신호 → t+1 거래일 종가. 비용 편도 0.1%, 세금 연 실현손익 + 국채 이자의 22%(공제·이월 없음), 기간 끝 청산 — l1b 규칙 그대로",
        "",
        f"## 판정: **{'PASS' if v['pass'] else 'FAIL'}**", "",
        "| 번호 | 기준 | 값 | 결과 |", "| --- | --- | --- | --- |",
        f"| ① | 세후 연복리 B0 대비 +2%p 이상 | {v['c1']:+.2f}%p | {ok(1)} |",
        f"| ② | 최대 낙폭이 B0보다 2%p 넘게 깊지 않음 | {v['c2']:+.2f}%p (B1 {full['B1']['mdd']:.2f}% / B0 {full['B0']['mdd']:.2f}%) | {ok(2)} |",
        f"| ③ | {sub_names[0]}, {sub_names[1]} 모두 세후 연복리 > B0 | "
        + " / ".join(f"{k} {subs[k]['B1']['posttax']:.2f}% vs {subs[k]['B0']['posttax']:.2f}%" for k in sub_names)
        + f" | {ok(3)} |",
        f"| ④ | 190·210일 세후가 200일과 1%p 이내 | " + ", ".join(f"{k}일 {d:+.2f}%p" for k, d in v["c4"].items()) + f" | {ok(4)} |",
        "",
        "## 전 기간", "",
        md_table([cand_row("B0 QQQ 1배", full["B0"], None), cand_row("B1 200일선 2배/국채", full["B1"], full["B0"]),
                  cand_row("B1 190일 (④)", nb[190], full["B0"]), cand_row("B1 210일 (④)", nb[210], full["B0"])]),
        "- 세후 연복리: 기간 끝 전량 청산(0.1%)·세금 뒤. 세전: 세금만 끈 장부의 끝 평가액. 최대 낙폭·연도별: 세후 장부 일별 평가액",
        "",
        "## 구간별 (구간 첫 거래일 10만 달러로 새로 시작)", "",
        md_table([{"구간": k, "후보": c, "세후 연복리 %": subs[k][c]["posttax"], "세전 연복리 %": subs[k][c]["pretax"],
                   "최대 낙폭 %": subs[k][c]["mdd"], "전환": subs[k][c]["switches"], "2배 보유 %": subs[k][c]["lev2"]}
                  for k in sub_names for c in ("B0", "B1")]),
        "## 연도별 수익 (세후 장부, %)", "",
    ]
    yrs = sorted(set(full["B0"]["yearly"].index))
    L.append(md_table([{"후보": n, **{str(y): float(r["yearly"].get(y, np.nan)) for y in yrs}}
                       for n, r in (("B0", full["B0"]), ("B1", full["B1"]), ("B1 실제 QLD (참고)", qld_ref))]))
    L.append("- 2026은 1~9월. 연도 수익 = 연말 평가액 / 전년 말 − 1 (세금 납부로 줄어든 것 포함)\n")
    moves = "; ".join(f"{d.date()}→{ASSET_KO[a]}" for d, a in bear["moves"]) or "없음"
    L += [
        "## 2022 약세장 구간 따로 (2022-01-03 ~ 2022-12-30)", "",
        "| 항목 | B1 | B0 |", "| --- | --- | --- |",
        f"| 구간 손실 (시작 전날 대비) | {bear['loss_pct']:.2f}% | {bear['b0_loss_pct']:.2f}% |",
        f"| 구간 안 최대 낙폭 | {bear['mdd_pct']:.2f}% | {bear['b0_mdd_pct']:.2f}% |",
        f"| 보유 비율 2배 / 국채 | {bear['share']['2x']:.1f}% / {bear['share']['cash']:.1f}% | 1배 100% |",
        f"| 구간 안 전환(체결일→자산) | {moves} | 없음 |", "",
        "## 참고 보고 (판정 제외) — 2배 자리에 실제 QLD", "",
        md_table([cand_row("B1 실제 QLD (auto_adjust=False Close)", qld_ref, full["B0"]),
                  cand_row("B1 합성 2배 (판정용)", full["B1"], full["B0"])]),
        "- QLD 종가는 배당 미포함, 보수·차입 비용은 가격에 들어 있음. 신호·국채·비용·세금은 같음",
        "",
        "## 데이터 검사", "", md_table(ctx["series"]), md_table(ctx["extra"]),
        "## 함께 볼 사실 — L1d 해외 재현 실패", "",
        "- B1(L1d의 G1)은 L1d에서 해외 5개 시장 재현에 실패했다 (`docs/results/l1d.md`). 이 봉인 구간 결과는 미국 나스닥(QQQ) 한 시장·4.7년 한 경로뿐이라, 통과·미통과와 관계없이 일반화 근거로는 약하다.",
        "- B1은 L1b(N 1972~98, U 1927~98)에서 합격했고 (`docs/results/l1b.md`), 이번이 봉인 ⓐ 1회 개봉이다. 같은 구간을 다시 판정에 쓰지 않는다.",
        "",
        "## 한계", "",
        "- 가상 2배 상품(판정용), 배당 미포함(B0·B1 동일), 다음 날 종가 체결, 달러 기준 세금 근사(원화 환산·공제 없음), 국채 이자 과세",
        "- 기간 4.7년, 약세장 1번(2022) — 표본이 매우 적다",
        "",
        "## 결과를 본 뒤 바꾼 점", "", "없음", "",
    ]
    RESULT_MD.write_text("\n".join(L), encoding="utf-8", newline="\n")
    print(f"[sealed-a] 보고서 → {RESULT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
