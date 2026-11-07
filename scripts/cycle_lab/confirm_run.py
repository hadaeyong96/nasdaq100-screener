"""사이클 실험실 2단계 — 실행기 (docs/prereg/cycle_lab_confirm.md, 태그 cycle-lab-confirm-prereg).

실행:
  python -m scripts.cycle_lab.confirm_run --fetch --pushed-at <시각>   # 데이터 받기(태그 push 뒤에만) + 계산
  python -m scripts.cycle_lab.confirm_run --smoke-fetch <폴더>         # 태그 전 수신 경로 점검(2012-12까지만, 계산 없음)
출력: docs/results/cycle_lab/v2/ — 이미 있으면 덮어쓰지 않고 멈춘다. 재실행·규칙 변경 없음.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.cycle_lab import confirm_data as cd  # noqa: E402
from scripts.cycle_lab import confirm_lab as cl  # noqa: E402
from scripts.cycle_lab.confirm_gate import CUTOFF, assert_cutoff  # noqa: E402
from scripts.cycle_lab.run import fmt, md_table  # noqa: E402

OUT = ROOT / "docs" / "results" / "cycle_lab" / "v2"
TAG = "cycle-lab-confirm-prereg"
JUDGE = pd.date_range("1946-01-31", CUTOFF, freq="ME")
METHOD_NAME = {"M2": "M2 신용", "M5": "M5 깊은 바닥", "M6": "M6 상대 강도(대조군)"}


def log(msg: str) -> None:
    print(msg, flush=True)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def check_order() -> None:
    """태그가 원격에 있고 HEAD가 태그 커밋이며 작업 트리가 깨끗할 때만 통과. 아니면 SystemExit."""
    if OUT.exists():
        raise SystemExit(f"BLOCKED — {OUT.relative_to(ROOT)} 이미 있음(재실행 금지)")
    remote = git("ls-remote", "origin", f"refs/tags/{TAG}")
    if not remote:
        raise SystemExit(f"BLOCKED — 원격에 태그 {TAG} 없음")
    if git("rev-list", "-n", "1", TAG) != git("rev-parse", "HEAD"):
        raise SystemExit("BLOCKED — HEAD가 태그 커밋이 아님")
    if git("status", "--porcelain"):
        raise SystemExit("BLOCKED — 작업 트리가 깨끗하지 않음")


def fetch_all(cache: Path, end: str, state_path: Path | None) -> None:
    """모든 입력 받기. 실패 목록을 모아 보고한다. 첫 수신 시각을 state_path에 남긴다."""
    jobs = [("French ind49", lambda: cd.fetch_french(cache, end))]
    jobs += [(f"ALFRED {s}", lambda s=s: cd.fetch_alfred(s, cache, end)) for s in cd.ALFRED_IDS]
    jobs += [(cd.FX_ID, lambda: cd.fetch_fx(cache, end))]
    jobs += [(t, lambda t=t: cd.fetch_yf(t, cache, end)) for t in cd.YF_TICKERS]
    failures = []
    state: dict = json.loads(state_path.read_text(encoding="utf-8")) if state_path is not None and state_path.exists() else {}
    for name, job in jobs:
        try:
            meta = job()
            if state_path is not None and "first_data_received_utc" not in state:
                state["first_data_received_utc"] = meta["fetched_utc"]
                state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
            log(f"받음 {name}: {meta['first']} ~ {meta['last']} ({meta['rows']}행)")
        except Exception as exc:  # noqa: BLE001 — 실패 목록을 모아 보고
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            log(f"실패 {name}: {type(exc).__name__}: {exc}")
    if failures:
        raise SystemExit("BLOCKED — 데이터 수신 실패: " + "; ".join(failures))


def load_inputs(cache: Path = cd.CACHE, allow_current_fetch: bool = True) -> dict:
    """캐시 → 계산 입력. 입력 최대 날짜를 assert한다. 출력: ind, spread, levels, qqq_daily, ndx_daily, fx, maxdates."""
    ind, fx_daily = cd.load_french(cache), cd.load_fx(cache)
    qqq, ndx = cd.load_yf("QQQ", cache), cd.load_yf("^NDX", cache)
    alfred = {s: cd.load_alfred(s, cache) for s in cd.ALFRED_IDS}
    rt = {}
    for s in cd.ALFRED_IDS:
        rt[s] = cl.realtime_values(alfred[s], None, JUDGE)
        if cl.has_gap(rt[s]["level"]) and allow_current_fetch:
            log(f"{s}: 1·2순위에 빈 달 — 3순위 현재값을 받습니다")
            rt[s] = cl.realtime_values(alfred[s], cd.fetch_current(s, cache), JUDGE)
    maxdates = {"french_ind49": ind.index.max(), cd.FX_ID: fx_daily.index.max(), "QQQ": qqq.index.max(), "^NDX": ndx.index.max()}
    for s in cd.ALFRED_IDS:
        maxdates[f"alfred_{s}(obs)"] = alfred[s]["obs"].max()
        maxdates[f"alfred_{s}(vintage)"] = alfred[s]["realtime_end"].max()
    bad = {k: v for k, v in maxdates.items() if pd.Timestamp(v) > CUTOFF}
    assert not bad, f"2021-12-31 뒤 날짜: {bad}"
    return {"ind": ind, "spread": rt["BAA"]["v2"] - rt["GS10"]["v2"], "rt": rt, "qqq_daily": qqq, "ndx_daily": ndx,
            "fx": cl.month_end_fx(fx_daily), "maxdates": maxdates}


def rank_rates(rt: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """시리즈별 대체 순위 비율(전체 판정 달·2013-01 이후 판정 달)."""
    rows = []
    for s, t in rt.items():
        lv = t["level"].dropna()
        ev = lv.loc[cl.FIRST_JUDGE:]
        rows.append({"series": s, "months": len(lv), "first_judge": str(lv.index.min().date()),
                     **{f"level{k}": float((lv == k).mean()) for k in (1, 2, 3)},
                     "eval_months": len(ev), **{f"eval_level{k}": float((ev == k).mean()) if len(ev) else np.nan for k in (1, 2, 3)}})
    return pd.DataFrame(rows)


def run_all(inputs: dict, n_random: int = cl.N_RANDOM) -> dict:
    """판정(보정값)과 참고(무보정) 계산. 출력: main, raw, corrections(2013-01 이후 바꾼 날 표)."""
    fixed, table = cl.qqq_monthly_total(inputs["qqq_daily"], inputs["ndx_daily"])
    raw, _ = cl.qqq_monthly_total(inputs["qqq_daily"], inputs["ndx_daily"], max_diff=None)
    assert_cutoff(fixed, raw)
    if len(fixed.loc[cl.EVAL_START:]) != len(cl.EVAL_MONTHS) or fixed.loc[cl.EVAL_START:].isna().any():
        raise SystemExit("BLOCKED — QQQ 월간 총수익 결측")
    main_res = cl.compute(inputs["ind"], inputs["spread"], fixed, inputs["fx"], n_random)
    raw_res = cl.compute(inputs["ind"], inputs["spread"], raw, inputs["fx"], n_random)
    return {"main": main_res, "raw": raw_res, "corrections": table.loc[cl.EVAL_START:]}


def cells_frame(res: dict, prefix: str = "") -> pd.DataFrame:
    """칸 결과 표(cells.csv 재료)."""
    rows = []
    for no, r in res["cells"].items():
        h = res["holm"][no]
        rows.append({"cell": no, "industry": r["industry"], "method": r["method"], "entries": r["entries"], "avg_weight": r["avg_weight"],
                     "avg_hold_months": r["avg_hold"], "cagr_krw_after_tax": r["cagr"], "qqq_cagr": r["qqq_cagr"], "constant_cagr": r["const_cagr"],
                     "end_krw": r["end_krw"], "qqq_end_krw": r["qqq_end_krw"], "constant_end_krw": r["const_end_krw"],
                     "mdd": r["mdd"], "qqq_mdd": r["qqq_mdd"], "mdd_diff": r["mdd_diff"], "taxes_paid_krw": r["taxes_krw"], "costs_krw": r["costs_krw"],
                     "p": r["p"], "pctile": r["pctile"], "holm_rank": h["rank"], "holm_threshold": h["threshold"],
                     "c1_beats_qqq": r["c1_beats_qqq"], "c2_beats_constant": r["c2_beats_const"], "c3_holm": r["c3_holm"], "c4_mdd_ok": r["c4_mdd_ok"],
                     "passed": r["passed"]})
    df = pd.DataFrame(rows)
    return df.add_prefix(prefix) if prefix else df


def ox(x: bool) -> str:
    return "O" if x else "X"


def write_outputs(out: Path, all_res: dict, inputs: dict, manifest: list[dict], order: dict, v1_diff: dict) -> None:
    """산출물 5개를 쓴다. 입력: 출력 폴더, run_all 결과, 입력, 캐시 meta, 순서 증빙, French 비교."""
    main_res, raw_res, corr = all_res["main"], all_res["raw"], all_res["corrections"]
    out.mkdir(parents=True, exist_ok=False)
    cells = cells_frame(main_res)
    raw_cells = cells_frame(raw_res)
    cells.assign(basis="보정(판정)").pipe(lambda d: pd.concat([d, raw_cells.assign(basis="무보정(참고)")])).to_csv(
        out / "cells.csv", index=False, encoding="utf-8-sig")

    curves = {"month": [m.strftime("%Y-%m") for m in main_res["cells"][1]["book_months"]],
              "qqq": main_res["cells"][1]["curves"]["qqq"]}
    for no, r in main_res["cells"].items():
        curves[f"cell{no}_{r['industry']}_{r['method']}_strategy"] = r["curves"]["strategy"]
        curves[f"cell{no}_{r['industry']}_{r['method']}_constant"] = r["curves"]["constant"]
    pd.DataFrame(curves).to_csv(out / "equity_curves.csv", index=False, encoding="utf-8-sig")

    ent = [{"cell": no, "industry": r["industry"], "method": r["method"], "signal": e["signal"].strftime("%Y-%m"),
            "first_buy": e["start"].strftime("%Y-%m"), "months_held": e["months"]}
           for no, r in main_res["cells"].items() for e in r["entries_list"]]
    pd.DataFrame(ent, columns=["cell", "industry", "method", "signal", "first_buy", "months_held"]).to_csv(
        out / "entries.csv", index=False, encoding="utf-8-sig")

    rates = rank_rates(inputs["rt"])
    man = pd.DataFrame(manifest)
    txt = {r.series: f"전체 1순위 {fmt(r.level1, d=1)}·2순위 {fmt(r.level2, d=1)}·3순위 {fmt(r.level3, d=1)}; "
                     f"2013-01 이후 판정 1순위 {fmt(r.eval_level1, d=1)}·2순위 {fmt(r.eval_level2, d=1)}·3순위 {fmt(r.eval_level3, d=1)}"
           for r in rates.itertuples()}
    man["fallback"] = [txt.get(n.replace("alfred_", ""), "") if n.startswith("alfred_") else "" for n in man["name"]]
    man[["name", "source", "first", "last", "rows", "sha256", "fetched_utc", "cutoff", "fallback"]].to_csv(
        out / "data_manifest.csv", index=False, encoding="utf-8-sig")

    def crit_table(res: dict) -> str:
        return md_table(pd.DataFrame([{
            "칸": f"{no} {r['industry']}×{METHOD_NAME[r['method']]}", "진입": r["entries"], "평균 비중": fmt(r["avg_weight"], d=1),
            "세후 연복리": fmt(r["cagr"]), "QQQ": fmt(r["qqq_cagr"]), "상시 보유": fmt(r["const_cagr"]), "MDD": fmt(r["mdd"]), "QQQ MDD": fmt(r["qqq_mdd"]),
            "낙폭 차이(양수=얕음)": fmt(r["mdd_diff"]), "백분위": "" if not np.isfinite(r["pctile"]) else f"{r['pctile']:.1f}", "p": f"{r['p']:.3f}",
            "① >QQQ": ox(r["c1_beats_qqq"]), "② >상시": ox(r["c2_beats_const"]), "③ Holm": ox(r["c3_holm"]), "④ 낙폭": ox(r["c4_mdd_ok"]),
            "합격": ox(r["passed"])} for no, r in res["cells"].items()]))

    def holm_table(res: dict) -> str:
        rows = sorted(res["holm"].items(), key=lambda kv: kv[1]["rank"])
        return md_table(pd.DataFrame([{"순서": h["rank"], "칸": f"{no} {res['cells'][no]['industry']}×{res['cells'][no]['method']}",
                                       "p": f"{res['cells'][no]['p']:.3f}", "기준": f"{h['threshold']:.4f}", "통과": ox(h["passed"])} for no, h in rows]))

    if len(corr):
        corr_txt = md_table(pd.DataFrame({"날짜": [d.strftime("%Y-%m-%d") for d in corr.index], "QQQ": [fmt(x) for x in corr["qqq"]], "^NDX": [fmt(x) for x in corr["ndx"]]}))
    else:
        corr_txt = "(바꾼 날 없음)"
    overall = max(pd.Timestamp(v) for v in inputs["maxdates"].values())
    same = main_res["verdict"]["result"] == raw_res["verdict"]["result"] and all(
        main_res["cells"][n]["passed"] == raw_res["cells"][n]["passed"] for n in main_res["cells"])
    rates_show = pd.DataFrame({"시리즈": rates["series"], "판정 달 수": rates["months"], "첫 판정 달": rates["first_judge"],
                               "1순위": [fmt(x, d=0) for x in rates["level1"]], "2순위(2021-12-31 vintage)": [fmt(x, d=0) for x in rates["level2"]],
                               "3순위(현재값)": [fmt(x, d=0) for x in rates["level3"]],
                               "2013-01 이후 1순위": [fmt(x, d=0) for x in rates["eval_level1"]], "2013-01 이후 2순위": [fmt(x, d=0) for x in rates["eval_level2"]],
                               "2013-01 이후 3순위": [fmt(x, d=0) for x in rates["eval_level3"]]})
    v = main_res["verdict"]
    report = "\n".join([
        "# 사이클 산업 예측 실험실 — 2단계(확인) 보고서", "",
        "연구 작업이며 투자 조언이 아니다. 실제 돈 변경 제안 없음. 2022-01-01 이후 데이터는 받지도 계산하지도 않았다.", "",
        f"## 최종 판정: {v['result']}", "", f"{v['kind']}.", "",
        "## 순서 증빙", "",
        f"- 사전 등록 커밋(태그 `{TAG}` = {order['tag_commit'][:7]}) 시각: {order['commit_time']}",
        f"- push 시각: {order['pushed_at']}",
        f"- 첫 데이터 수신 시각(UTC): {order['first_received']}",
        f"- 입력 최대 날짜 assert: 통과 — 전체 최대 {overall.date()} ≤ 2021-12-31 (입력 {len(inputs['maxdates'])}개)", "",
        "## 칸별 합격 기준 (판정: QQQ 보정값, 원화 세후, 2013-01 ~ 2021-12)", "", crit_table(main_res), "",
        "## Holm (m=3, α=0.05)", "", holm_table(main_res), "",
        "## 참고: QQQ 무보정 (판정에 쓰지 않음)", "",
        f"두 결과의 칸별 합격이 {'같다' if same else '**다르다**'}. 사전 등록대로 판정은 바꾸지 않는다.", "", crit_table(raw_res), "",
        "## QQQ 보정 날짜 (2013-01 이후)", "", f"보정 일수: {len(corr)}일", "", corr_txt, "",
        "## 대체 비율", "", md_table(rates_show), "",
        "## 설정", "",
        "- 시작 1억 원, 환율 DEXKOUS(그 달 말 이하 최근 값), 비용 편도 0.1%, 세금 `core/tax.py`(원화 실현손익 통산 − 250만 원, 22%, 다음 해 5월 말 납부).",
        "- 배당 원천징수: 비교 대상 모두 미적용(사용자 결정). `research/mu-cycle`은 15% 적용이라 설정이 다르다.",
        "- 산업 포트폴리오의 직접 매매 가능성은 판정하지 않는다.",
        f"- French 2012-12 이전 값과 1단계 캐시 비교: {v1_diff['compared']}칸 중 {v1_diff['different']}칸 다름, 최대 차이 {v1_diff['max_abs_pct']:.4f}%p (보고만).", "",
        "## 편차·기대", "",
        "- 1단계 편차와 확인 내용: `docs/prereg/cycle_lab_discovery_deviation.md`. PerSv×M5는 보정 없이는 1단계 자격 미달(89.5백분위)이었다.",
        "- 사전 기대: FAIL 가능성이 높음(1단계 BH q=0.56, 2013~2021은 다른 연구에서도 쓰인 구간).", "",
        "## 파일", "", "`cells.csv`(보정·무보정), `equity_curves.csv`(월말 그날 전부 팔면 세후 값), `entries.csv`, `data_manifest.csv`", ""])
    (out / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--pushed-at", default="")
    ap.add_argument("--smoke-fetch", default="")
    args = ap.parse_args()

    if args.smoke_fetch:
        fetch_all(Path(args.smoke_fetch), "2012-12-31", None)
        log("수신 경로 점검 끝(2012-12까지만, 계산 없음)")
        return

    check_order()
    cd.CACHE.mkdir(parents=True, exist_ok=True)
    state_path = cd.CACHE / "run_log.json"
    if args.fetch:
        fetch_all(cd.CACHE, cl.CUTOFF.strftime("%Y-%m-%d"), state_path)
    inputs = load_inputs()
    log("입력 최대 날짜 assert 통과")
    all_res = run_all(inputs)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    order = {"tag_commit": git("rev-list", "-n", "1", TAG), "commit_time": git("log", "-1", "--format=%cI", TAG),
             "pushed_at": args.pushed_at or "(인자 없음)", "first_received": state.get("first_data_received_utc", "(기록 없음)")}
    write_outputs(OUT, all_res, inputs, cd.manifest_rows(), order, cd.compare_v1(inputs["ind"]))
    log(cells_frame(all_res["main"]).to_string())
    log(json.dumps(all_res["main"]["verdict"], ensure_ascii=False))


if __name__ == "__main__":
    main()
