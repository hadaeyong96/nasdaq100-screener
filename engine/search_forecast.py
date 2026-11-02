"""검색량 실적 예고 — 주간 실행 (docs/prereg/search_forecast.md, 구현 해석 docs/prereg/search_forecast_impl.md).

앞으로 기록: 수집 → (분기 끝 + 10일) 점수 잠금 → 시즌이 모이면 예측 잠금 → 실적 뒤 채점. 과거 수익률 계산 없음. 실제 주문 없음.
잠금 파일 docs/forecast_locks/YYYY-Qn.txt에는 시각과 SHA-256만 쓴다(종목·방향·점수는 DB에만).
아침 편지(engine.daily)와 따로 돈다 — 여기서 실패해도 편지에는 영향이 없다.

실행: python -m engine.search_forecast [--db 경로] [--lock-dir 폴더] [--now ISO시각]
미리보기(잠금 없음, 화면 출력만): python -m engine.search_forecast --preview --db <임시 DB>
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from core import search_forecast as sf
from store import search_forecast as st

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "search_forecast.yaml"
LOCK_DIR = ROOT / "docs" / "forecast_locks"


@dataclass
class Sources:
    """네트워크 함수 묶음(테스트에서 바꿔 끼운다)."""
    trends: Callable[[str], pd.Series]
    wiki: Callable[[str], pd.Series]
    edgar: Callable[[str], dict]
    prices: Callable[[str], pd.Series]


def load_config(path=CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def live_sources(cfg: dict, now: pd.Timestamp) -> Sources:
    from data import search_trends as dst

    c = cfg["collect"]
    client = None

    def trends(term):
        nonlocal client
        if client is None:
            client = dst.TrendsClient(cfg["geo"], cfg["timeframe"], c["request_delay_sec"], c["retries"], c["backoff_sec"])
        return client.weekly(term)

    w = cfg["wiki"]
    start = (now - pd.DateOffset(years=2)).date().isoformat()
    return Sources(trends=trends,
                   wiki=lambda a: dst.wiki_monthly(a, w["user_agent"], w["years"], w["project"], now.tz_localize(None)),
                   edgar=lambda t: dst.edgar_bundle(t, cfg["revenue_tags"]),
                   prices=lambda t: dst.closes(t, start))


def _iso(now: pd.Timestamp) -> str:
    return now.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


# ── 1. 수집 ─────────────────────────────────────────────────────────────────


def collect(conn, cfg: dict, src: Sources, now: pd.Timestamp) -> dict:
    """검색어·위키 문서마다 한 번씩 받아 원값 그대로 저장. 실패는 기록만(다음 주에 다시)."""
    at = _iso(now)
    rep = {"terms_ok": 0, "terms_fail": [], "wiki_ok": 0, "wiki_fail": []}
    for tk, c in cfg["companies"].items():
        for term in c["terms"]:
            try:
                n = st.add_trends(conn, tk, term, src.trends(term), at)
                st.log_fetch(conn, "trends", term, at, True, n)
                rep["terms_ok"] += 1
            except Exception as e:  # noqa: BLE001 — 기록만
                st.log_fetch(conn, "trends", term, at, False, 0, f"{type(e).__name__}: {e}")
                rep["terms_fail"].append(term)
            conn.commit()
        for art in c.get("wiki", []):
            try:
                n = st.add_wiki(conn, tk, art, src.wiki(art), at)
                st.log_fetch(conn, "wiki", art, at, True, n)
                rep["wiki_ok"] += 1
            except Exception as e:  # noqa: BLE001
                st.log_fetch(conn, "wiki", art, at, False, 0, f"{type(e).__name__}: {e}")
                rep["wiki_fail"].append(art)
            conn.commit()
    return rep


# ── 2. 점수 ─────────────────────────────────────────────────────────────────


def company_windows(edgar_data: dict, now: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """EDGAR 분기 끝(실제) + 추정 → 분기 창 목록."""
    rev = sf.quarterly_revenue(edgar_data.get("revenue_facts") or [])
    ends = sf.project_quarter_ends(list(rev["end"]), now.tz_localize(None).normalize())
    return sf.quarter_windows(ends)


def score_one(conn, cfg: dict, tk: str, windows, end: pd.Timestamp) -> dict:
    terms = {t: st.latest_pull(conn, t) for t in cfg["companies"][tk]["terms"]}
    upto = [w for w in windows if w[1] <= end]
    r = sf.company_score(terms, upto[-(cfg["baseline_quarters"] + 1):], n_base=cfg["baseline_quarters"],
                         shift_days=cfg["yoy_shift_days"], min_weeks=cfg["min_weeks"], agg=cfg["term_agg"])
    return {"ticker": tk, "fiscal_end": end.date().isoformat(), "season": sf.season_of(end), "quarter_start": upto[-1][0].date().isoformat(),
            **{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}}


def _append_lock(lock_dir: Path, season: str, line: str) -> None:
    lock_dir.mkdir(parents=True, exist_ok=True)
    with open(lock_dir / f"{season}.txt", "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


def lock_scores(conn, cfg: dict, windows_by: dict, now: pd.Timestamp, lock_dir: Path) -> list[dict]:
    """분기 끝 + score_delay_days가 지난(시작 시즌 이후) 분기마다 점수 행을 만들어 잠근다. 같은 분기(±20일)는 한 번만.
    점수를 못 내면(검색량 부족) 잠그지 않고 다음 실행에서 다시 본다."""
    have = st.predictions(conn)
    first = sf.season_end(cfg["start_season"])
    today = now.tz_convert("UTC").tz_localize(None).normalize()
    new = []
    for tk, windows in windows_by.items():
        old = pd.to_datetime(have.loc[have["ticker"] == tk, "fiscal_end"])
        for _, end in windows:
            if sf.season_end(sf.season_of(end)) < first or end + pd.Timedelta(days=cfg["score_delay_days"]) > today:
                continue
            if len(old) and (abs(old - end) <= pd.Timedelta(days=20)).any():
                continue
            row = score_one(conn, cfg, tk, windows, end)
            if row["score"] is None:  # 검색량이 모자라 점수를 못 냄(수집 실패 등) — 잠그지 않고 다음 주에 다시
                print(f"[search_forecast] {tk} {row['fiscal_end']} 점수 없음 — 다음 실행에서 다시")
                continue
            new.append({**row, "computed_at": _iso(now)})
    for season in sorted({r["season"] for r in new}):
        rows = [r for r in new if r["season"] == season]
        digest = sf.lock_hash(rows)
        for r in rows:
            st.add_prediction(conn, {**r, "score_lock_hash": digest})
        _append_lock(lock_dir, season, sf.lock_line(_iso(now), digest))
    conn.commit()
    return new


def lock_seasons(conn, cfg: dict, now: pd.Timestamp, lock_dir: Path) -> list[str]:
    """시즌의 회사 점수가 다 모이거나(또는 시즌 끝 + 점수 지연 + 여유일이 지나면) 3등분 예측을 잠근다.
    이미 잠긴 시즌에 늦게 들어온 행은 '예측 없음'으로 잠근다."""
    p = st.predictions(conn)
    today = now.tz_convert("UTC").tz_localize(None).normalize()
    done = []
    for season, g in p.groupby("season"):
        todo = g[g["prediction"].isna()]
        if todo.empty:
            continue
        already = g["pred_locked_at"].notna().any()
        due = sf.season_end(season) + pd.Timedelta(days=cfg["score_delay_days"] + cfg["season_grace_days"])
        if not already and not (set(cfg["companies"]) <= set(g["ticker"]) or today >= due):
            continue
        if already:
            labels = {(r.ticker, r.fiscal_end): sf.NONE for r in todo.itertuples()}
        else:
            lab = sf.terciles({f"{r.ticker}|{r.fiscal_end}": (np.nan if r.score is None else r.score) for r in g.itertuples()})
            labels = {tuple(k.split("|")): v for k, v in lab.items()}
        at = _iso(now)
        rows = [{**r, "prediction": labels.get((r["ticker"], r["fiscal_end"]), r["prediction"]), "pred_locked_at": at}
                for r in g.drop(columns=["pred_lock_hash"]).to_dict("records") if (r["ticker"], r["fiscal_end"]) in labels]
        digest = sf.lock_hash(rows)
        st.set_season_predictions(conn, labels, at, digest)
        _append_lock(lock_dir, season, sf.lock_line(at, digest))
        done.append(season)
    conn.commit()
    return done


# ── 3. 채점 ─────────────────────────────────────────────────────────────────


def grade(conn, cfg: dict, edgar_by: dict, src: Sources, now: pd.Timestamp) -> int:
    """예측이 잠긴 행 중 아직 다 채점되지 않은 것: 발표일·유효·매출 가속 정답·QQQ 대비 초과수익(기간이 찬 것만)."""
    p = st.predictions(conn)
    p = p[p["prediction"].notna()]
    have = st.scores(conn).set_index(["ticker", "fiscal_end"])
    px: dict[str, pd.Series] = {}
    n = 0
    for r in p.itertuples():
        key = (r.ticker, r.fiscal_end)
        if key in have.index and have.loc[key, ["accelerated", "ex_1d", "ex_1m", "ex_6m"]].notna().all():
            continue
        ed = edgar_by.get(r.ticker)
        if not ed:
            continue
        end = pd.Timestamp(r.fiscal_end)
        rev = sf.quarterly_revenue(ed.get("revenue_facts") or [])
        m = rev[(rev["end"] - end).abs() <= pd.Timedelta(days=20)]
        actual = m["end"].iloc[0] if len(m) else None
        ann = sf.announcement_date(ed.get("filings") or [], actual if actual is not None else end)
        if ann is None:
            continue
        ex = {k: None for k in cfg["grading"]["windows"]}
        try:
            for t in (r.ticker, cfg["grading"]["benchmark"]):
                if t not in px:
                    px[t] = src.prices(t)
            ex = sf.excess_returns(px[r.ticker], px[cfg["grading"]["benchmark"]], ann, cfg["grading"]["windows"])
        except Exception as e:  # noqa: BLE001 — 주가 실패는 다음 주에 다시
            print(f"[search_forecast] {r.ticker} 주가 실패(다음 주 다시): {type(e).__name__}")
        acc = sf.accelerated(rev, actual) if actual is not None else None
        v1, v2 = sf.is_valid(pd.Timestamp(r.computed_at), ann), sf.is_valid(pd.Timestamp(r.pred_locked_at), ann)
        st.upsert_score(conn, {"ticker": r.ticker, "fiscal_end": r.fiscal_end, "season": r.season, "prediction": r.prediction,
                               "announced": ann.date().isoformat(), "valid": None if v1 is None else int(v1),
                               "valid_pred_lock": None if v2 is None else int(v2),
                               "actual_end": actual.date().isoformat() if actual is not None else None,
                               "accelerated": None if acc is None else int(acc), "ex_1d": ex.get("-1~+1일"),
                               "ex_1m": ex.get("1개월"), "ex_6m": ex.get("6개월"), "scored_at": _iso(now)})
        n += 1
    conn.commit()
    return n


# ── 실행 ────────────────────────────────────────────────────────────────────


def fetch_edgar(cfg: dict, src: Sources) -> tuple[dict, list[str]]:
    out, fail = {}, []
    for tk in cfg["companies"]:
        try:
            out[tk] = src.edgar(tk)
        except Exception as e:  # noqa: BLE001 — 모아서 보고
            fail.append(f"{tk}({type(e).__name__})")
    return out, fail


def run(cfg: dict, db_path, now: pd.Timestamp, src: Sources, lock_dir: Path = LOCK_DIR, do_collect: bool = True) -> dict:
    """한 번 실행: 수집 → EDGAR → 점수 잠금 → 시즌 예측 잠금 → 채점. 출력: 보고 dict."""
    conn = st.connect(db_path)
    try:
        rep = collect(conn, cfg, src, now) if do_collect else {}
        edgar_by, efail = fetch_edgar(cfg, src)
        windows_by = {tk: company_windows(e, now) for tk, e in edgar_by.items()}
        rep["edgar_fail"] = efail
        rep["scored"] = len(lock_scores(conn, cfg, windows_by, now, lock_dir))
        rep["seasons_locked"] = lock_seasons(conn, cfg, now, lock_dir)
        rep["graded"] = grade(conn, cfg, edgar_by, src, now)
    finally:
        conn.close()
    return rep


def preview(cfg: dict, db_path, now: pd.Timestamp, src: Sources) -> pd.DataFrame:
    """수집 뒤 회사별 가장 최근(분기 끝 + 10일이 지난) 분기 점수와 그 시즌 안 3등분 — 잠금·DB 예측 기록 없음."""
    conn = st.connect(db_path)
    try:
        edgar_by, _ = fetch_edgar(cfg, src)
        today = now.tz_convert("UTC").tz_localize(None).normalize()
        rows = []
        for tk, e in edgar_by.items():
            w = company_windows(e, now)
            done = [x for x in w if x[1] + pd.Timedelta(days=cfg["score_delay_days"]) <= today]
            if done:
                rows.append(score_one(conn, cfg, tk, w, done[-1][1]))
    finally:
        conn.close()
    df = pd.DataFrame(rows)
    df["예측"] = None
    for season, g in df.groupby("season"):
        lab = sf.terciles(dict(zip(g["ticker"], g["score"].astype(float))))
        df.loc[g.index, "예측"] = g["ticker"].map(lab)
    return df


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="검색량 실적 예고 주간 실행")
    ap.add_argument("--db", default=None)
    ap.add_argument("--lock-dir", default=str(LOCK_DIR))
    ap.add_argument("--now", default=None, help="ISO 시각(UTC) — 시험용")
    ap.add_argument("--preview", action="store_true", help="수집 + 최근 분기 점수 화면 출력(잠금 없음)")
    a = ap.parse_args(argv)
    cfg = load_config()
    now = pd.Timestamp(a.now, tz="UTC") if a.now else pd.Timestamp.now(tz="UTC")
    src = live_sources(cfg, now)
    if a.preview:
        conn = st.connect(a.db)
        rep = collect(conn, cfg, src, now)
        conn.close()
        print(f"[search_forecast] 검색어 성공 {rep['terms_ok']} / 실패 {len(rep['terms_fail'])} {rep['terms_fail']}")
        print(f"[search_forecast] 위키 성공 {rep['wiki_ok']} / 실패 {len(rep['wiki_fail'])} {rep['wiki_fail']}")
        df = preview(cfg, a.db, now, src)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(df.to_string(index=False))
        return
    rep = run(cfg, a.db, now, src, Path(a.lock_dir))
    print(f"[search_forecast] 검색어 성공 {rep['terms_ok']} / 실패 {len(rep['terms_fail'])} {rep['terms_fail']} · "
          f"위키 성공 {rep['wiki_ok']} / 실패 {len(rep['wiki_fail'])} · EDGAR 실패 {rep['edgar_fail'] or '없음'} · "
          f"점수 잠금 {rep['scored']}행 · 시즌 예측 잠금 {rep['seasons_locked'] or '없음'} · 채점 {rep['graded']}행")


if __name__ == "__main__":
    main()
