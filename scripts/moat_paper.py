"""해자 실시간 paper 검증 — 매월 수동 실행용 (docs/design/moat_paper.md, moat-paper-v1-locked).

처음 실행: 시작 보유 목록(P1·P2 종목·비중·진입가·등급·유통주식수 출처)을
docs/paper/moat_paper_start.json으로 저장한다. 체결가(2026-10-01 시가)가 아직 없으면
그 종목은 "진입 대기"로 남기고, 이후 실행 때 값이 생기면 채운다(core.moat_paper.
merge_price_fills — 이미 채워진 값은 절대 다시 안 바꾼다. 시작 파일이 완전히 "체결 완료"
상태가 되면 그 다음부터는 이 스크립트가 시작 파일 자체를 다시는 건드리지 않는다).

이후 실행(또는 체결 완료 뒤 매월 실행): 시작 파일을 읽어 현재 성과를
outputs/moat_paper_YYYY-MM.md로 출력하고 store/moat_paper.db에 월말 값을 기록한다.
보유 종목의 해자 등급은 매달 다시 계산해 기록만 한다(교체일 전까지 보유·비중은 안 바꿈).

실행(.env에 SEC_USER_AGENT 필요):
    python -u -m scripts.moat_paper
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from core import moat  # noqa: E402
from core import moat_backtest as mbt  # noqa: E402
from core import moat_paper as mpaper  # noqa: E402
from data import edgar  # noqa: E402
from data.prices import fetch_universe_prices  # noqa: E402
from data.universe_history import point_in_time_universe  # noqa: E402
from scripts.moat_h4_backtest import get_locked_hash  # noqa: E402
from scripts.moat_report import (  # noqa: E402
    dedupe_rows_by_cik,
    fetch_sector_from_yfinance,
    get_sector,
    load_sector_cache,
    resolve_cik,
    save_sector_cache,
    sector_breakdown,
)
from store import moat_paper as store_mp  # noqa: E402

PAPER_DIR = ROOT / "docs" / "paper"
START_PATH = PAPER_DIR / "moat_paper_start.json"
RULES_DOC_PATH = ROOT / "docs" / "design" / "moat_paper.md"
OUT_DIR = ROOT / "outputs"
LOG_PATH = ROOT / "outputs" / "moat" / "paper_report.log"

JUDGMENT_DATE = date(2026, 9, 30)  # moat_paper.md "판단 기준일" — 고정값, date.today()를 쓰지 않는다
ENTRY_DATE = date(2026, 10, 1)  # moat_paper.md "진입" — 고정값(다음 거래일)

# [고정값] moat_paper.md P2 "종목 최대 10%" — config.yaml moat: 섹션(잠금 대상)은 건드리지
# 않는다는 작업 지시에 따라 여기 상수로만 둔다.
P2_STOCK_CAP_PCT = 10.0


def log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def rules_doc_hash() -> str:
    """docs/design/moat_paper.md(잠긴 규칙 문서) 해시 — 시작 파일에 남겨 어떤 규칙 버전으로
    만들었는지 추적한다 (core.moat.compute_config_hash와 같은 16자리 16진 방식)."""
    return hashlib.sha256(RULES_DOC_PATH.read_bytes()).hexdigest()[:16]


def fetch_shares_outstanding_yfinance(ticker: str) -> float | None:
    """yfinance 현재 유통주식수(우선순위 3번, moat_paper.md P2 — dei·분기 희석주식수가
    둘 다 없을 때만 쓴다). 실패하면 None (네트워크)."""
    import yfinance as yf

    try:
        info = yf.Ticker(ticker).info
        val = info.get("sharesOutstanding")
        return float(val) if val else None
    except Exception:
        return None


# ── 1. 유니버스·등급·유통주식수 (기준일, 2026-09-30) ────────────────────────────


def build_universe_and_grades(cfg: dict, user_agent: str) -> dict:
    universe = sorted(point_in_time_universe(JUDGMENT_DATE))
    log(f"기준일({JUDGMENT_DATE}) 나스닥100 구성종목 {len(universe)}개")

    ticker_to_cik = edgar.fetch_ticker_to_cik(user_agent=user_agent)
    sector_cache = load_sector_cache()

    rows = []
    raw_shares: dict[str, float | None] = {}
    shares_source: dict[str, str | None] = {}
    for ticker in universe:
        sector = get_sector(ticker, sector_cache, fetch_sector_from_yfinance)
        cik = resolve_cik(ticker, ticker_to_cik)
        if cik is None:
            rows.append({"ticker": ticker, "cik": None, "grade": "판단 불가", "sector": sector})
            raw_shares[ticker] = None
            shares_source[ticker] = None
            continue
        try:
            facts = edgar.fetch_company_facts(cik, user_agent=user_agent)
        except edgar.EdgarFetchError as exc:
            log(f"  {ticker}: companyfacts 요청 실패 -> 판단 불가 ({exc})")
            rows.append({"ticker": ticker, "cik": cik, "grade": "판단 불가", "sector": sector})
            raw_shares[ticker] = None
            shares_source[ticker] = None
            continue

        profile = moat.analyze_company(facts, ticker, cfg, JUDGMENT_DATE)
        rows.append({"ticker": ticker, "cik": cik, "grade": profile.grade, "sector": sector})

        dei_val = mpaper.latest_cover_page_shares(facts, JUDGMENT_DATE)
        diluted_val = None if dei_val else mpaper.latest_quarterly_diluted_shares(facts, JUDGMENT_DATE)
        yf_val = fetch_shares_outstanding_yfinance(ticker) if (dei_val is None and diluted_val is None) else None
        shares, source = mpaper.shares_outstanding_from_candidates(dei_val, diluted_val, yf_val)
        raw_shares[ticker] = shares
        shares_source[ticker] = source

    save_sector_cache(sector_cache)

    ticker_cik_map = {r["ticker"]: r["cik"] for r in rows}
    aggregated_shares = mpaper.aggregate_shares_by_cik(raw_shares, ticker_cik_map)

    deduped = dedupe_rows_by_cik(rows)
    wide_tickers = sorted(r["ticker"] for r in deduped if r["grade"] == "넓음")
    log(f"넓음 등급(회사 기준 중복 제거) {len(wide_tickers)}개: {wide_tickers}")

    return {
        "universe": universe,
        "sectors": {r["ticker"]: r["sector"] for r in rows},
        "grades": {r["ticker"]: r["grade"] for r in rows},
        "cik": ticker_cik_map,
        "shares_outstanding": {t: {"value": aggregated_shares.get(t), "source": shares_source.get(t)} for t in universe},
        "wide_tickers": wide_tickers,
    }


# ── 2. 체결가(진입가) 채우기 ────────────────────────────────────────────────────


def fetch_entry_opens(tickers: list[str], cfg: dict) -> tuple[dict[str, float], list[str]]:
    """2026-10-01 확정 시가를 구한다 (data.prices는 당일 미확정 봉을 자동으로 버리므로,
    아직 장이 안 끝났으면 자연히 못 구한다 — 그게 "진입 대기"다).

    출력: ({티커: 시가(슬리피지 반영 전)}, 못 구한 티커 목록)
    """
    price_result = fetch_universe_prices(tickers, cfg)
    entry_ts = pd.Timestamp(ENTRY_DATE)
    opens: dict[str, float] = {}
    missing: list[str] = []
    for t in tickers:
        df = price_result.prices.get(t)
        if df is not None and entry_ts in df.index and pd.notna(df.loc[entry_ts, "open"]):
            opens[t] = float(df.loc[entry_ts, "open"])
        else:
            missing.append(t)
    return opens, missing


def build_entries(tickers: list[str], raw_opens: dict[str, float], slippage_pct: float) -> dict[str, dict]:
    entries = {}
    for t in tickers:
        if t in raw_opens:
            mpaper.enforce_paper_floor(ENTRY_DATE)  # 안전장치 재확인(설계상 항상 통과)
            entry_price = mpaper.compute_entry_price(raw_opens[t], slippage_pct)
            entries[t] = {"status": "체결", "entry_open": raw_opens[t], "entry_price": entry_price}
        else:
            entries[t] = {"status": "진입 대기"}
    return entries


def finalize_if_ready(start: dict, cfg: dict) -> None:
    """P1·P2·QQQM·QQEW 체결가가 모두 모이면 비중을 계산해 얼린다(그 뒤로는 절대 안 바꿈)."""
    needed = list(start["wide_tickers"]) + ["QQQM", "QQEW"]
    if not all(start["entries"].get(t, {}).get("status") == "체결" for t in needed):
        start["status"] = "진입 대기"
        return

    sector_cap = cfg["moat_backtest"]["sector_cap_pct"]
    sectors = start["sectors"]
    wide = start["wide_tickers"]
    entry_prices = {t: start["entries"][t]["entry_price"] for t in wide}

    p1_weights = mbt.allocate_equal_weight_with_sector_cap(wide, sectors, sector_cap)

    market_caps = {}
    for t in wide:
        shares = start["shares_outstanding"].get(t, {}).get("value")
        market_caps[t] = (shares * entry_prices[t]) if shares else 0.0
    p2_weights = mpaper.market_cap_weights_with_caps(wide, market_caps, sectors, sector_cap, P2_STOCK_CAP_PCT)

    start["p1_weights"] = p1_weights
    start["p2_weights"] = p2_weights
    start["p2_market_caps"] = market_caps
    start["status"] = "체결 완료"
    start["entered_at"] = datetime.now().isoformat(timespec="seconds")
    log("모든 보유 종목·QQQM·QQEW 체결 완료 — P1·P2 비중을 계산해 고정한다")


def build_skeleton(cfg: dict, user_agent: str) -> dict:
    universe_data = build_universe_and_grades(cfg, user_agent)
    tickers_needed = sorted(set(universe_data["universe"]) | {"QQQM", "QQEW"})
    raw_opens, missing = fetch_entry_opens(tickers_needed, cfg)
    slippage_pct = cfg["backtest"]["costs"]["slippage_pct"]
    entries = build_entries(tickers_needed, raw_opens, slippage_pct)

    start = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rules_doc_hash": rules_doc_hash(),
        "judgment_date": JUDGMENT_DATE.isoformat(),
        "entry_date": ENTRY_DATE.isoformat(),
        "universe": universe_data["universe"],
        "sectors": universe_data["sectors"],
        "grades": universe_data["grades"],
        "cik": universe_data["cik"],
        "shares_outstanding": universe_data["shares_outstanding"],
        "wide_tickers": universe_data["wide_tickers"],
        "random": {
            "pool": universe_data["universe"], "n": len(universe_data["wide_tickers"]),
            "seed": cfg["moat_backtest"]["random_seed"], "trials": cfg["moat_backtest"]["random_trials"],
        },
        "entries": entries,
        "p1_weights": None, "p2_weights": None, "p2_market_caps": None,
        "status": "진입 대기",
    }
    log(f"시가를 아직 못 구한 티커 {len(missing)}개(진입 대기): {missing[:20]}{'...' if len(missing) > 20 else ''}")
    finalize_if_ready(start, cfg)
    return start


def try_fill_pending(start: dict, cfg: dict) -> dict:
    pending = [t for t, e in start["entries"].items() if e.get("status") == "진입 대기"]
    if not pending:
        finalize_if_ready(start, cfg)
        return start
    raw_opens, _missing = fetch_entry_opens(pending, cfg)
    slippage_pct = cfg["backtest"]["costs"]["slippage_pct"]
    new_fills = build_entries(pending, raw_opens, slippage_pct)
    start["entries"] = mpaper.merge_price_fills(start["entries"], new_fills)
    finalize_if_ready(start, cfg)
    return start


# ── 3. 월간 보고 ────────────────────────────────────────────────────────────────


def _weights_table(weights: dict[str, float], top_n: int = 10) -> str:
    ranked = sorted(weights.items(), key=lambda kv: -kv[1])[:top_n]
    return "\n".join(f"| {t} | {w * 100:.2f}% |" for t, w in ranked)


def render_pending_report(start: dict, period: str, current_grades: dict[str, str]) -> str:
    pending = [t for t, e in start["entries"].items() if e.get("status") == "진입 대기"]
    wide = start["wide_tickers"]
    sector_rows = [{"sector": start["sectors"].get(t)} for t in wide]
    sectors_count = sector_breakdown(sector_rows)
    lines = [
        f"# 해자 paper {period} — 진입 대기",
        "",
        f"판단 기준일: {start['judgment_date']} · 진입(예정)일: {start['entry_date']}",
        f"P1·P2 대상(넓음) {len(wide)}개 종목 · 아직 체결가를 못 구한 티커 {len(pending)}개",
        "",
        "## 아직 체결 대기 중인 티커",
        ", ".join(sorted(pending)[:50]) + (" ..." if len(pending) > 50 else "") or "없음",
        "",
        "## 해자 등급 분포 (P1·P2 대상, 이번 달 재계산)",
    ]
    for t in sorted(wide):
        lines.append(f"- {t}: {current_grades.get(t, '?')} (기준일 등급 {start['grades'].get(t)})")
    lines += ["", "## 업종 분포 (넓음 종목 수 기준)"]
    for s, n in sectors_count.items():
        lines.append(f"- {s}: {n}개")
    lines += ["", "참고: 체결가가 전부 모이면 다음 실행 때 자동으로 P1·P2 비중이 확정됩니다."]
    return "\n".join(lines)


def render_evaluated_report(
    start: dict, period: str, current_grades: dict[str, str],
    p1_factor: float | None, p2_factor: float | None, qqqm_factor: float | None, qqew_factor: float | None,
    random_factors: list[float], p1_percentile: float | None,
) -> str:
    wide = start["wide_tickers"]
    sector_rows = [{"sector": start["sectors"].get(t)} for t in wide]
    sectors_count = sector_breakdown(sector_rows)
    lines = [
        f"# 해자 paper {period}",
        "",
        f"진입일: {start['entry_date']} · 체결 완료: {start.get('entered_at', '?')}",
        "",
        "## 평가 배수 (진입 시점=1.00, 세전·마크투마켓 — 교체 전까지 매도가 없어 세금 없음)",
        f"- P1(넓음·동일비중): {p1_factor:.4f}" if p1_factor is not None else "- P1: 평가 불가",
        f"- P2(넓음·시가총액비중): {p2_factor:.4f}" if p2_factor is not None else "- P2: 평가 불가",
        f"- QQQM(대조군): {qqqm_factor:.4f}" if qqqm_factor is not None else "- QQQM: 평가 불가",
        f"- QQEW(참고): {qqew_factor:.4f}" if qqew_factor is not None else "- QQEW: 평가 불가",
        f"- 무작위 {len(random_factors)}개 중앙값: {sorted(random_factors)[len(random_factors)//2]:.4f}" if random_factors else "- 무작위 포트폴리오: 평가 불가",
        f"- P1의 무작위 대비 백분위: {p1_percentile}" if p1_percentile is not None else "- P1 백분위: 계산 불가",
        "", "## P1 상위 10개 비중",
        "| 종목 | 비중 |", "|---|---|",
        _weights_table(start["p1_weights"] or {}),
        "", "## P2 상위 10개 비중",
        "| 종목 | 비중 |", "|---|---|",
        _weights_table(start["p2_weights"] or {}),
        "", "## 업종 분포 (넓음 종목 수 기준)",
    ]
    for s, n in sectors_count.items():
        lines.append(f"- {s}: {n}개")
    lines += ["", "## 해자 등급 재계산 (이번 달)"]
    downgraded = [t for t in wide if current_grades.get(t) != start["grades"].get(t)]
    lines.append(f"기준일 대비 등급이 바뀐 종목 {len(downgraded)}개: {', '.join(downgraded) if downgraded else '없음'}")
    return "\n".join(lines)


def write_monthly_report_and_record(start: dict, cfg: dict, period: str, user_agent: str) -> None:
    recorded_at = datetime.now().isoformat(timespec="seconds")
    conn = store_mp.connect()
    try:
        current_grades: dict[str, str] = {}
        today = date.today()
        for t in start["wide_tickers"]:
            cik = start["cik"].get(t)
            grade = "판단 불가"
            if cik is not None:
                try:
                    facts = edgar.fetch_company_facts(cik, user_agent=user_agent)
                    grade = moat.analyze_company(facts, t, cfg, today).grade
                except edgar.EdgarFetchError:
                    pass
            current_grades[t] = grade
            store_mp.record_grade(conn, recorded_at, period, t, grade)

        if start["status"] != "체결 완료":
            store_mp.record_monthly(conn, recorded_at, period, "P1", None, "진입 대기", notes="진입가 미확정")
            store_mp.record_monthly(conn, recorded_at, period, "P2", None, "진입 대기", notes="진입가 미확정")
            report = render_pending_report(start, period, current_grades)
        else:
            tickers_needed = sorted(set(start["universe"]) | {"QQQM", "QQEW"})
            price_result = fetch_universe_prices(tickers_needed, cfg)
            current_prices = {t: float(df["close"].iloc[-1]) for t, df in price_result.prices.items() if len(df)}

            entry_prices_wide = {t: start["entries"][t]["entry_price"] for t in start["wide_tickers"]}
            p1_factor = mpaper.mark_to_market_factor(start["p1_weights"], entry_prices_wide, current_prices)
            p2_factor = mpaper.mark_to_market_factor(start["p2_weights"], entry_prices_wide, current_prices)
            qqqm_factor = mpaper.mark_to_market_factor({"QQQM": 1.0}, {"QQQM": start["entries"]["QQQM"]["entry_price"]}, current_prices)
            qqew_factor = mpaper.mark_to_market_factor({"QQEW": 1.0}, {"QQEW": start["entries"]["QQEW"]["entry_price"]}, current_prices)

            rnd = start["random"]
            universe_entry_prices = {
                t: start["entries"][t]["entry_price"] for t in start["universe"] if start["entries"].get(t, {}).get("status") == "체결"
            }
            random_factors: list[float] = []
            for i in range(rnd["trials"]):
                rng = random.Random(f"{rnd['seed']}-{i}")
                picked = mbt.draw_random_portfolio(rnd["pool"], rnd["n"], rng)
                w = mbt.allocate_equal_weight_with_sector_cap(picked, start["sectors"], cfg["moat_backtest"]["sector_cap_pct"])
                f = mpaper.mark_to_market_factor(w, universe_entry_prices, current_prices)
                if f is not None:
                    random_factors.append(f)
            p1_percentile = mbt.percentile_rank(p1_factor, random_factors) if p1_factor is not None and random_factors else None

            store_mp.record_monthly(conn, recorded_at, period, "P1", p1_factor, "평가")
            store_mp.record_monthly(conn, recorded_at, period, "P2", p2_factor, "평가")
            store_mp.record_monthly(conn, recorded_at, period, "QQQM", qqqm_factor, "평가")
            store_mp.record_monthly(conn, recorded_at, period, "QQEW", qqew_factor, "평가")
            median = sorted(random_factors)[len(random_factors) // 2] if random_factors else None
            store_mp.record_monthly(conn, recorded_at, period, "RANDOM_MEDIAN", median, "평가", notes=f"n={len(random_factors)}, p1_percentile={p1_percentile}")

            report = render_evaluated_report(start, period, current_grades, p1_factor, p2_factor, qqqm_factor, qqew_factor, random_factors, p1_percentile)
    finally:
        conn.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"moat_paper_{period}.md"
    out_path.write_text(report, encoding="utf-8")
    log(f"월간 보고서 저장: {out_path}")


def main() -> None:
    cfg = load_config()
    locked_hash = get_locked_hash()
    warning = moat.check_lock(cfg["moat"], locked_hash)
    if warning:
        log(warning)
        raise SystemExit("moat 설정이 잠긴 기준과 다릅니다 — paper를 멈춥니다.")
    log(f"check_lock 통과 — moat 설정이 잠긴 기준(해시 {locked_hash})과 일치")

    user_agent = edgar.get_user_agent()
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    if START_PATH.exists():
        start = json.loads(START_PATH.read_text(encoding="utf-8"))
        current_hash = rules_doc_hash()
        if start.get("rules_doc_hash") != current_hash:
            log(f"⚠️ moat_paper.md 해시가 시작 파일과 다릅니다(시작 {start.get('rules_doc_hash')}, 현재 {current_hash}) — 규칙 문서가 잠금 뒤 바뀐 것으로 보입니다.")
        if start.get("status") != "체결 완료":
            start = try_fill_pending(start, cfg)
            START_PATH.write_text(json.dumps(start, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            log(f"시작 파일 갱신(진입 대기 칸만 채움): status={start['status']}")
        else:
            log("시작 파일이 이미 체결 완료 상태 — 보유 종목·비중은 그대로 둔다(덮어쓰지 않음)")
    else:
        start = build_skeleton(cfg, user_agent)
        START_PATH.write_text(json.dumps(start, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        log(f"시작 파일 생성: {START_PATH} (status={start['status']})")

    period = date.today().strftime("%Y-%m")
    write_monthly_report_and_record(start, cfg, period, user_agent)


if __name__ == "__main__":
    main()
