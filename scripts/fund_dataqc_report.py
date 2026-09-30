"""AI 펀드 F2: engine.dataqc.build_report를 실제 2015~2021 데이터에 연결해 연도별 확보율·
판정(PASS/INCONCLUSIVE)을 낸다 (docs/design/fund_sim.md 5.1, 5.7). 빠진 종목 목록(유료
데이터 업체 문의용)도 함께 만든다.

98% 기준은 바꾸지 않는다(config.yaml dataqc.min_coverage_pct 그대로 읽음) — 이 스크립트는
그 기준을 실제 데이터에 적용해서 결과를 내는 것만 한다.

**봉인 구간(2022-01-01 이후) 데이터는 절대 안 받는다**: end=cfg["backtest"]["train_end"]
(2021-12-31)로 고정하고, engine.backtest.prepare_data가 cfg의 seal_date를 항상 적용하므로
실수로 그 뒤를 요청해도 core.seal.SealedDataError로 막힌다(unseal 인자를 아예 안 씀).

실행(반드시 python -u로):
    python -u -m scripts.fund_dataqc_report

출력:
    outputs/backtest/fund_dataqc/report.md   — 연도별 확보율·판정, 빠진 종목 목록
    outputs/backtest/fund_dataqc/report.json — 같은 내용의 구조화 데이터
    outputs/backtest/fund_dataqc/missing_tickers.csv — 빠진 종목 목록(유료 데이터 문의용)
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from data import universe_history as uh  # noqa: E402
from data.market_calendar import trading_days_between  # noqa: E402
from engine import backtest as bt  # noqa: E402
from engine import dataqc as dqc  # noqa: E402

OUT_DIR = bt.OUTPUT_ROOT / "fund_dataqc"
LOG_PATH = OUT_DIR / "report.log"
REPORT_MD_PATH = OUT_DIR / "report.md"
REPORT_JSON_PATH = OUT_DIR / "report.json"
MISSING_CSV_PATH = OUT_DIR / "missing_tickers.csv"


def log(msg: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ── 위키백과 표에서 회사명·편출 사유까지 뽑는다 (data.universe_history.parse_changes는
# 티커만 남기고 버리므로, 이 스크립트 전용으로 따로 파싱한다 — 기존 모듈은 건드리지 않는다.
# fetch_changes_wikitext·_parse_date_cell·_clean_ticker_cell은 그대로 재사용) ──────────────


@dataclass(frozen=True)
class ChangeDetail:
    date: date
    added: str | None
    added_name: str | None
    removed: str | None
    removed_name: str | None
    reason: str


def _clean_text(text: str) -> str:
    """위키텍스트 각주·링크를 걷어내고 사람이 읽을 텍스트만 남긴다 (순수 함수)."""
    text = text.strip()
    text = re.sub(r"<ref[^>]*/?>.*?(</ref>|$)", "", text, flags=re.S)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", lambda m: m.group(2) or m.group(1), text)
    text = re.sub(r"'''?", "", text)
    return text.strip()


def parse_changes_with_detail(wikitext: str) -> list[ChangeDetail]:
    """parse_changes와 같은 표를 파싱하되 회사명(cells[2]/[4])·편출 사유(cells[5])도 남긴다."""
    m = re.search(r'\{\|\s*class="wikitable[^\n]*id="changes"(.*?)\n\|\}', wikitext, re.S)
    if not m:
        return []
    body = m.group(1)
    out: list[ChangeDetail] = []
    for chunk in body.split("\n|-"):
        chunk = chunk.strip("\n")
        if not chunk or chunk.lstrip().startswith("!"):
            continue
        cells = chunk.split("\n|")
        cells[0] = cells[0].lstrip("|")
        cells = [c.strip() for c in cells]
        if len(cells) < 5:
            continue
        d = uh._parse_date_cell(cells[0])
        if d is None:
            continue
        added = uh._clean_ticker_cell(cells[1])
        removed = uh._clean_ticker_cell(cells[3])
        if added is None and removed is None:
            continue
        added_name = _clean_text(cells[2]) if len(cells) > 2 and cells[2].strip() else None
        removed_name = _clean_text(cells[4]) if len(cells) > 4 and cells[4].strip() else None
        reason = _clean_text(cells[5]) if len(cells) > 5 and cells[5].strip() else ""
        out.append(ChangeDetail(date=d, added=added, added_name=added_name, removed=removed, removed_name=removed_name, reason=reason))
    return out


def ticker_membership_intervals(checkpoints: list[tuple[date, frozenset]], ticker: str) -> list[tuple[date, date | None]]:
    """checkpoints에서 ticker가 연속으로 소속돼 있던 구간들을 찾는다 (순수 함수).

    출력: [(시작일, 종료일 또는 None(마지막 체크포인트까지 계속 소속))]
    """
    intervals: list[tuple[date, date | None]] = []
    start: date | None = None
    for d, members in checkpoints:
        present = ticker in members
        if present and start is None:
            start = d
        elif not present and start is not None:
            intervals.append((start, d))
            start = None
    if start is not None:
        intervals.append((start, None))
    return intervals


def overlap(a_start: date, a_end: date | None, b_start: date, b_end: date) -> tuple[date, date] | None:
    """[a_start, a_end](a_end=None이면 무한대)와 [b_start, b_end]의 겹치는 구간."""
    lo = max(a_start, b_start)
    hi = b_end if a_end is None else min(a_end, b_end)
    if lo > hi:
        return None
    return (lo, hi)


def build_missing_ticker_rows(
    failed_tickers: dict[str, str],
    full_checkpoints: list[tuple[date, frozenset]],
    changes: list,
    study_start: date,
    study_end: date,
) -> list[dict]:
    """빠진 종목 목록(유료 데이터 업체 문의용)을 만든다.

    입력: failed_tickers(prepare_data 결과 — ticker: 실패 사유 문자열), full_checkpoints
         (2000-01-01부터 시작한 넓은 범위의 membership_checkpoints — 2014년 이전 소속 기간도
         잡기 위함), changes(parse_changes_with_detail 결과), study_start/end(우리 백테스트가
         실제로 필요로 하는 구간 — config.yaml의 warmup_start~train_end)
    출력: 티커별 행 목록(회사명, 소속 기간, 편출 사유, 필요한 가격 기간)
    """
    by_removed: dict[tuple[str, date], ChangeDetail] = {}
    name_by_ticker: dict[str, str] = {}
    for c in changes:
        if c.removed:
            by_removed[(c.removed, c.date)] = c
            if c.removed_name:
                name_by_ticker.setdefault(c.removed, c.removed_name)
        if c.added and c.added_name:
            name_by_ticker.setdefault(c.added, c.added_name)

    rows = []
    for ticker in sorted(failed_tickers):
        intervals = ticker_membership_intervals(full_checkpoints, ticker)
        needed_ranges = []
        for s, e in intervals:
            ov = overlap(s, e, study_start, study_end)
            if ov:
                needed_ranges.append(ov)

        reasons = []
        for s, e in intervals:
            if e is None:
                continue
            detail = by_removed.get((ticker, e))
            if detail and detail.reason:
                reasons.append(f"{e.isoformat()}: {detail.reason}")
        reason_text = " / ".join(reasons) if reasons else "확인 필요(위키백과 표에 사유 텍스트 없음)"

        membership_text = "; ".join(
            f"{s.isoformat()}~{e.isoformat() if e else '현재'}" for s, e in intervals
        ) if intervals else "확인 필요(변경 이력에서 소속 구간을 못 찾음 — 티커 표기가 다를 수 있음)"

        needed_text = "; ".join(f"{s.isoformat()}~{e.isoformat()}" for s, e in needed_ranges) if needed_ranges else "겹치는 구간 없음(연구 구간 밖)"

        rows.append(
            {
                "ticker": ticker,
                "company_name": name_by_ticker.get(ticker, "확인 필요"),
                "membership_periods": membership_text,
                "removal_reason": reason_text,
                "needed_price_range": needed_text,
                "fetch_error": failed_tickers[ticker],
            }
        )
    return rows


ALT_SOURCE_DIR = ROOT / "data" / "cache" / "alt_prices" / "tiingo"


def load_alt_source_indicator_map(alt_dir: Path = ALT_SOURCE_DIR) -> dict[str, "pd.DataFrame"]:
    """scripts/fund_fetch_missing_prices.py가 저장한 대체 출처(Tiingo) 가격을 읽어
    dataqc가 바로 쓸 수 있는 {ticker: DataFrame(close 열 포함)} 형태로 돌려준다.

    입력: alt_dir(비어 있거나 없으면 빈 dict — 대체 출처를 안 받았어도 이 스크립트가
         그대로 동작해야 한다)
    출력: {ticker: DataFrame(날짜 인덱스, "close" 열 — split-only 조정됨)}
    """
    import pandas as pd

    if not alt_dir.exists():
        return {}
    out: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(alt_dir.glob("*.csv")):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if "close" in df.columns and not df.empty:
            out[csv_path.stem] = df
    return out


def main() -> None:
    cfg = bt.load_config()
    bt_cfg = cfg["backtest"]
    warmup_start = bt._parse_date(bt_cfg["warmup_start"])  # 2014-01-01
    train_start = bt._parse_date(bt_cfg["train_start"])  # 2015-01-01
    train_end = bt._parse_date(bt_cfg["train_end"])  # 2021-12-31, 봉인 이전 — 여기서 절대 안 넘어간다
    min_coverage_pct = cfg["dataqc"]["min_coverage_pct"]

    log(f"=== F2 dataqc 실제 데이터 연결 시작 ({train_start}~{train_end}, 절대 {bt_cfg.get('seal_date')} 이후는 안 받음) ===")

    t0 = time.time()
    data = bt.prepare_data(cfg, warmup_start, train_end)  # force_universe_mode=None(기본, POINT_IN_TIME), unseal 안 씀
    log(f"prepare_data 완료: universe_mode={data.universe_mode} · 종목 {len(data.indicator_map)}개(실패 {len(data.failed_tickers)}) · {time.time()-t0:.0f}초")

    # 대체 출처(Tiingo, scripts/fund_fetch_missing_prices.py) 병합 — 있으면 반영, 없으면
    # 이전과 완전히 같게 동작한다(2026-10-01 사용자 지시 4번).
    alt_prices = load_alt_source_indicator_map()
    indicator_map = dict(data.indicator_map)
    added_from_alt = [t for t in alt_prices if t not in indicator_map]
    indicator_map.update(alt_prices)
    if alt_prices:
        log(f"대체 출처(Tiingo) 병합: {len(added_from_alt)}개 종목 추가({sorted(added_from_alt)}) — data/cache/alt_prices/tiingo/")
    else:
        log("대체 출처 없음 — yfinance 결과만 사용(이전과 동일)")

    # ── 연도별 확보율·판정 ──────────────────────────────────────────────────
    years = list(range(train_start.year, train_end.year + 1))
    year_reports = {}
    for y in years:
        y_start = max(date(y, 1, 1), train_start)
        y_end = min(date(y, 12, 31), train_end)
        y_days = [d.date() for d in trading_days_between(y_start, y_end)]
        report = dqc.build_report(data.checkpoints, y_days, indicator_map, cfg)
        judgement = "PASS" if not report.inconclusive else "INCONCLUSIVE"
        year_reports[y] = {
            "coverage_pct": report.coverage_pct,
            "ticker_days_expected": report.ticker_days_expected,
            "ticker_days_priced": report.ticker_days_priced,
            "judgement": judgement,
            "reasons": report.reasons,
        }
        log(f"{y}: 확보율 {report.coverage_pct}% ({report.ticker_days_priced}/{report.ticker_days_expected}) -> {judgement}")

    all_days = [d.date() for d in trading_days_between(train_start, train_end)]
    overall_report = dqc.build_report(data.checkpoints, all_days, indicator_map, cfg)
    overall_judgement = "PASS" if not overall_report.inconclusive else "INCONCLUSIVE"
    log(f"전체({train_start}~{train_end}): 확보율 {overall_report.coverage_pct}% -> {overall_judgement}")
    log(f"구성종목 수 범위 이상 체크포인트: {len(overall_report.constituent_count_issues)}건")
    log(f"이상치·분할 누락 의심(±{cfg['dataqc']['extreme_daily_move_pct']}%): {len(overall_report.extreme_move_issues)}건")

    # ── 빠진 종목 목록 ──────────────────────────────────────────────────────
    log("빠진 종목의 소속 기간·편출 사유를 위키백과 변경 이력에서 찾는 중...")
    wikitext_changes = parse_changes_with_detail_from_cache()
    full_checkpoints = uh.membership_checkpoints(
        set(_get_current_tickers()), uh.get_changes(), date(2000, 1, 1)
    )
    missing_rows = build_missing_ticker_rows(
        data.failed_tickers, full_checkpoints, wikitext_changes, warmup_start, train_end
    )
    for row in missing_rows:
        row["recovered_via_alt_source"] = row["ticker"] in alt_prices
    recovered_count = sum(1 for r in missing_rows if r["recovered_via_alt_source"])
    log(f"빠진 종목 {len(missing_rows)}개(yfinance 기준) — 그중 대체 출처로 복구됨: {recovered_count}개 — outputs/backtest/fund_dataqc/missing_tickers.csv에 저장")

    missing_by_year: dict[int, int] = {y: 0 for y in years}
    for row in missing_rows:
        for part in row["needed_price_range"].split("; "):
            if "~" not in part:
                continue
            s_str, e_str = part.split("~")
            try:
                s_d, e_d = date.fromisoformat(s_str), date.fromisoformat(e_str)
            except ValueError:
                continue
            for y in years:
                y_start, y_end = date(y, 1, 1), date(y, 12, 31)
                if s_d <= y_end and e_d >= y_start:
                    missing_by_year[y] += 1

    # 연구 구간(2014~2021)과 전혀 안 겹치는 티커(2022년 이후에 나스닥100에 새로 들어온 종목이
    # 어쩌다 같이 실패한 것뿐 — 이번 F2 학습 구간에는 필요 없다)를 실제로 필요한 것과 구분한다.
    out_of_scope_tickers = [r["ticker"] for r in missing_rows if r["needed_price_range"] == "겹치는 구간 없음(연구 구간 밖)"]
    in_scope_rows = [r for r in missing_rows if r["needed_price_range"] != "겹치는 구간 없음(연구 구간 밖)"]

    # ── 저장 ────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "period": {"start": train_start.isoformat(), "end": train_end.isoformat()},
        "min_coverage_pct_threshold": min_coverage_pct,
        "year_reports": {str(y): r for y, r in year_reports.items()},
        "overall": {
            "coverage_pct": overall_report.coverage_pct,
            "judgement": overall_judgement,
            "constituent_count_issue_count": len(overall_report.constituent_count_issues),
            "extreme_move_issue_count": len(overall_report.extreme_move_issues),
        },
        "missing_ticker_count": len(missing_rows),
        "missing_ticker_count_in_scope": len(in_scope_rows),
        "missing_ticker_count_out_of_scope": len(out_of_scope_tickers),
        "out_of_scope_tickers": out_of_scope_tickers,
        "missing_ticker_count_by_year": missing_by_year,
        "recovered_via_alt_source_count": recovered_count,
        "recovered_via_alt_source_tickers": sorted(t for t in alt_prices if t in {r["ticker"] for r in missing_rows}),
        "note": "98% 기준(dataqc.min_coverage_pct)은 설계 문서 그대로 유지 — 이 보고서는 그 기준을 실제 데이터에 적용한 결과일 뿐이다.",
    }
    REPORT_JSON_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(MISSING_CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ticker", "company_name", "membership_periods", "removal_reason", "needed_price_range", "fetch_error", "recovered_via_alt_source"]
        )
        writer.writeheader()
        writer.writerows(missing_rows)

    md_lines = [
        "# AI 펀드 F2 — 데이터 품질 검사(dataqc) 실제 데이터 결과",
        "",
        f"- 실행: {summary['generated_at']}",
        f"- 구간: {train_start} ~ {train_end} (봉인 구간 2022-01-01 이후는 요청하지 않음)",
        f"- 판정 기준: 구성종목 가격 확보율 >= {min_coverage_pct}% (config.yaml dataqc.min_coverage_pct, 안 바꿈)",
        "",
        "## 연도별 확보율·판정",
        "",
        "| 연도 | 확보율 | 종목·일수(받음/기대) | 판정 |",
        "| --- | --- | --- | --- |",
    ]
    for y in years:
        r = year_reports[y]
        md_lines.append(f"| {y} | {r['coverage_pct']}% | {r['ticker_days_priced']}/{r['ticker_days_expected']} | **{r['judgement']}** |")
    md_lines += [
        f"| 전체({train_start.year}~{train_end.year}) | {overall_report.coverage_pct}% | {overall_report.ticker_days_priced}/{overall_report.ticker_days_expected} | **{overall_judgement}** |",
        "",
        f"- 구성종목 수 범위(config.yaml dataqc.expected_constituent_count) 이상 체크포인트: {len(overall_report.constituent_count_issues)}건",
        f"- 하루 등락률 ±{cfg['dataqc']['extreme_daily_move_pct']}% 이상(이상치·분할 누락 의심): {len(overall_report.extreme_move_issues)}건",
        "",
        "## 빠진 종목 (yfinance 기준)",
        "",
        f"- yfinance로 시세를 못 받은 종목 총 {len(missing_rows)}개",
        f"  - 연구 구간(2014~2021)과 겹쳐 **실제로 필요한 종목: {len(in_scope_rows)}개**",
        f"  - 연구 구간과 안 겹치는 종목(2022년 이후 나스닥100 신규 편입, 이번 F2 학습 구간에는 불필요): {len(out_of_scope_tickers)}개 — " + ", ".join(out_of_scope_tickers),
        f"- 연도별 필요 구간 겹침(실제로 필요한 {len(in_scope_rows)}개 기준): " + ", ".join(f"{y} {n}개" for y, n in missing_by_year.items()),
        f"- **대체 출처(Tiingo)로 복구됨: {recovered_count}개**(scripts/fund_fetch_missing_prices.py, 위 연도별 확보율에 이미 반영됨) — {', '.join(sorted(t for t in alt_prices if t in {r['ticker'] for r in missing_rows})) or '없음'}",
        f"- 전체 목록: `{MISSING_CSV_PATH.relative_to(ROOT)}` (티커·회사명·소속기간·편출사유·필요 가격기간·대체출처 복구 여부)",
        "",
        "## 결론",
        "",
    ]
    passing_years = [y for y in years if year_reports[y]["judgement"] == "PASS"]
    if not overall_report.inconclusive:
        md_lines.append("확보율 기준을 만족했다.")
    elif passing_years:
        md_lines.append(
            f"**전체 구간은 데이터 품질 미달 → INCONCLUSIVE(참고용).** 다만 연도별로는 {', '.join(str(y) for y in passing_years)}년이 "
            f"98% 기준을 PASS했다 — 나머지 해({', '.join(str(y) for y in years if y not in passing_years)})가 기준 미달이라 전체 평균은 INCONCLUSIVE."
        )
    else:
        md_lines.append("**데이터 품질 미달 → INCONCLUSIVE(참고용).** 98% 기준을 만족한 해가 학습 구간에 하나도 없다.")
    REPORT_MD_PATH.write_text("\n".join(md_lines), encoding="utf-8")
    log(f"=== 완료: {REPORT_MD_PATH}, {REPORT_JSON_PATH}, {MISSING_CSV_PATH} ===")


def _get_current_tickers() -> list[str]:
    from data.universe import get_universe

    return list(get_universe()["ticker"])


def parse_changes_with_detail_from_cache() -> list[ChangeDetail]:
    """캐시된(또는 새로 받은) 위키텍스트를 회사명·사유까지 포함해 파싱한다."""
    cache_path = uh.CACHE_PATH.parent / "nasdaq100_changes_wikitext.txt"
    if cache_path.exists():
        wikitext = cache_path.read_text(encoding="utf-8")
    else:
        wikitext = uh.fetch_changes_wikitext()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(wikitext, encoding="utf-8")
    return parse_changes_with_detail(wikitext)


if __name__ == "__main__":
    main()
