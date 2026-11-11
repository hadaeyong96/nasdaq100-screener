"""샘플 데이터로 개인용·공개용 보고서 미리보기를 만든다 (네트워크 없음).

사용법:
    python scripts/report_preview.py                  # outputs/report_preview.html
    python scripts/report_preview.py --screenshot after  # + 폰 폭(375)·PC 폭(1200) 스크린샷
    python scripts/report_preview.py --pead              # 실적급등 추천 편지 개인판·단체방판(가짜 신호 2·관찰 3·끝남 2·비교 그래프)

샘플: 오늘의 추천 4건(긴 조건 문구 포함) + 시장 온도 6칸. 실제 계좌·보유 정보는 없다.
단체방도 같은 보고서·첨부 설명을 받는다(2026-10-15) — 첨부 설명은 outputs/caption_preview.txt로 남긴다.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import macro_status  # noqa: E402
from notify import briefing, report_html  # noqa: E402

OUT = ROOT / "outputs"


def _macro_row(th, slug, name, code, fcode, value, unit, series, note=None, value_before=None, **extra):
    """engine.daily._build_macro_rows와 같은 모양의 시장 온도 행 하나."""
    row = {
        "slug": slug, "name": name, "code": code, "value": value, "unit": unit, "as_of": "2026-09-30",
        "change_1w": -0.08, "series": series, "is_stale": False, "short_range": False, "note": note,
        "badge": macro_status.classify_macro(fcode, value, th, value_before=value_before), "ref_values": [],
    }
    row.update(extra)
    return row


def _buy_row(ticker, kr, limit, stop, stop_pct, condition, score):
    """engine.daily._build_buy_row + 사이징 결과와 같은 모양의 1차 정찰(b1) 행 하나."""
    return {
        "ticker": ticker, "kr": kr, "stage": "A1", "bucket": "b1", "stage_label": "1차 정찰",
        "key": f"{ticker}-A1", "limit": limit, "stop": stop, "stop_pct": stop_pct, "decision": "매수",
        "note": condition, "condition_summary": condition, "score": score, "grade": "",
        "qty": 3, "amount_krw": 800_000, "max_loss_krw": 40_000, "is_new_position": True,
    }


def sample_summary(cfg: dict) -> dict:
    """미리보기용 summary (engine.daily.build_report_summary 결과와 같은 키)."""
    th = cfg["macro"]["thresholds"]
    macro_rows = [
        _macro_row(th, "fear-greed", "공포·탐욕 지수", "CNN Fear & Greed", "FEAR_GREED", 38, "/100",
                   [55.0, 48.0, 45.0, 50.0, 42.0, 38.0], "지금 구간: 공포"),
        _macro_row(th, "dgs10", "미국 10년물 국채금리", "DGS10", "DGS10", 4.17, "%",
                   [4.0, 4.3, 4.5, 4.2, 4.17], "3개월 변화 +0.30%p (보조 설명)"),
        _macro_row(th, "t10y2y", "장단기 금리차 (10년−2년)", "T10Y2Y", "T10Y2Y", 0.52, "%p",
                   [0.1, 0.3, 0.45, 0.52], short_range=True),
        _macro_row(th, "hy", "하이일드 스프레드", "BAMLH0A0HYM2", "BAMLH0A0HYM2", 3.05, "%", [3.2, 3.1, 3.3, 3.05]),
        _macro_row(th, "fed", "미국 기준금리 (상단)", "DFEDTARU", "DFEDTARU", 4.25, "%", [4.5, 4.5, 4.25],
                   value_before=4.5),
        _macro_row(th, "fx", "원/달러 환율", "DEXKOUS · KRW=X", "DEXKOUS", 1392.0, "원",
                   [1350.0, 1380.0, 1410.0, 1392.0], is_stale=True, **{"as_of": "2026-09-26"}),
    ]
    b1 = [
        _buy_row("NVDA", "엔비디아", 118.42, 109.80, -7.3, "RSI 27.4 → 31.9 · 거래량 부족(0.62배) · 실적 임박 10/9", 31),
        _buy_row("GOOGL", "알파벳 A", 1234.56, 1180.05, -4.4, "RSI 28.9 → 30.4", 27),
        _buy_row("ODFL", "올드 도미니언 프레이트 라인", 9.87, None, None,
                 "RSI 22.1 → 30.3 · 거래량 부족(0.41배) · 실적 임박 10/14 · 스윙 저점 확인 안 됨 — 손절가 미확인", 22),
        _buy_row("CDNS", "케이던스 디자인 시스템즈", 245.10, 236.00, -3.7, "", 18),
    ]
    return {
        "mode": "live", "mode_label": "실전", "as_of": pd.Timestamp("2026-09-30"),
        "buy_groups": {"b1": b1, "b2": [], "b3": [], "b9": []}, "buy_count": len(b1),
        "filtered_rows": [], "sell_rows": [], "warn_rows": [], "data_status_rows": [],
        "unfilled_rows": [], "hold_rows": [], "watch_rows": [], "held_tickers_count": 0,
        "max_concurrent": 8, "macro_rows": macro_rows,
    }


def _fake_line(ticker, name, start, n, drift, seed, status="open"):
    import numpy as np

    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n)
    v = np.cumsum(rng.normal(drift, 1.2, n))
    v[0] = rng.normal(0.3, 0.5)
    return {"ticker": ticker, "name": name, "status": status, "dates": [d.date().isoformat() for d in days],
            "values": [float(x) for x in v], "vs_qqq": float(v[-1] - drift * n * 0.4)}


def sample_pead(summary: dict) -> dict:
    """실적급등 추천 편지 미리보기 칸(engine.pead_watch.run의 letter와 같은 모양, 모두 가짜 숫자)."""
    import numpy as np

    summary.update(as_of=pd.Timestamp("2026-11-11"), letter_today_kst=pd.Timestamp("2026-11-12"),
                   thermometer={"temp": 58, "zone": "미지근함", "delta": 2, "data_date": "2026-11-11"},
                   puddle_v31={"personal": ["웅덩이 v3.2: QQQ 대비 +0.4%p, 보유 2종목입니다. (실제 돈 아님)"],
                               "group": ["웅덩이 v3.2: QQQ 대비 +0.4%p입니다. (실제 돈 아님)"]},
                   puddle_rebound={"personal": ["반등 확인 v2: QQQ 대비 −0.2%p, 보유 없음입니다. (실제 돈 아님)"],
                                   "group": ["반등 확인 v2: QQQ 대비 −0.2%p입니다. (실제 돈 아님)"]})
    L = {
        "as_of": "2026-11-11", "next_day": "2026-11-12",
        "actions": {"buy": [{"ticker": "TTWO", "name": "테이크투인터랙티브", "label": "나스닥100"},
                            {"ticker": "MCD", "name": "맥도날드", "label": "S&P100"}],
                    "sell": [{"ticker": "AMAT", "name": "어플라이드 머티어리얼즈", "label": "나스닥100"}]},
        "new": [{"ticker": "TTWO", "name": "테이크투인터랙티브", "label": "나스닥100", "d0": "2026-11-11", "ret": 0.083,
                 "excess": 0.090, "vol_ratio": 2.9, "buy_day": "2026-11-12", "check_day": "2027-02-10", "sell_day": "2027-03-25",
                 "late": False, "skip": None},
                {"ticker": "MCD", "name": "맥도날드", "label": "S&P100", "d0": "2026-11-10", "ret": 0.065, "excess": 0.056,
                 "vol_ratio": 2.1, "buy_day": "2026-11-12", "check_day": "2027-02-10", "sell_day": "2027-03-25",
                 "late": True, "skip": None}],
        "watching": [
            {"ticker": "APP", "name": "앱러빈", "label": "나스닥100", "status": "open", "held": 4, "hold": 90, "ret": 0.071,
             "vs_qqq": 0.064, "buy_day": "2026-11-06", "plan_day": "2027-03-19", "check_day": "2027-02-04"},
            {"ticker": "BKNG", "name": "부킹홀딩스", "label": "둘 다", "status": "open", "held": 5, "hold": 90, "ret": -0.012,
             "vs_qqq": -0.018, "buy_day": "2026-11-05", "plan_day": "2027-03-18", "check_day": "2027-02-03"},
            {"ticker": "AMAT", "name": "어플라이드 머티어리얼즈", "label": "나스닥100", "status": "open", "held": 89, "hold": 90,
             "ret": 0.153, "vs_qqq": 0.082, "buy_day": "2026-07-06", "plan_day": "2026-11-12", "check_day": "2026-09-29"}],
        "closed": [{"ticker": "CSX", "name": "CSX", "label": "나스닥100", "buy_day": "2026-07-01", "sell_day": "2026-11-06",
                    "ret": 0.042, "vs_qqq": -0.011},
                   {"ticker": "LMT", "name": "록히드 마틴", "label": "S&P100", "buy_day": "2026-06-26", "sell_day": "2026-11-03",
                    "ret": 0.118, "vs_qqq": 0.071}],
        "scorecards": {"main": {"hold": 90, "n": 2, "mean": 2.71, "win": 50.0, "basis": "cost"},
                       "compare": {"hold": 60, "n": 4, "mean": -0.42, "win": 50.0, "basis": "cost"}},
        "earnings": {"rows": [{"ticker": "NVDA", "name": "엔비디아", "label": "둘 다", "date": "2026-11-18"},
                              {"ticker": "WMT", "name": "월마트", "label": "둘 다", "date": "2026-11-19"},
                              {"ticker": "INTU", "name": "인튜이트", "label": "나스닥100", "date": "2026-11-19"}],
                     "failed": 3},
        "counts": {"new": 2, "watching": 3},
    }
    lines = [_fake_line("APP", "앱러빈", "2026-11-06", 4, 1.6, 1), _fake_line("BKNG", "부킹홀딩스", "2026-11-05", 5, -0.2, 2),
             _fake_line("AMAT", "어플라이드 머티어리얼즈", "2026-07-06", 89, 0.17, 3),
             _fake_line("CSX", "CSX", "2026-07-01", 91, 0.05, 4, "closed"),
             _fake_line("LMT", "록히드 마틴", "2026-06-26", 91, 0.13, 5, "closed")]
    qd = pd.bdate_range("2026-06-26", "2026-11-11")
    qv = np.cumsum(np.random.default_rng(9).normal(0.06, 0.8, len(qd)))
    L["chart"] = {"lines": lines, "qqq": {"dates": [d.date().isoformat() for d in qd], "values": [float(x) for x in qv]}}
    summary["pead_watch"] = {"status": "ok", "letter": L}
    from notify import pead_chart

    summary["pead_chart"] = pead_chart.render(L["chart"], OUT / "pead_chart_preview.png", "2026-11-11")
    return summary


def _screenshot(paths: dict[str, Path], tag: str) -> list[Path]:
    """paths{이름: html} -> outputs/{이름}_{tag}_{폭}.png (폰 375px·PC 1200px, 전체 페이지)."""
    from playwright.sync_api import sync_playwright

    shots = []
    with sync_playwright() as p:
        # 설치된 Playwright 버전과 브라우저 버전이 다를 때를 위해 실행 파일 경로를 환경변수로 받을 수 있다
        exe = os.environ.get("CHROMIUM_PATH")
        browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        for name, html in paths.items():
            for width in (375, 1200):
                page = browser.new_page(viewport={"width": width, "height": 800}, device_scale_factor=2)
                page.goto(html.resolve().as_uri())
                page.wait_for_timeout(300)
                out = OUT / f"{name}_{tag}_{width}.png"
                page.screenshot(path=str(out), full_page=True)
                shots.append(out)
                page.close()
        browser.close()
    return shots


def main() -> None:
    parser = argparse.ArgumentParser(description="샘플 데이터로 보고서 미리보기 만들기")
    parser.add_argument("--screenshot", metavar="TAG", help="스크린샷 파일 이름에 붙일 표시 (예: before, after)")
    parser.add_argument("--pead", action="store_true", help="실적급등 추천 편지 개인판·단체방판 미리보기(가짜 숫자)")
    args = parser.parse_args()

    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    summary = sample_summary(cfg)
    OUT.mkdir(exist_ok=True)
    if args.pead:
        sample_pead(summary)
        paths = {}
        for ed in ("personal", "group"):
            out = report_html.render_report(summary, cfg, OUT, edition=ed)
            paths[f"pead_{ed}"] = out.replace(OUT / f"report_preview_pead_{ed}.html")
            print(f"미리보기({ed}): {paths[f'pead_{ed}'].resolve()}")
        if args.screenshot:
            for shot in _screenshot(paths, args.screenshot):
                print(f"스크린샷: {shot}")
        return

    private = report_html.render_report(summary, cfg, OUT)
    private_preview = private.replace(OUT / "report_preview.html")
    text = briefing.build_report_caption(summary, cfg)
    (OUT / "caption_preview.txt").write_text(text, encoding="utf-8")

    print(f"미리보기: {private_preview}")
    print("── 첨부 설명 ──")
    print(text)
    if args.screenshot:
        for shot in _screenshot({"report": private_preview}, args.screenshot):
            print(f"스크린샷: {shot}")


if __name__ == "__main__":
    main()
