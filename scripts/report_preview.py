"""샘플 데이터로 개인용·공개용 보고서 미리보기를 만든다 (네트워크 없음).

사용법:
    python scripts/report_preview.py                  # outputs/report_preview.html
    python scripts/report_preview.py --screenshot after  # + 폰 폭(375)·PC 폭(1200) 스크린샷
    python scripts/report_preview.py --pead              # 실적급등 추천 편지 개인판·단체방판(가짜 신호 2·관찰 3·끝남 2·비교 그래프·실적 뉴스 3)

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


def _fake_px(seed, days, drift=0.0005, vol=0.012, start=100.0, jump_at=None, jump=0.0):
    """가짜 일봉(open·high·low·close·volume) — 미리보기 전용."""
    import numpy as np

    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, len(days))
    if jump_at is not None:
        r[days.get_loc(pd.Timestamp(jump_at))] += jump
    c = start * np.cumprod(1 + r)
    o = c / (1 + rng.normal(0, vol / 2, len(days)))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol / 2, len(days))))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol / 2, len(days))))
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": 1e6}, index=days)


# 미리보기 보유 종목: (티커, 이름, 소속, 매수일, 시드, 하루 기대, 시작가) — 새 추천 1(APP) · 매도 임박 1(AMAT) · 나머지 4
_PEAD_OPEN = [("APP", "앱러빈", "나스닥100", "2026-11-06", 11, 0.006, 400.0),
              ("AMAT", "어플라이드 머티어리얼즈", "나스닥100", None, 12, 0.0012, 190.0),
              ("BKNG", "부킹홀딩스", "둘 다", "2026-10-15", 13, -0.0004, 5100.0),
              ("IDXX", "아이덱스 래버러토리스", "나스닥100", "2026-09-16", 14, 0.0015, 610.0),
              ("ROP", "로퍼 테크놀로지스", "나스닥100", "2026-08-20", 15, -0.0008, 540.0),
              ("MCD", "맥도날드", "S&P100", "2026-10-01", 16, 0.0007, 300.0)]
_PEAD_CLOSED = [("CSX", "CSX", "나스닥100", "2026-06-23", 17, 0.0003, 35.0),
                ("LMT", "록히드 마틴", "S&P100", "2026-06-18", 18, 0.0011, 470.0)]


def sample_pead_letter(as_of="2026-11-11") -> dict:
    """가짜 가격·매매로 engine.pead_watch.chart_data를 실제로 돌려 편지 칸(letter)을 만든다 — 미리보기·테스트 공용, 네트워크 없음."""
    from core import pead_timing as pt
    from engine import pead_watch as ew

    days = pd.bdate_range("2026-01-02", "2027-12-31")
    ia = int(days.get_loc(pd.Timestamp(as_of)))
    raw = {"QQQ": _fake_px(9, days, 0.0004, 0.009, 560.0)}
    trades, names = [], {"QQQ": "QQQ"}
    for status, rows in (("open", _PEAD_OPEN), ("closed", _PEAD_CLOSED)):
        for t, name, label, buy, seed, drift, start in rows:
            ib = int(days.get_loc(pd.Timestamp(buy))) if buy else ia + 1 - 90  # None = 다음 거래일 매도(매도 임박)
            buy = days[ib].date().isoformat()
            raw[t] = _fake_px(seed, days, drift, 0.016, start, days[ib - 1], 0.09)
            names[t] = name
            tr = {"ledger": "main", "ticker": t, "d0": days[ib - 1].date().isoformat(), "label": label, "status": status,
                  "buy_day": buy, "plan_day": days[ib + 90].date().isoformat(), "buy_px": float(raw[t]["open"].iloc[ib]),
                  "q_buy": float(raw["QQQ"]["open"].iloc[ib])}
            if status == "closed":
                tr.update(sell_day=days[ib + 90].date().isoformat(), sell_px=float(raw[t]["close"].iloc[ib + 90]),
                          q_sell=float(raw["QQQ"]["close"].iloc[ib + 90]))
            trades.append(tr)
    raw = {t: d.loc[:as_of] for t, d in raw.items()}
    al = {t: pt.align(d, days) for t, d in raw.items()}
    sigs = {("APP", trades[0]["d0"]): {"ret": 0.310, "excess": 0.375, "vol_ratio": 5.1}}
    chart = ew.chart_data(trades, al, al["QQQ"], days, ia, names, raw, sigs, 90, 60)
    nxt = days[ia + 1].date().isoformat()
    soon = [c for c in chart["cards"] if c["sell_in"] == 1]
    return {
        "as_of": as_of, "next_day": nxt,
        "actions": {"buy": [{"ticker": "TTWO", "name": "테이크투인터랙티브", "label": "나스닥100"}],
                    "sell": [{"ticker": c["ticker"], "name": c["name"], "label": c["label"]} for c in soon]},
        "new": [{"ticker": "TTWO", "name": "테이크투인터랙티브", "label": "나스닥100", "d0": as_of, "ret": 0.083,
                 "excess": 0.090, "vol_ratio": 2.9, "buy_day": nxt, "check_day": days[ia + 61].date().isoformat(),
                 "sell_day": days[ia + 91].date().isoformat(), "late": False, "skip": None}],
        "watching": [], "closed": [
            {"ticker": "CSX", "name": "CSX", "label": "나스닥100", "buy_day": "2026-06-23", "sell_day": "2026-10-30",
             "ret": 0.042, "vs_qqq": -0.011},
            {"ticker": "LMT", "name": "록히드 마틴", "label": "S&P100", "buy_day": "2026-06-18", "sell_day": "2026-10-27",
             "ret": 0.118, "vs_qqq": 0.071}],
        "scorecards": {"main": {"hold": 90, "n": 2, "mean": 2.71, "win": 50.0, "basis": "cost"},
                       "compare": {"hold": 60, "n": 4, "mean": -0.42, "win": 50.0, "basis": "cost"}},
        "earnings": {"rows": [{"ticker": "NVDA", "name": "엔비디아", "label": "둘 다", "date": "2026-11-18"},
                              {"ticker": "WMT", "name": "월마트", "label": "둘 다", "date": "2026-11-19"}], "failed": 3},
        "counts": {"new": 1, "watching": len(chart["cards"])},
        "chart": chart,
    }


def sample_pead(summary: dict) -> dict:
    """실적급등 추천 편지 v2 미리보기(가짜 숫자): 새 추천 1(매수 대기) · 보유 6(새 추천 1 · 매도 임박 1 · 나머지 4) · 끝난 2."""
    summary.update(as_of=pd.Timestamp("2026-11-11"), letter_today_kst=pd.Timestamp("2026-11-12"),
                   thermometer={"temp": 58, "zone": "미지근함", "delta": 2, "data_date": "2026-11-11"},
                   puddle_v31={"personal": ["웅덩이 v3.2: QQQ 대비 +0.4%p, 보유 2종목입니다. (실제 돈 아님)"],
                               "group": ["웅덩이 v3.2: QQQ 대비 +0.4%p입니다. (실제 돈 아님)"]},
                   puddle_rebound={"personal": ["반등 확인 v2: QQQ 대비 −0.2%p, 보유 없음입니다. (실제 돈 아님)"],
                                   "group": ["반등 확인 v2: QQQ 대비 −0.2%p입니다. (실제 돈 아님)"]})
    L = sample_pead_letter()
    summary["pead_watch"] = {"status": "ok", "letter": L}
    now = pd.Timestamp("2026-11-11 22:00", tz="UTC")
    summary["earnings_news"] = {"status": "ok", "failed": [], "top": [
        {"title": "앱러빈, 3분기 매출 예상 상회…주가 31% 급등", "source": "연합뉴스", "url": "https://example.com/1", "published": now},
        {"title": "테이크투, 'GTA6' 기대에 실적 전망 상향", "source": "한국경제", "url": "https://example.com/2", "published": now},
        {"title": "엔비디아 실적 발표 앞두고 반도체주 관망", "source": "뉴스1", "url": "https://example.com/3", "published": now}]}
    from engine import daily
    from notify import pead_chart

    summary["pead_chart"] = pead_chart.render(L["chart"], OUT / "pead_chart_preview.png", "2026-11-11")
    summary["pead_cards"] = daily.attach_pead_cards(L, OUT, "preview", summary["letter_today_kst"])
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
