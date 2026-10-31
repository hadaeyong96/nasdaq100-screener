"""가상 실험 수익곡선 그래프 (notify/curve_chart.py) — 가짜 기록, 네트워크 막음."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from notify import curve_chart as cc

START = "2026-10-23"
TICKERS = ("NVDA", "AMD", "MSFT")


def fake_history(n: int = 60, start: str = START):
    """가짜 n거래일 기록: v3.1 앞섬·반등 확인 뒤처짐, 매매 3건(v3.1 1차 NVDA·2차 NVDA, 반등 확인 AMD 1차 뒤 매도)."""
    cal = pd.bdate_range(start, periods=n)
    t = np.arange(n)
    q = 99.9e6 * (1 + 0.0006 * t + 0.01 * np.sin(t / 5))
    v31 = q * (1 + 0.0004 * t)
    rb = q * (1 - 0.0003 * t)
    d = [str(x.date()) for x in cal]
    v31_rows = [{"date": d[i], "v31": v31[i], "qqq": q[i], "v3": v31[i], "fills": [], "sells": [],
                 "random": {"10": q[i] * (1 - 0.0004 * i), "50": q[i], "90": q[i] * (1 + 0.0006 * i)}} for i in range(n)]
    rb_rows = [{"date": d[i], "value": rb[i], "qqq": q[i], "v31": v31[i], "fills": [], "sells": []} for i in range(n)]
    trades = [(v31_rows, 10, "fills", {"tk": "NVDA", "tranche": 1, "usd": 1.0}), (v31_rows, 25, "fills", {"tk": "NVDA", "tranche": 2, "usd": 1.0}),
              (rb_rows, 15, "fills", {"tk": "AMD", "tranche": 1, "usd": 1.0}),
              (rb_rows, 40, "sells", {"tk": "AMD", "reason": "직전 120거래일 최고가 돌파"})]
    for rows, i, key, t in trades:
        if i < n:
            rows[i][key] = [{**t, ("date" if key == "fills" else "exit_date"): d[i]}]
    return v31_rows, rb_rows


def test_curve_frame_rebases_to_100():
    v, r = fake_history(10)
    f = cc.curve_frame(v, r, START)
    assert f.loc[START, ["qqq", "v31", "rebound"]].tolist() == pytest.approx([100.0, 100.0, 100.0])
    i = 7
    assert f["v31"].iloc[i] == pytest.approx(v[i]["v31"] / v[0]["v31"] * 100)
    assert f["rebound"].iloc[i] == pytest.approx(r[i]["value"] / r[0]["value"] * 100)
    assert f["band_hi"].iloc[i] == pytest.approx(v[i]["random"]["90"] / v[0]["v31"] * 100)
    early = [{**v[0], "date": "2026-10-22"}] + v  # 시작일 전 줄은 버린다
    assert cc.curve_frame(early, r, START).index[0] == pd.Timestamp(START)


def test_curve_frame_missing_ledger_is_nan():
    v, _ = fake_history(6)
    f = cc.curve_frame(v, [], START)
    assert f["rebound"].isna().all() and f["v31"].notna().all()


def test_ahead_behind_segments():
    idx = pd.bdate_range(START, periods=6)
    q = pd.Series([100, 100, 100, 100, 100, 100.0], index=idx)
    line = pd.Series([100, 101, 102, 99, 98, 100.0], index=idx)
    assert cc.ahead_segments(line, q) == [(idx[1], idx[2], "ahead"), (idx[3], idx[4], "behind")]
    v, r = fake_history(60)
    f = cc.curve_frame(v, r, START)
    assert {k for *_, k in cc.ahead_segments(f["v31"], f["qqq"])} == {"ahead"}
    assert {k for *_, k in cc.ahead_segments(f["rebound"], f["qqq"])} == {"behind"}


def test_trade_marks():
    v, r = fake_history(60)
    f = cc.curve_frame(v, r, START)
    m = cc.trade_marks(v, f, "v31") + cc.trade_marks(r, f, "rebound")
    assert [(x["kind"], x["tk"], x["line"]) for x in m] == [("buy1", "NVDA", "v31"), ("buy2", "NVDA", "v31"),
                                                            ("buy1", "AMD", "rebound"), ("sell", "AMD", "rebound")]
    assert m[0]["y"] == pytest.approx(f["v31"].iloc[10]) and m[3]["date"] == f.index[40]


def test_end_labels_text():
    idx = pd.bdate_range(START, periods=2)
    f = pd.DataFrame({"qqq": [100, 100.0], "v31": [100, 100.4], "rebound": [100, 99.75]}, index=idx)
    assert [t for _, t, _ in cc.end_labels(f)] == ["QQQ 100.0", "v3.2 100.4 (+0.4%p)", "반등 확인 v2 99.8 (-0.2%p)"]  # 2026-10-27 시총 판


def test_render_personal_has_tickers_group_has_none(tmp_path):
    v, r = fake_history(60)
    out = cc.build(v, r, tmp_path, start=START, as_of=v[-1]["date"])
    assert out["days"] == 60
    p, g = out["personal"], out["group"]
    assert p["path"].exists() and g["path"].exists()
    assert {"NVDA", "AMD"} <= set(p["labels"])
    assert all(t not in lab for lab in g["labels"] for t in TICKERS)
    assert "가상 실험 성적 — QQQ 기준 (시작 10/23 = 100)" in g["labels"]
    assert any(lab.startswith("가상 1억 · 실제 돈 아님 · 미국 ") for lab in g["labels"])
    assert any(lab.startswith("v3.2 ") and "%p" in lab for lab in g["labels"])
    from matplotlib import image

    h, w = image.imread(str(g["path"])).shape[:2]
    assert (w, h) == (1200, 700)


def test_fewer_than_min_days_shows_waiting(tmp_path):
    v, r = fake_history(3)
    out = cc.build(v, r, tmp_path, start=START, as_of=v[-1]["date"], min_days=5)
    assert out["days"] == 3
    for ed in ("personal", "group"):
        assert out[ed]["waiting"] and out[ed]["labels"][0] == "기록이 쌓이는 중입니다(3일째)."


def test_chart_day_is_monday_kst():
    cfg = {"chart": {"weekday": "월", "min_days": 5}}
    assert cc.is_chart_day(pd.Timestamp("2026-10-26"), cfg)  # 월
    assert not cc.is_chart_day(pd.Timestamp("2026-10-27"), cfg)
    assert cc.is_chart_day(pd.Timestamp("2026-10-30"), {"chart": {"weekday": "금"}})


def test_config_chart_block(cfg):
    assert cfg["chart"]["weekday"] == "월" and cfg["chart"]["min_days"] == 5


# ── 넣을 곳: daily 연결·편지 자료 문제·HTML·텔레그램 ─────────────────────────


def test_attach_curve_chart_ok_and_failure(tmp_path, cfg, monkeypatch):
    from engine import daily

    v, r = fake_history(8)
    monkeypatch.setattr(cc, "read_rows", lambda p: (v, r))
    s = {"as_of": pd.Timestamp(v[-1]["date"]), "letter_check": {"lag": 0, "notes": []}}
    daily._attach_curve_chart(s, cfg, out_dir=tmp_path)
    assert Path(s["curve_chart"]["personal"]).exists() and Path(s["curve_chart"]["group"]).exists()
    assert not s["curve_chart"].get("error")

    def boom(p):
        raise RuntimeError("x")

    monkeypatch.setattr(cc, "read_rows", boom)
    s2 = {"as_of": pd.Timestamp(v[-1]["date"]), "letter_check": {"lag": 0, "notes": []}}
    daily._attach_curve_chart(s2, cfg, out_dir=tmp_path)
    assert s2["curve_chart"]["error"]
    from notify import letter_plain as lp

    assert "그래프를 만들지 못했습니다." in lp.problems(s2, lp.load_phrases(), lp.PERSONAL)
    assert "그래프를 만들지 못했습니다." not in lp.problems(s, lp.load_phrases(), lp.PERSONAL)


def test_report_html_embeds_edition_chart(tmp_path, cfg):
    from notify import report_html
    from tests.test_fresh_start import _summary

    v, r = fake_history(30)  # 매매가 있어야 개인판(종목명)과 단체방판 그림이 다르다
    out = cc.build(v, r, tmp_path, start=START, as_of=v[-1]["date"])
    s = _summary(copy.deepcopy(cfg))
    s["curve_chart"] = {"personal": str(out["personal"]["path"]), "group": str(out["group"]["path"])}
    ctx_p = report_html.build_context(s, cfg, "personal")
    ctx_g = report_html.build_context(s, cfg, "group")
    assert ctx_p["curve_chart_src"].startswith("data:image/png;base64,")
    assert ctx_p["curve_chart_src"] != ctx_g["curve_chart_src"]  # 판마다 자기 그림
    html = report_html.render_report(s, cfg, tmp_path).read_text(encoding="utf-8")  # 월요일 편지 — 그래프는 본문 조건부 줄 아래
    extras = html[html.index('id="sec-extras"'):html.index('id="fold-details"')]
    assert 'class="curve-chart"' in extras and html.count('aria-label="가상 실험 성적 그래프"') == 1
    tue = report_html.render_report({**s, "letter_today_kst": s["letter_today_kst"] + pd.Timedelta(days=1)}, cfg,
                                    tmp_path).read_text(encoding="utf-8")  # 평일 — 「자세히 보기」 가상 실험 상세 안(2026-10-31)
    lab = tue[tue.index('id="sec-lab"'):tue.index('id="guide"')]
    assert 'class="curve-chart"' in lab and tue.count('aria-label="가상 실험 성적 그래프"') == 1
    s.pop("curve_chart")
    assert report_html.build_context(s, cfg, "personal")["curve_chart_src"] is None


class _Resp:
    ok = True
    status_code = 200

    def raise_for_status(self):
        return None


def _send_env(monkeypatch, tmp_path, cfg):
    from notify import telegram

    calls = []

    def fake_post(url, data=None, files=None, timeout=None):
        calls.append((url.rsplit("/", 1)[-1], (data or {}).get("caption")))
        return _Resp()

    monkeypatch.setattr(telegram.requests, "post", fake_post)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "1")
    monkeypatch.setenv("TELEGRAM_GROUP_CHAT_ID", "2")
    monkeypatch.setattr(telegram, "ROOT", tmp_path)
    monkeypatch.setattr(telegram.db, "db_path_for_mode", lambda m: tmp_path / "s.db")
    return telegram, calls


def _send_summary(tmp_path, today, chart=True):
    rep = tmp_path / "r.html"
    rep.write_text("<html></html>", encoding="utf-8")
    img = tmp_path / "c.png"
    img.write_bytes(b"\x89PNG")
    s = {"as_of": pd.Timestamp("2026-10-30"), "mode": "live", "report_path": str(rep), "group_report_path": str(rep),
         "letter_today_kst": pd.Timestamp(today)}
    if chart:
        s["curve_chart"] = {"personal": str(img), "group": str(img)}
    return s, rep


@pytest.mark.parametrize("today", ["2026-11-02", "2026-11-03"])
def test_only_report_file_sent_even_on_monday(tmp_path, monkeypatch, cfg, today):
    """2026-10-29: 월요일 성적 그래프 sendPhoto 중단(그래프는 보고서 안에) — 요일과 관계없이 두 방 모두 sendDocument 하나만."""
    telegram, calls = _send_env(monkeypatch, tmp_path, cfg)
    s, rep = _send_summary(tmp_path, today)
    telegram.send_briefing("본문", s, cfg)
    assert [m for m, _ in calls] == ["sendDocument"]
    calls.clear()
    telegram.send_group_briefing("본문", rep, s, cfg)
    assert [m for m, _ in calls] == ["sendDocument"]


def test_letter_still_sent_when_chart_failed(tmp_path, monkeypatch, cfg):
    telegram, calls = _send_env(monkeypatch, tmp_path, cfg)
    s, rep = _send_summary(tmp_path, "2026-11-02", chart=False)
    s["curve_chart"] = {"error": "RuntimeError"}
    telegram.send_briefing("본문", s, cfg)
    assert [m for m, _ in calls] == ["sendDocument"]
    assert calls[0][1].endswith(" · 자료 문제 있음")  # 「자료 문제: 그래프를 만들지 못했습니다.」


# ── 웅덩이 후보 종목 차트 (notify/candidate_chart, 개인판 HTML 전용) ─────────


def fake_candidate(tk="AAPL", day=5, stage="wait", n=120, seed=0):
    """가짜 후보 자료(engine.puddle_rebound.candidate_data와 같은 모양) — 지표는 장부 함수로 계산."""
    from core import puddle_rebound as pr
    from core import puddle_v31 as pv

    cal = pd.bdate_range(end="2026-10-23", periods=330)
    rng = np.random.default_rng(seed)
    close = pd.DataFrame({tk: 100 * np.cumprod(1 + 0.001 + 0.015 * rng.standard_normal(len(cal)))}, index=cal)
    ind = {**pv.indicators(close, close * 0 + 1e6), **pr.rebound_indicators(close)}
    win = cal[-n:]
    m, sig, cross = pr.macd_cross(close[tk])
    return {"tk": tk, "tranche": 1, "stage": stage, "day": day, "rsi": float(ind["rsi"][tk].iloc[-1]), "as_of": cal[-1],
            "level": 40.0, "close": close[tk].reindex(win), "sma60": ind["sma60"][tk].reindex(win),
            "sma120": ind["sma120"][tk].reindex(win), "sma200": ind["sma200"][tk].reindex(win), "rsi_line": ind["rsi"][tk].reindex(win),
            "macd": m.reindex(win), "signal": sig.reindex(win), "golden": [d for d in win if cross.get(d) == 1.0],
            "recross": [d for d in win if pr.rsi_cross_up(ind["rsi_prev"][tk].get(d), ind["rsi"][tk].get(d))],
            "puddle": (cal[-day], cal[-1]), "trades": [{"date": cal[-3], "kind": "buy1"}]}


@pytest.mark.parametrize("n, expect", [(0, 0), (2, 2), (5, 4)])
def test_candidate_targets_and_images_count(tmp_path, n, expect):
    from engine import puddle_rebound as eng
    from notify import candidate_chart as cand

    row = {"date": "2026-10-23", "waiting": [{"tk": f"T{i}", "tranche": 1, "day": 9 - i, "rsi": 35.0} for i in range(n)], "buys": []}
    targets = eng.chart_targets([row])
    assert len(targets) == expect and [t["day"] for t in targets] == sorted(t["day"] for t in targets)  # 대기일 짧은 순
    imgs = cand.render_all([fake_candidate(t["tk"], t["day"]) for t in targets], tmp_path)
    assert len(imgs) == expect and all(i["path"].exists() for i in imgs)


def test_candidate_buy_stage_day_and_title(tmp_path):
    from engine import puddle_rebound as eng
    from notify import candidate_chart as cand

    prev = {"date": "2026-10-22", "waiting": [{"tk": "AMD", "tranche": 1, "day": 6, "rsi": 39.0}], "buys": []}
    last = {"date": "2026-10-23", "waiting": [{"tk": "AAPL", "tranche": 2, "day": 9, "rsi": 36.04}],
            "buys": [{"tk": "AMD", "tranche": 1, "rsi": 41.23}]}
    t = eng.chart_targets([prev, last])
    assert [(x["tk"], x["stage"], x["day"]) for x in t] == [("AMD", "buy", 7), ("AAPL", "wait", 9)]
    assert cand.title(t[0]) == "AMD — 웅덩이 7일째 · RSI 41.2 · 반등 확인 매수"
    assert cand.title(t[1]) == "AAPL — 웅덩이 9일째 · RSI 36.0 · 반등 확인 대기 (2차)"
    from matplotlib import image

    p = cand.render(fake_candidate(), tmp_path / "c.png")["path"]
    h, w = image.imread(str(p)).shape[:2]
    assert (w, h) == (1200, 900)


def test_candidate_data_matches_ledger_indicators(tmp_path, monkeypatch):
    """엔진이 넘기는 후보 차트 지표 = 장부 계산(World 지표·rebound_indicators) 값 그대로."""
    from core import puddle_rebound as pr
    from engine import puddle_rebound as eng
    from tests.test_puddle_rebound import _history_world, _run

    cal, data = _history_world()
    _run(tmp_path, str(cal[-17].date()), monkeypatch)
    hit = None
    for k in range(-16, 0):  # 대기·매수 종목이 있는 날을 찾는다(미리보기 — 저장 안 함)
        r = _run(tmp_path, str(cal[k].date()), monkeypatch, actions=False)
        if r.get("candidates"):
            hit = (cal[k], r)
            break
    assert hit, "합성 세계에 웅덩이 후보가 있어야 한다"
    d, r = hit
    c = r["candidates"][0]
    close = pd.DataFrame({t: data[t]["close"] for t in ("AAA", "BBB")}).reindex(cal)
    ind = pr.rebound_indicators(close)
    win = c["close"].index
    assert len(win) == 120 and win[-1] == d
    pd.testing.assert_series_equal(c["rsi_line"], ind["rsi"]["AAA"].reindex(win), check_names=False)
    m, sig, cross = pr.macd_cross(close["AAA"].dropna()[lambda s: s.index <= d])
    pd.testing.assert_series_equal(c["macd"], m.reindex(win), check_names=False)
    assert c["golden"] == [x for x in win if cross.get(x) == 1.0]
    assert all(ind["gc3"]["AAA"].get(x) == 1.0 for x in c["golden"])  # 장부의 3일 창이 그날 1.0
    assert c["sma60"].equals(close["AAA"].rolling(60, min_periods=60).mean().reindex(win))


def test_candidate_charts_only_in_personal_html(tmp_path, cfg):
    from engine import daily
    from notify import report_html
    from tests.test_fresh_start import _summary

    s = _summary(copy.deepcopy(cfg))
    s["_rebound_candidates"] = [fake_candidate("AAPL", 4), fake_candidate("MSFT", 7, seed=1)]
    daily._attach_candidate_charts(s, out_dir=tmp_path)
    assert len(s["candidate_charts"]) == 2
    personal = report_html.render_report(s, cfg, tmp_path).read_text(encoding="utf-8")
    group = report_html.render_report(s, cfg, tmp_path, edition="group").read_text(encoding="utf-8")
    assert personal.count('class="curve-chart cand-chart"') == 2  # 2026-10-28: ■ 웅덩이 후보 칸 안(접기 아님)
    assert "웅덩이 후보 차트" not in group and "cand-chart\"" not in group and "AAPL — 웅덩이" not in group
    s2 = _summary(copy.deepcopy(cfg))
    daily._attach_candidate_charts(s2, out_dir=tmp_path)
    assert s2["candidate_charts"] == [] and "▶ 웅덩이 후보 차트" not in report_html.render_report(s2, cfg, tmp_path).read_text(encoding="utf-8")
