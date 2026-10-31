"""시장 온도계 (thermometer/) — 네트워크 없이 합성 자료로."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from factory import blocks
from thermometer import __main__ as tmain
from thermometer import history
from thermometer.core import (get_temperature, load_config, score_breadth, score_fear_greed, score_gap, score_vix,
                              zone_of)
from thermometer.message import build_message, change_text

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config()
CFG_ON = {**CFG, "telegram": {"enabled": True}}  # 독립 메시지 켠 설정(2026-10-29부터 기본은 꺼짐) — 발송 경로 시험용
KST = tmain.KST


def make_prices(n=260, end="2026-10-27", seed=0, n_stocks=10, with_fg=True):
    idx = pd.bdate_range(end=end, periods=n)
    rng = np.random.default_rng(seed)
    qqq = pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)), index=idx)
    stocks = pd.DataFrame({f"S{i}": 50 * np.cumprod(1 + rng.normal(0.0003, 0.015, n)) for i in range(n_stocks)}, index=idx)
    vclose = pd.Series(18 + rng.normal(0, 2, n), index=idx).clip(lower=10)
    vix = pd.DataFrame({"close": vclose, "high": vclose + 1}, index=idx)
    fg = {d.date().isoformat(): float(40 + i % 30) for i, d in enumerate(idx)} if with_fg else {}
    return {"qqq": qqq, "stocks": stocks, "vix": vix, "fear_greed": fg}


def _cut(prices, ts):
    out = dict(prices)
    out["qqq"] = prices["qqq"].loc[:ts]
    out["stocks"] = prices["stocks"].loc[:ts]
    out["vix"] = prices["vix"].loc[:ts]
    out["fear_greed"] = {k: v for k, v in prices["fear_greed"].items() if k <= ts.date().isoformat()}
    return out


# ── 경계값 ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("x,deg", [(-15, 0), (15, 20), (0, 10), (-25, 0), (30, 20), (7.5, 15), (-7.5, 5)])  # ±15% (2026-10-28)
def test_gap_boundaries(x, deg):
    assert score_gap(x, CFG) == pytest.approx(deg)


@pytest.mark.parametrize("v,deg", [(12, 20), (35, 0), (50, 0), (9, 20), (23.5, 10)])
def test_vix_boundaries(v, deg):
    assert score_vix(v, CFG) == pytest.approx(deg)


@pytest.mark.parametrize("p,deg", [(0, 0), (100, 20), (50, 10)])
def test_breadth_and_fear_greed_boundaries(p, deg):
    assert score_breadth(p, CFG) == pytest.approx(deg)
    assert score_fear_greed(p, CFG) == pytest.approx(deg)


@pytest.mark.parametrize("t,name", [(0, "얼음"), (20, "얼음"), (21, "차가움"), (40, "차가움"), (41, "미지근함"), (60, "미지근함"),
                                    (61, "따뜻함"), (80, "따뜻함"), (81, "뜨거움"), (100, "뜨거움")])
def test_zone_boundaries(t, name):
    assert zone_of(t, CFG)["name"] == name


# ── 계산 ────────────────────────────────────────────────────────────────────


def test_full_temperature_is_rounded_sum():
    p = make_prices()
    r = get_temperature("2026-10-27", prices=p, cfg=CFG)
    assert r["missing"] == [] and r["data_date"] == "2026-10-27"
    total = sum(v["deg"] for v in r["parts"].values())
    assert r["temp"] == int(np.floor(total + 0.5))
    assert 0 <= r["temp"] <= 100


def test_missing_items_rescaled():
    p = make_prices()
    p4 = {**p, "fear_greed": {}}
    r = get_temperature("2026-10-27", prices=p4, cfg=CFG)
    assert r["missing"] == ["fear_greed"]
    total = sum(v["deg"] for k, v in r["parts"].items() if k != "fear_greed")
    assert r["temp"] == int(np.floor(total * 100 / 80 + 0.5))
    r2 = get_temperature("2026-10-27", prices={"qqq": p["qqq"]}, cfg=CFG)
    assert r2["missing"] == ["breadth", "vix", "fear_greed"]
    assert r2["temp"] == int(np.floor((r2["parts"]["sma_fast"]["deg"] + r2["parts"]["sma_slow"]["deg"]) * 100 / 40 + 0.5))
    msg = build_message(r2, CFG, datetime(2026, 10, 28).date())
    assert "▶ 자료 없음: 시장 폭, 공포 지수 VIX, 공포·탐욕. 나머지로 환산했습니다." in msg


def test_stale_fear_greed_is_missing():
    p = make_prices()
    p["fear_greed"] = {"2026-10-20": 50.0}  # 7일 전 → stale_days(3) 넘음
    assert "fear_greed" in get_temperature("2026-10-27", prices=p, cfg=CFG)["missing"]


def test_strong_recovery_flag():
    idx = pd.bdate_range(end="2026-10-27", periods=80)
    down = np.linspace(100, 60, 80)
    s = pd.DataFrame({f"S{i}": down.copy() for i in range(10)}, index=idx)
    s.iloc[-1] = 200.0  # 오늘 전 종목이 50일선 위로
    p = {"qqq": pd.Series(np.linspace(80, 100, 80), index=idx), "stocks": s}
    b = get_temperature("2026-10-27", prices=p, cfg=CFG)["parts"]["breadth"]
    assert b["raw"] == pytest.approx(100) and b["strong_recovery"] is True
    up = pd.DataFrame({f"S{i}": np.linspace(60, 100, 80) for i in range(10)}, index=idx)  # 늘 100% → 회복 아님
    b2 = get_temperature("2026-10-27", prices={**p, "stocks": up}, cfg=CFG)["parts"]["breadth"]
    assert b2["strong_recovery"] is False
    msg = build_message(get_temperature("2026-10-27", prices=p, cfg=CFG), CFG, datetime(2026, 10, 28).date())
    assert "(50일선 위 종목 100%, 강한 회복)" in msg


def test_breadth_missing_when_coverage_low():
    p = make_prices()
    p["members"] = set(p["stocks"].columns) | {f"X{i}" for i in range(10)}  # 절반은 가격 없음
    assert "breadth" in get_temperature("2026-10-27", prices=p, cfg=CFG)["missing"]


def test_vix_peak_note():
    idx = pd.bdate_range(end="2026-10-27", periods=30)
    close = pd.Series(18.0, index=idx)
    high = pd.Series(19.0, index=idx)
    high.iloc[-6] = 31.2
    p = {"qqq": pd.Series(100.0, index=idx), "vix": pd.DataFrame({"close": close, "high": high})}
    r = get_temperature("2026-10-27", prices=p, cfg=CFG)
    assert r["parts"]["vix"]["peak"] == pytest.approx(31.2)
    assert "(18.0, 지난달 31.2에서 하락)" in build_message(r, CFG, datetime(2026, 10, 28).date())
    high.iloc[-6] = 24.0  # 25 미만이면 덧붙이지 않음
    p["vix"] = pd.DataFrame({"close": close, "high": high})
    assert get_temperature("2026-10-27", prices=p, cfg=CFG)["parts"]["vix"]["peak"] is None


def test_change_text_and_delta():
    assert change_text(3) == "어제보다 3도 ↑"
    assert change_text(-2) == "어제보다 2도 ↓"
    assert change_text(0) == "어제와 같음"
    assert change_text(None) is None
    p = make_prices()
    r = get_temperature("2026-10-27", prices=p, cfg=CFG)
    prev = get_temperature("2026-10-26", prices=p, cfg=CFG)
    assert r["prev_temp"] == prev["temp"] and r["delta"] == r["temp"] - prev["temp"]


def test_interpretation_lines():
    idx = pd.bdate_range(end="2026-10-27", periods=260)
    rising = pd.Series(np.linspace(50, 150, 260), index=idx)
    assert get_temperature("2026-10-27", prices={"qqq": rising}, cfg=CFG)["note"] == "장기 흐름 회복"
    assert get_temperature("2026-10-27", prices={"qqq": rising[::-1].set_axis(idx)}, cfg=CFG)["note"] == "장기 흐름 약세"
    # 200일 창 앞부분만 높았다가 내려온 뒤 반등: 150일선 위·200일선 아래
    v = np.concatenate([np.full(110, 300.0), np.full(149, 100.0), [110.0]])
    r = get_temperature("2026-10-27", prices={"qqq": pd.Series(v, index=idx)}, cfg=CFG)
    assert r["parts"]["sma_fast"]["raw"] > 0 > r["parts"]["sma_slow"]["raw"]
    assert r["note"] == "회복 초기(빠른 신호만 켜짐)"


def test_no_future_data():
    p = make_prices(seed=3)
    for d in ("2026-06-01", "2026-08-14", "2026-10-26"):
        ts = pd.Timestamp(d)
        full = get_temperature(d, prices=p, cfg=CFG)
        cut = get_temperature(d, prices=_cut(p, ts), cfg=CFG)
        assert full == cut


def test_past_date_and_holiday():
    p = make_prices()
    r = get_temperature("2026-09-15", prices=p, cfg=CFG)
    assert r["data_date"] == "2026-09-15" and r["temp"] is not None
    sat = get_temperature("2026-10-24", prices=p, cfg=CFG)  # 토요일 → 그 전 금요일 자료
    assert sat["data_date"] == "2026-10-23"


def test_message_format():
    p = make_prices()
    r = get_temperature("2026-10-27", prices=p, cfg=CFG)
    msg = build_message(r, CFG, datetime(2026, 10, 28).date())
    lines = msg.splitlines()
    assert lines[0] == "[시장 온도계] 10/28(수)"
    assert lines[1].startswith(f"■ 오늘 온도: {r['temp']}도 · {r['zone']}")
    assert lines[-1] == "※ 시장 상태를 보는 온도계입니다. 매매 신호가 아닙니다."
    assert all(not ln.startswith(" ") for ln in lines)
    assert any(ln.startswith("▶ 150일선: ") and "150일선보다" in ln for ln in lines)
    assert any(ln.startswith("▶ 공포·탐욕: ") for ln in lines)


# ── 블록 · 편지 · 같은 값 ───────────────────────────────────────────────────


def test_block_true_false():
    p = make_prices()
    t = get_temperature("2026-10-27", prices=p, cfg=CFG)["temp"]
    fn = blocks.get_block("market_temperature")["fn"]
    assert fn("2026-10-27", t, t, prices=p, cfg=CFG) is True
    assert fn("2026-10-27", t + 1, 100, prices=p, cfg=CFG) is False
    assert fn("2026-10-27", 0, t - 1, prices=p, cfg=CFG) is False
    assert blocks.market_temperature("2026-10-27", 0, 100, prices={"qqq": pd.Series(dtype=float)}, cfg=CFG) is False
    assert blocks.BLOCKS["market_temperature"]["params"] == ["temp_min", "temp_max"]


def test_same_value_telegram_letter_block(tmp_path, monkeypatch):
    """같은 기준일: 텔레그램 메시지(켰을 때) · 보고서 온도(get_temperature 직접 호출) · 블록이 같은 값."""
    from engine import daily
    from notify import letter_plain

    p = make_prices()
    monkeypatch.setenv(CFG["history"]["drive_env"], "x")
    sent = []
    r = tmain.run(now=datetime(2026, 10, 28, 7, 50, tzinfo=KST), prices=p, send=sent.append, history_path=tmp_path / "vb.db",
                  cfg=CFG_ON, schedule_cron="50 22 * * *", real_send=True, on_date="2026-10-27")
    assert r["status"] == "sent"
    temp, zone = r["result"]["temp"], r["result"]["zone"]
    assert f"■ 오늘 온도: {temp}도 · {zone}" in sent[0]
    s = {"as_of": pd.Timestamp("2026-10-27")}
    daily._attach_thermometer(s, temp_fn=lambda d: get_temperature(d, prices=p, cfg=CFG))
    assert letter_plain.market_main_lines(s, {}, letter_plain.load_phrases())[0].startswith(f"시장 온도 {temp}도 · {zone}")
    assert blocks.market_temperature("2026-10-27", temp, temp, prices=p, cfg=CFG) is True


def test_letter_line_omitted_on_failure():
    from engine import daily
    from notify import letter_plain

    s = {"as_of": pd.Timestamp("2026-10-27")}
    daily._attach_thermometer(s)  # 테스트에서는 오늘 자료 받기가 막혀 있음(conftest) → 실패
    assert s["thermometer"] is None and letter_plain.thermometer_result(s) is None
    assert "letter_check" not in s  # 자료 문제 줄도 남기지 않는다
    daily._attach_thermometer(s, temp_fn=lambda d: (_ for _ in ()).throw(ValueError("x")))
    assert letter_plain.thermometer_result(s) is None
    old = get_temperature("2026-10-27", prices=make_prices(end="2026-10-26"), cfg=CFG)  # 기준일 자료가 아직 없음
    assert letter_plain.thermometer_result({**s, "thermometer": old}) is None
    assert letter_plain.thermometer_result({}) is None and letter_plain.thermometer_result({"thermometer": "이상한 값"}) is None


def test_daily_records_history_once_per_base_date(tmp_path):
    """2026-10-29: 독립 메시지를 끈 뒤 기록은 아침 편지 실행(_attach_thermometer record=True)이 남긴다 — 같은 기준일 한 번만,
    기준일 자료가 아직 없거나 record=False면 남기지 않는다."""
    from engine import daily

    db = tmp_path / "vb.db"
    p = make_prices()
    fn = lambda d: get_temperature(d, prices=p, cfg=CFG)  # noqa: E731
    daily._attach_thermometer({"as_of": pd.Timestamp("2026-10-27")}, temp_fn=fn, history_path=db)
    assert history.load(db) == []
    for _ in range(2):
        daily._attach_thermometer({"as_of": pd.Timestamp("2026-10-27")}, temp_fn=fn, record=True, history_path=db)
    rows = history.load(db)
    assert [r["date"] for r in rows] == ["2026-10-27"] and rows[0]["temp"] == str(fn("2026-10-27")["temp"])
    old = make_prices(end="2026-10-26")
    daily._attach_thermometer({"as_of": pd.Timestamp("2026-10-27")}, record=True, history_path=tmp_path / "x.db",
                              temp_fn=lambda d: get_temperature(d, prices=old, cfg=CFG))
    assert history.load(tmp_path / "x.db") == []


def test_standalone_message_off_by_default(tmp_path, monkeypatch):
    """2026-10-29: 독립 메시지 기본 꺼짐(config telegram.enabled false) — 실제 발송 환경이어도 보내지 않고 기록도 없음."""
    monkeypatch.setenv(CFG["history"]["drive_env"], "x")
    assert CFG["telegram"]["enabled"] is False
    r, sent = _run(tmp_path, monkeypatch, datetime(2026, 10, 28, 7, 50, tzinfo=KST), cfg=CFG)
    assert r["status"] == "disabled" and not sent and not (tmp_path / "vb.db").exists()
    assert r["text"].startswith("[시장 온도계] 10/28(수)")


def test_history_in_virtual_books_db(tmp_path):
    db = tmp_path / "virtual_books.db"
    import sqlite3

    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")  # 가상 장부 기존 표는 그대로
    conn.execute("INSERT INTO meta VALUES ('puddle_v32_state', '{}')")
    conn.commit()
    conn.close()
    r = get_temperature("2026-10-27", prices=make_prices(), cfg=CFG)
    history.append(r, db)
    history.append(r, db)  # 같은 기준일은 한 번만
    assert [x["date"] for x in history.load(db)] == ["2026-10-27"] and history.find("2026-10-27", db)["temp"] == str(r["temp"])
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT value FROM meta WHERE key='puddle_v32_state'").fetchone() == ("{}",)
    conn.close()
    assert history.find("2026-10-27", tmp_path / "none.db") is None and not (tmp_path / "none.db").exists()
    assert CFG["history"]["path"] == "data/virtual_books.db" and CFG["history"]["drive_env"] == "GOOGLE_DRIVE_STATE_FILE_ID_VBOOKS"


# ── 실행: 중복 막기 · 요일 · 지연 · 보관 ────────────────────────────────────


def _run(tmp_path, monkeypatch, now, prices=None, cron="50 22 * * *", base="2026-10-27", **kw):
    monkeypatch.setattr(tmain, "base_date_for", lambda now_kst: pd.Timestamp(base).date())
    sent = kw.pop("sent", [])
    r = tmain.run(now=now, prices=prices or make_prices(), send=sent.append, history_path=tmp_path / "vb.db",
                  cfg=kw.pop("cfg", CFG_ON), schedule_cron=cron, real_send=kw.pop("real_send", True), **kw)
    return r, sent


def test_duplicate_send_blocked(tmp_path, monkeypatch):
    monkeypatch.setenv(CFG["history"]["drive_env"], "x")
    now = datetime(2026, 10, 28, 7, 50, tzinfo=KST)
    sent = []
    r1, _ = _run(tmp_path, monkeypatch, now, sent=sent)
    r2, _ = _run(tmp_path, monkeypatch, now, cron="50 23 * * *", sent=sent)
    assert (r1["status"], r2["status"]) == ("sent", "already_sent") and len(sent) == 1
    assert len(history.load(tmp_path / "vb.db")) == 1


@pytest.mark.parametrize("day,skip", [(25, True), (26, True), (27, False), (31, False)])  # 10/25 일 · 10/26 월 · 화 · 토
def test_no_send_on_sunday_monday(tmp_path, monkeypatch, day, skip):
    monkeypatch.setenv(CFG["history"]["drive_env"], "x")
    base = {25: "2026-10-23", 26: "2026-10-23", 27: "2026-10-26", 31: "2026-10-30"}[day]
    idx_end = base
    r, sent = _run(tmp_path, monkeypatch, datetime(2026, 10, day, 7, 50, tzinfo=KST), prices=make_prices(end=idx_end), base=base)
    assert (r["status"] == "skip_weekday") is skip
    assert bool(sent) is not skip


def test_not_ready_defers_then_fails_on_final(tmp_path, monkeypatch):
    monkeypatch.setenv(CFG["history"]["drive_env"], "x")
    now = datetime(2026, 10, 28, 7, 50, tzinfo=KST)
    old = make_prices(end="2026-10-26")
    r, sent = _run(tmp_path, monkeypatch, now, prices=old)
    assert r["status"] == "not_ready" and not sent
    with pytest.raises(RuntimeError):
        _run(tmp_path, monkeypatch, now, prices=old, cron=CFG["schedule"]["final_cron"])


def test_interpretation_confirmed_text():
    assert CFG["interpretation"]["fast_below_slow_above"] == "단기 흐름 약해짐(200일선은 위)"


def test_no_storage_not_sent(tmp_path, monkeypatch):
    monkeypatch.delenv(CFG["history"]["drive_env"], raising=False)
    now = datetime(2026, 10, 28, 7, 50, tzinfo=KST)
    r, sent = _run(tmp_path, monkeypatch, now)
    assert r["status"] == "no_storage" and not sent
    with pytest.raises(RuntimeError):
        _run(tmp_path, monkeypatch, now, cron=None)


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    r, sent = _run(tmp_path, monkeypatch, datetime(2026, 10, 28, 7, 50, tzinfo=KST), real_send=True, dry_run=True)
    assert r["status"] == "dry_run" and not sent and not (tmp_path / "vb.db").exists()
    assert r["text"].startswith("[시장 온도계] 10/28(수)")


def test_main_returns_nonzero_on_failure(monkeypatch):
    monkeypatch.setattr(tmain, "run", lambda **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert tmain.main([]) == 1


def test_workflow_manual_only_and_shares_concurrency():
    """2026-10-29: thermometer.yml 예약 끔(수동 실행만). 예전 예약 값은 config schedule에 남긴다(다시 켤 때 같게)."""
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "thermometer.yml").read_text(encoding="utf-8"))
    on = wf.get("on", wf.get(True))
    assert "schedule" not in on and "workflow_dispatch" in on
    assert CFG["schedule"]["final_cron"] == CFG["schedule"]["crons"][-1]
    daily = yaml.safe_load((ROOT / ".github" / "workflows" / "daily.yml").read_text(encoding="utf-8"))
    assert wf["concurrency"]["group"] == daily["concurrency"]["group"]  # 같은 가상 장부 DB를 쓰므로 한 번에 하나만(2026-10-28)
    steps = {st.get("name"): st for st in wf["jobs"]["thermometer"]["steps"]}
    up = steps["Upload virtual books state to Google Drive"]
    assert "steps.download.outcome == 'success'" in up["if"]  # 못 받은 DB를 올려 장부를 지우지 않게
    dsteps = [st.get("run", "") for st in daily["jobs"][next(iter(daily["jobs"]))]["steps"]]
    assert any("upload --mode vbooks" in r for r in dsteps)  # 편지 실행이 남긴 온도 기록이 드라이브에 올라간다
    text = (ROOT / ".github" / "workflows" / "thermometer.yml").read_text(encoding="utf-8")
    assert "THERMO" not in text and "THERMO" not in (ROOT / ".github" / "workflows" / "daily.yml").read_text(encoding="utf-8")


def test_report_uses_first_recorded_value_without_recompute(tmp_path):
    """2026-10-31: 그 기준일 기록이 있으면 처음 기록된 값 그대로(다시 계산 안 함), 어제 대비는 앞선 기록 행과 비교.
    기록이 없을 때만 계산해서 기록한다."""
    from engine import daily
    from notify import letter_plain

    db = tmp_path / "vb.db"
    p = make_prices()
    first = get_temperature("2026-10-27", prices=p, cfg=CFG)
    history.append({**first, "temp": 48, "zone": "미지근함"}, db)  # 처음 기록된 값(예: 아침 7시 실행)
    prev = get_temperature("2026-10-26", prices=p, cfg=CFG)
    history.append({**prev, "temp": 57, "zone": "미지근함"}, db)

    def boom(d):
        raise AssertionError("기록이 있으면 다시 계산하지 않는다")

    s = {"as_of": pd.Timestamp("2026-10-27")}
    daily._attach_thermometer(s, temp_fn=boom, record=True, history_path=db)
    t = s["thermometer"]
    assert t["temp"] == 48 and t["zone"] == "미지근함" and t["data_date"] == "2026-10-27"
    assert t["prev_temp"] == 57 and t["delta"] == -9 and t["source"] == "history"
    assert t["parts"]["vix"]["raw"] == pytest.approx(first["parts"]["vix"]["raw"], abs=1e-4)
    assert letter_plain.thermometer_result(s) is t
    ph = letter_plain.load_phrases()
    assert letter_plain.market_main_lines(s, {}, ph)[0] == "시장 온도 48도 · 미지근함 (어제보다 9도 ↓)"
    assert letter_plain.caption_line({**s, "letter_today_kst": pd.Timestamp("2026-10-28")}, {}).endswith("· 시장 온도 48도")
    assert len(history.load(db)) == 2  # 새 줄 없음

    # 기록이 없는 기준일: 계산해서 기록 → 다음 실행은 (자료가 바뀌어도) 그 기록 그대로
    s2 = {"as_of": pd.Timestamp("2026-10-28")}
    p2 = make_prices(end="2026-10-28")
    daily._attach_thermometer(s2, temp_fn=lambda d: get_temperature(d, prices=p2, cfg=CFG), record=True, history_path=db)
    rec = history.find("2026-10-28", db)
    assert rec is not None and s2["thermometer"]["temp"] == int(rec["temp"]) and s2["thermometer"]["prev_temp"] == 48
    s3 = {"as_of": pd.Timestamp("2026-10-28")}
    daily._attach_thermometer(s3, temp_fn=boom, record=True, history_path=db)
    assert s3["thermometer"]["temp"] == int(rec["temp"]) and len(history.load(db)) == 3


def test_history_reads_do_not_create_file_or_table(tmp_path):
    import sqlite3

    assert history.find("2026-10-27", tmp_path / "none.db") is None and not (tmp_path / "none.db").exists()
    db = tmp_path / "other.db"
    sqlite3.connect(db).execute("CREATE TABLE t (x)").connection.close()
    assert history.find("2026-10-27", db) is None and history.previous("2026-10-27", db) is None
    tables = {r[0] for r in sqlite3.connect(db).execute("SELECT name FROM sqlite_master")}
    assert tables == {"t"}
