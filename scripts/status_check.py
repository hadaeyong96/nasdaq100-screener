"""스크리너 상태 점검 (읽기 전용).

사용
    python scripts/status_check.py              # 기본 점검
    python scripts/status_check.py --download   # 드라이브에서 최신 DB 받아서 점검
    python scripts/status_check.py --tests      # pytest까지
    python scripts/status_check.py --md         # outputs/status_YYYY-MM-DD.md 저장
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "hadaeyong96/nasdaq100-screener"
KST = timezone(timedelta(hours=9))
DBS = {
    "live (state.db)": ROOT / "data" / "state.db",
    "paper (paper_state.db)": ROOT / "data" / "paper_state.db",
    "가상 장부 (virtual_books.db)": ROOT / "data" / "virtual_books.db",
}
ENV_KEYS = [
    "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "TELEGRAM_GROUP_CHAT_ID", "FRED_API_KEY",
    "GOOGLE_SERVICE_ACCOUNT_JSON", "GOOGLE_SHEETS_ID",
    "GOOGLE_DRIVE_STATE_FILE_ID_LIVE", "GOOGLE_DRIVE_STATE_FILE_ID_PAPER",
    "GOOGLE_DRIVE_STATE_FILE_ID_VBOOKS",
]
FORWARD_BOOKS = {  # 앞으로 기록 장부 → 일별 표 (2026-11-13 웅덩이 v3.2·반등 v2 중단 → 실적급등 장부, docs/prereg/pead_watch.md)
    "실적급등 장부 (pead_watch)": "pead_watch_daily",
    "실적급등 포트폴리오 (pead_watch_portfolio)": "pead_watch_portfolio",
}
DATE_COLS = ("date", "base_date", "as_of_date", "fill_date", "last_date", "run_at", "exit_date")

OK, WARN, BAD, INFO = "✅", "⚠️", "❌", "·"
_lines: list[str] = []
_worst = [0]


def out(mark: str, text: str) -> None:
    _worst[0] = max(_worst[0], {WARN: 1, BAD: 2}.get(mark, 0))
    line = f"  {mark} {text}"
    print(line)
    _lines.append(line)


def section(title: str) -> None:
    print(f"\n[{title}]")
    _lines.append(f"\n## {title}")


def sh(*args: str) -> str:
    try:
        return subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                              timeout=60, encoding="utf-8").stdout.strip()
    except Exception:
        return ""


def last_trading_day() -> date | None:
    try:
        sys.path.insert(0, str(ROOT))
        from data.market_calendar import latest_closed_trading_day
        from zoneinfo import ZoneInfo
        return latest_closed_trading_day(datetime.now(ZoneInfo("America/New_York")))
    except Exception:
        return None


def check_git() -> None:
    section("1. 코드 (git)")
    if not (ROOT / ".git").exists():
        out(BAD, "git 저장소가 아님")
        return
    branch = sh("git", "rev-parse", "--abbrev-ref", "HEAD")
    out(INFO, f"브랜치: {branch}")
    out(INFO, "마지막 커밋: " + sh("git", "log", "-1", "--format=%h %ci %s")[:110])
    sh("git", "fetch", "--quiet", "origin")
    counts = sh("git", "rev-list", "--left-right", "--count", f"HEAD...origin/{branch}")
    if counts:
        ahead, behind = (int(x) for x in counts.split())
        if behind:
            out(WARN, f"원격보다 {behind}커밋 뒤처짐 → git pull 필요")
        if ahead:
            out(WARN, f"원격에 안 올린 커밋 {ahead}개 → git push 필요")
        if not ahead and not behind:
            out(OK, "원격(origin)과 같음")
    dirty = [l for l in sh("git", "status", "--porcelain").splitlines() if l.strip()]
    out(WARN if dirty else OK, f"커밋 안 한 변경 {len(dirty)}개" if dirty else "작업 폴더 깨끗함")
    for l in dirty[:8]:
        out(INFO, f"   {l}")


def check_env() -> None:
    section("2. 파이썬 환경")
    v = sys.version_info
    out(OK if (v.major, v.minor) == (3, 11) else WARN,
        f"Python {v.major}.{v.minor}.{v.micro} (Actions는 3.11)")
    from importlib.metadata import version, PackageNotFoundError
    for pkg in ("pandas", "numpy", "yfinance", "pandas_market_calendars", "gspread", "python-dotenv"):
        try:
            ver = version(pkg)
            if pkg == "pandas" and int(ver.split(".")[0]) >= 3:
                out(WARN, f"pandas {ver} — pandas 3에서 테스트 3개 실패. requirements에 pandas<3 고정 권장")
            else:
                out(OK, f"{pkg} {ver}")
        except PackageNotFoundError:
            out(BAD, f"{pkg} 미설치 → pip install -r requirements.txt")


def check_secrets() -> None:
    section("3. 비밀값 (.env, 값은 출력 안 함)")
    env_file = ROOT / ".env"
    filled = {k: bool(os.environ.get(k, "").strip()) for k in ENV_KEYS}
    if env_file.exists():
        for raw in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" in raw and not raw.lstrip().startswith("#"):
                k, _, val = raw.partition("=")
                if k.strip() in filled:
                    filled[k.strip()] = filled[k.strip()] or bool(val.strip().strip('"\''))
        out(OK, ".env 파일 있음")
    else:
        out(WARN, ".env 파일 없음 (로컬 발송·드라이브 동기화 불가, Actions는 Secrets 사용)")
    for k, ok in filled.items():
        out(OK if ok else INFO, f"{k}: {'채워짐' if ok else '비어 있음'}")


def check_config() -> None:
    section("4. 설정 (config.yaml)")
    try:
        import yaml
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    except Exception as e:
        out(BAD, f"config.yaml 읽기 실패: {e}")
        return
    out(INFO, f"운용 모드: {cfg.get('mode')}")
    on, off, shadow = [], [], []

    def walk(node, path="", parent_on=True):
        if isinstance(node, dict):
            here = parent_on
            if "enabled" in node and isinstance(node["enabled"], bool):
                name = path or "(최상위)"
                if node["enabled"] and parent_on:
                    on.append(name)
                elif node["enabled"]:
                    shadow.append(name)
                else:
                    off.append(name)
                here = parent_on and node["enabled"]
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else str(k), here)

    walk(cfg)
    out(OK, f"실제로 도는 기능 {len(on)}개: " + ", ".join(on))
    if shadow:
        out(INFO, f"켜져 있지만 상위가 꺼져 안 도는 것 {len(shadow)}개: " + ", ".join(shadow))
    out(INFO, f"꺼짐 {len(off)}개: " + ", ".join(off))


def has_schedule(wf_path: str) -> bool:
    """워크플로 파일 경로(.github/workflows/x.yml) → `on:` 아래 schedule이 있으면 True.
    파일을 못 읽으면 True(예전처럼 오래되면 경고)."""
    try:
        text = (ROOT / wf_path).read_text(encoding="utf-8")
    except Exception:
        return True
    return re.search(r"^\s+schedule:", text, re.MULTILINE) is not None


def check_actions() -> None:
    section("5. 자동 실행 (GitHub Actions)")
    url = f"https://api.github.com/repos/{REPO}/actions/runs?per_page=60"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "status-check"})
    tok = os.environ.get("GITHUB_TOKEN", "").strip()
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            runs = json.load(r)["workflow_runs"]
    except Exception as e:
        out(WARN, f"Actions 조회 실패({type(e).__name__}) — 비공개면 GITHUB_TOKEN 필요, 또는 gh run list 사용")
        return
    by_wf: dict[str, list[dict]] = {}
    for run in runs:
        by_wf.setdefault(run["name"], []).append(run)
    now = datetime.now(timezone.utc)
    for name, rs in by_wf.items():
        last = rs[0]
        t = datetime.fromisoformat(last["created_at"].replace("Z", "+00:00"))
        age_h = (now - t).total_seconds() / 3600
        concl = last["conclusion"] or last["status"]
        recent = rs[:10]
        fails = sum(1 for x in recent if x["conclusion"] == "failure")
        mark = OK if concl == "success" else (INFO if concl in ("skipped", "in_progress", "queued") else BAD)
        scheduled = has_schedule(last.get("path", ""))
        stale_limit = 24 * 8 if "weekly" in name.lower() else 30
        if scheduled and age_h > stale_limit:
            mark = WARN if mark == OK else mark
        out(mark, f"{name}: 마지막 {t.astimezone(KST):%m-%d %H:%M} KST · {concl} · "
                  f"최근 {len(recent)}회 중 실패 {fails}회" + ("" if scheduled else " · 예약 없음(수동 실행만)"))
        if mark == BAD:
            out(INFO, f"   확인: {last['html_url']}")


def check_dbs(download: bool) -> None:
    section("6. 기록 (로컬 DB)")
    if download:
        for mode in ("live", "paper", "vbooks"):
            r = subprocess.run([sys.executable, "scripts/drive_sync.py", "download", "--mode", mode],
                               cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
            out(OK if r.returncode == 0 else WARN, f"드라이브 받기 {mode}: "
                + (r.stdout.strip().splitlines() or [r.stderr.strip()[-120:]])[-1])
    ltd = last_trading_day()
    if ltd:
        out(INFO, f"마지막 마감 거래일(NYSE): {ltd}")
    for label, path in DBS.items():
        if not path.exists():
            out(INFO, f"{label}: 로컬에 없음 (--download로 드라이브에서 받기)")
            continue
        mtime = datetime.fromtimestamp(path.stat().st_mtime, KST)
        out(INFO, f"{label}: {path.stat().st_size/1024:,.0f} KB · 수정 {mtime:%Y-%m-%d %H:%M}")
        try:
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            tables = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            newest = ""
            for t in tables:
                n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                cols = [c[1] for c in con.execute(f'PRAGMA table_info("{t}")')]
                dc = next((c for c in DATE_COLS if c in cols), None)
                last = con.execute(f'SELECT MAX("{dc}") FROM "{t}"').fetchone()[0] if dc and n else None
                if last and dc != "run_at":
                    newest = max(newest, str(last)[:10])
                if n:
                    out(INFO, f"   {t}: {n:,}행" + (f" · 마지막 {str(last)[:16]}" if last else ""))
            if "runs" in tables:
                cols = [c[1] for c in con.execute('PRAGMA table_info("runs")')]
                if "status" in cols:
                    row = con.execute("SELECT base_date, status, error FROM runs ORDER BY run_at DESC LIMIT 1").fetchone()
                    if row:
                        out(OK if row[1] in ("ok", "success", "done") else WARN,
                            f"   마지막 실행: 기준일 {row[0]} · 상태 {row[1]}" + (f" · {str(row[2])[:80]}" if row[2] else ""))
            con.close()
            if ltd and newest:
                lag = (ltd - date.fromisoformat(newest)).days
                if lag <= 0:
                    out(OK, f"   최신 기록 {newest} — 최신 상태")
                elif lag <= 4:
                    out(WARN, f"   최신 기록 {newest} — {lag}일 뒤처짐 (주말·연휴면 정상)")
                else:
                    out(BAD, f"   최신 기록 {newest} — {lag}일 뒤처짐 (자동 실행 확인 또는 --download)")
        except Exception as e:
            out(BAD, f"   DB 읽기 실패: {e}")
    check_forward_books(DBS["가상 장부 (virtual_books.db)"], ltd)


def check_forward_books(path: Path, ltd: date | None) -> None:
    """가상 장부 DB의 앞으로 기록 장부(웅덩이 v3.2·반등 v2) 일별 표 → 마지막 날짜와 마감 거래일 대비 지연."""
    if not path.exists():
        return
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            for label, table in FORWARD_BOOKS.items():
                try:
                    last = con.execute(f'SELECT MAX(date) FROM "{table}"').fetchone()[0]
                except sqlite3.OperationalError:
                    out(WARN, f"   {label} ({table}): 표 없음")
                    continue
                if not last:
                    out(WARN, f"   {label} ({table}): 기록 없음")
                    continue
                lag = (ltd - date.fromisoformat(str(last)[:10])).days if ltd else None
                mark = OK if lag is None or lag <= 0 else (WARN if lag <= 4 else BAD)
                out(mark, f"   {label}: 마지막 {str(last)[:10]}" + (f" ({lag}일 뒤처짐)" if lag and lag > 0 else ""))
        finally:
            con.close()
    except Exception as e:
        out(BAD, f"   앞으로 기록 장부 읽기 실패: {e}")


def check_tests() -> None:
    section("7. 테스트 (pytest)")
    # --basetemp: 기본 임시 폴더(Temp\pytest-of-user)는 이 PC에서 종료 정리 때 WinError 5가 나 출력이 비었다
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", "-W", "ignore",
                        "--basetemp", str(ROOT / ".pytest_tmp")],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    tail = [l for l in r.stdout.splitlines() if l.startswith("FAILED") or " passed" in l or " failed" in l]
    for l in tail[-12:]:
        out(OK if r.returncode == 0 else BAD, l[:140])
    if not tail:  # 결과 줄이 없음 — 수집 오류·플러그인 오류 등은 stderr에만 남는다
        out(BAD, f"pytest 결과 줄 없음 · 종료 코드 {r.returncode}")
        for l in (r.stderr or "").strip().splitlines()[-10:]:
            out(INFO, f"   {l[:160]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="스크리너 상태 점검")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--tests", action="store_true")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args(argv)
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except Exception:
        pass
    now = datetime.now(KST)
    print(f"스크리너 상태 점검 — {now:%Y-%m-%d %H:%M} KST")
    _lines.append(f"# 스크리너 상태 점검 — {now:%Y-%m-%d %H:%M} KST")
    for step in (check_git, check_env, check_secrets, check_config, check_actions):
        try:
            step()
        except Exception as e:
            out(BAD, f"{step.__name__} 오류: {e}")
    check_dbs(a.download)
    if a.tests:
        check_tests()
    verdict = ["PASS (정상)", "WARN (확인 필요)", "FAIL (조치 필요)"][_worst[0]]
    print(f"\n종합: {verdict}")
    _lines.append(f"\n**종합: {verdict}**")
    if a.md:
        p = ROOT / "outputs" / f"status_{now:%Y-%m-%d}.md"
        p.parent.mkdir(exist_ok=True)
        p.write_text("\n".join(_lines) + "\n", encoding="utf-8")
        print(f"저장: {p}")
    return _worst[0]


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())