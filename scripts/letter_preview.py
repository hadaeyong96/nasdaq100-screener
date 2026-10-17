"""아침 편지 짧은 판 미리보기 (일회성 점검, 2026-10-17) — 실제 발송·DB·시트 쓰기 없음.

오늘 확정 봉으로 engine.daily.run(dry_run=True) → 가상 장부 run(dry_run=True, 임시 DB) → 개인판·단체방판 보고서를
outputs/letter_preview/에 그리고, 접힌 상태에서 보이는 글을 화면에 출력한다.

가상 장부 DB가 없는 PC용 근사: 1~5번은 고정 시작일부터 다시 계산(실제와 같음), 6·7번은 시작일 = 기준일이면 첫날을
새로 계산, 9번(한 달 가속)은 이 미리보기에서만 시작일 고정을 풀어 기준일에 새로 시작한다(DB 없이 고정 시작일이면 멈추게 돼 있어서).

실행: python scripts/letter_preview.py [--scorecard 2026-10]   (--scorecard는 성적표 날 모양만 흉내 냄)
"""

from __future__ import annotations

import argparse
import copy
import html as htmllib
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import daily  # noqa: E402
from engine import virtual_books as vbooks  # noqa: E402

OUT = ROOT / "outputs" / "letter_preview"


def visible_text(html: str) -> str:
    """접기를 모두 닫은 상태에서 보이는 글: 맨 위 칸들 + 접기 제목 줄."""
    body = html[html.index("<header>"):html.index("<script>")]
    body = re.sub(r"<style.*?</style>", "", body, flags=re.S)
    # 접기(<details>)는 제목만 남긴다 — 안쪽 details까지 한 번에 지우려고 바깥부터 짝을 맞춘다
    out, i = [], 0
    while True:
        j = body.find("<details", i)
        if j < 0:
            out.append(body[i:])
            break
        out.append(body[i:j])
        depth, k = 0, j
        while True:
            a, b = body.find("<details", k), body.find("</details>", k)
            if a != -1 and a < b:
                depth, k = depth + 1, a + 8
            else:
                depth, k = depth - 1, b + 10
                if depth == 0:
                    break
        blk = body[j:k]
        m = re.search(r"<summary>(.*?)</summary>", blk, re.S)
        out.append("\n" + (m.group(1) if m else "▶ (접기)") + "\n")
        i = k
    text = re.sub(r"<(br|/div|/h1|/h2|/section|/header|/footer)[^>]*>", "\n", "".join(out))
    text = htmllib.unescape(re.sub(r"<[^>]+>", "", text))
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorecard", default=None, help="성적표 날 모양 흉내(예: 2026-10) — 계산은 하지 않고 접기 자리만")
    ap.add_argument("--dec-date", default=None, help="12월 공제 안내를 이 날짜(예: 2026-12-01)를 기준일로 가정해 계산·출력(시트 읽기만)")
    a = ap.parse_args()
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    cfg = daily.load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    daily.OUTPUT_DIR = OUT  # 미리보기 파일은 여기로만
    summary = daily.run(cfg, "live", do_replay=False, dry_run=True)
    if summary.get("skipped"):
        raise SystemExit("기준일이 이미 처리돼 요약이 비었습니다")
    vcfg = copy.deepcopy(cfg)
    vcfg["virtual_books"]["accel"]["start_date"] = None  # 미리보기 전용(위 설명)
    vcfg["virtual_books"]["accel"]["start_fx"] = cfg["virtual_books"]["accel"]["start_fx"]
    with tempfile.TemporaryDirectory() as td:
        vb = vbooks.run(vcfg, db_path=Path(td) / "vb_preview.db", dry_run=True, briefing=daily._briefing_recs(summary))
    print("가상 장부 경고:", "; ".join(vb.get("warnings", [])) or "없음")
    print("가상 장부 오류:", "; ".join(vb.get("errors", [])) or "없음")
    daily._attach_regime_line(summary, vb, cfg)
    daily._attach_market_glance(summary, cfg)
    daily._attach_dec_tax(summary, cfg)
    if a.scorecard:
        summary["scorecard"] = {"month": a.scorecard, "lines": ["(미리보기 — 성적표 접기 자리)"]}
    daily.render_editions(summary, cfg)
    for label, key in (("개인 채팅판", "report_path"), ("투자클럽 단체방판", "group_report_path")):
        p = Path(summary[key])
        print(f"\n===== {label} ({p.name}) =====")
        print(visible_text(p.read_text(encoding="utf-8")))
    from notify import briefing

    print("\n===== 첨부 설명 =====")
    print("[개인]\n" + briefing.build_report_caption(summary, cfg))
    print("[단체방]\n" + briefing.build_report_caption(summary, cfg, edition="group"))
    if a.dec_date:
        import pandas as pd

        from notify import letter

        fake = {**summary, "as_of": pd.Timestamp(a.dec_date)}
        daily._attach_dec_tax(fake, cfg)  # 기준일만 바꿔 가정(QQQM 종가·환율은 지금 최신 값)
        print(f"\n===== {a.dec_date} 가정 개인판 (할 일 줄 + 12월 접기) =====")
        print(letter.todo_line(fake, cfg))
        if fake.get("dec_tax"):
            print("▶ 올해 세금 없는 매도 한도")
            print("\n".join("  " + x for x in fake["dec_tax"]["lines"]))
        else:
            print("(12월 안내 없음 — 표시일 아님 또는 계산 실패, 위 로그 참고)")
        path = daily.report_html.render_report(fake, cfg, OUT / "dec")
        print(f"(보고서 파일: {path})")


if __name__ == "__main__":
    main()
