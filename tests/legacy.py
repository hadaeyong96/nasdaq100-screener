"""새 출발(2026-10-22) 전 기능을 시험하는 테스트용: 설정에서 끈 기능을 다시 켠 사본.

가상 장부·계획·체결 시트 읽기·종목 접기·월간 성적표·편지 가상 장부 칸은 코드가 남아 있고(설정으로 끔),
예전 테스트는 그 동작을 그대로 지킨다. 새 기본값(모두 끔)은 tests/test_fresh_start.py가 본다.
"""

from __future__ import annotations

import copy


def legacy_cfg(cfg: dict) -> dict:
    """끈 기능을 다시 켠 깊은 사본."""
    c = copy.deepcopy(cfg)
    live = c.setdefault("live", {})
    live["sheet_plan_fills"] = True
    live.setdefault("auto_plan", {})["enabled"] = True
    rep = c.setdefault("report", {})
    rep["show_stock_folds"] = True
    rep["monthly_scorecard"] = True
    c.setdefault("letter", {})["show_books"] = True
    vb = c.get("virtual_books")
    if vb is not None:
        vb["enabled"] = True
        if "penny" in vb:
            vb["penny"]["enabled"] = True
    return c
