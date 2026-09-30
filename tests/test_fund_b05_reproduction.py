"""scripts/fund_b05_reproduction.py의 순수 함수 테스트. 네트워크 없음.

회귀 방지: 실제 B0.5 재현 실행(2026-09-30)에서 stage2_diff == 0.0(완전 일치)일 때
`stage2_diff or 999`(파이썬에서 0.0은 falsy) 때문에 2단계가 완전히 일치했는데도
1단계가 "가장 가깝다"로 잘못 뽑힌 버그가 실제로 발생했다 — pick_closest_stage로
분리해 고쳤다.
"""

from __future__ import annotations

from scripts.fund_b05_reproduction import pick_closest_stage


def test_pick_closest_stage_prefers_exact_match_of_zero_not_missing():
    """실제로 있었던 버그: stage2_diff == 0.0이면 1단계가 아니라 2단계를 골라야 한다."""
    assert pick_closest_stage(stage1_diff=-0.43, stage2_diff=0.0) == "2단계"


def test_pick_closest_stage_picks_smaller_absolute_diff():
    assert pick_closest_stage(stage1_diff=-0.43, stage2_diff=0.2) == "2단계"
    assert pick_closest_stage(stage1_diff=0.05, stage2_diff=0.2) == "1단계"


def test_pick_closest_stage_ties_prefer_stage1():
    assert pick_closest_stage(stage1_diff=0.3, stage2_diff=-0.3) == "1단계"


def test_pick_closest_stage_none_stage1_falls_back_to_stage2():
    assert pick_closest_stage(stage1_diff=None, stage2_diff=0.5) == "2단계"


def test_pick_closest_stage_none_stage2_falls_back_to_stage1():
    assert pick_closest_stage(stage1_diff=0.5, stage2_diff=None) == "1단계"
