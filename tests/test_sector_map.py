"""core/sector_map.py 테스트 (네트워크 없음)."""

from pathlib import Path

import pandas as pd
import pytest

from core import sector_map as sm

MAP_FILE = Path(__file__).resolve().parents[1] / "configs" / "sector_pullback_sic_map.csv"


@pytest.fixture(scope="module")
def rules():
    return pd.read_csv(MAP_FILE, comment="#")


def test_committed_map_is_valid(rules):
    sm.validate_rules(rules)
    assert set(rules["sector"]) == set(sm.SECTORS)


@pytest.mark.parametrize("sic,sector", [
    (3571, "기술"),       # 컴퓨터 (산업 기계 3500~3599 안의 더 좁은 범위)
    (3674, "기술"),       # 반도체
    (7372, "기술"),       # 소프트웨어 (사업 서비스 안)
    (7389, "산업"),       # 기타 사업 서비스
    (2834, "헬스"),       # 의약품 (화학 안)
    (2821, "소재"),       # 플라스틱 원료
    (3711, "소비재량"),   # 자동차 (운송 장비 안)
    (3721, "산업"),       # 항공기
    (4813, "통신"),
    (4724, "소비재량"),   # 여행사
    (4731, "산업"),       # 화물 운송 주선 (0단계 뒤 사용자 결정 1)
    (4700, "산업"),
    (4911, "유틸리티"),
    (4953, "산업"),       # 폐기물
    (5331, "필수소비"),
    (5961, "소비재량"),
    (5122, "헬스"),
    (6022, "금융"),
    (6798, "부동산"),     # 리츠 (지주·투자 안)
    (1311, "에너지"),
    (8731, "헬스"),
    ("3674", "기술"),
])
def test_known_codes(rules, sic, sector):
    assert sm.sic_to_sector(sic, rules) == sector


@pytest.mark.parametrize("sic", [None, "", "abc", 9995, 9999, 50])
def test_unmapped(rules, sic):
    assert sm.sic_to_sector(sic, rules) is None


def test_narrowest_wins_regardless_of_order():
    r = pd.DataFrame({"sic_from": [3670, 3000], "sic_to": [3679, 3999], "sector": ["기술", "산업"]})
    assert sm.sic_to_sector(3674, r) == "기술"
    assert sm.sic_to_sector(3674, r.iloc[::-1]) == "기술"


def test_validate_rejects_unknown_sector_and_duplicates():
    with pytest.raises(ValueError):
        sm.validate_rules(pd.DataFrame({"sic_from": [1], "sic_to": [2], "sector": ["IT"]}))
    with pytest.raises(ValueError):
        sm.validate_rules(pd.DataFrame({"sic_from": [1, 1], "sic_to": [2, 2], "sector": ["기술", "산업"]}))


def test_monthly_counts_and_summary():
    m1, m2 = pd.Timestamp("2012-01-31"), pd.Timestamp("2012-02-29")
    members = {m1: {"A", "B", "C"}, m2: {"A", "D"}}
    sector_of = {"A": "기술", "B": "기술", "C": None}
    c = sm.monthly_sector_counts(members, sector_of)
    assert c.loc[m1, "기술"] == 2 and c.loc[m1, sm.UNMAPPED] == 1
    assert c.loc[m2, "기술"] == 1 and c.loc[m2, sm.UNMAPPED] == 1  # D는 sector_of에 없음 → 실패
    s = sm.summarize_counts(c, 3)
    assert s.loc["기술", "월 평균"] == 1.5 and s.loc["기술", "3종목 미만 달"] == 2
