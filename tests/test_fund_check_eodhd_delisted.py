"""scripts/fund_check_eodhd_delisted.py의 순수 함수 테스트 — 네트워크 없이 돈다."""

from __future__ import annotations

from scripts import fund_check_eodhd_delisted as chk


def test_ticker_variants_includes_dot_and_dash_forms():
    assert chk._ticker_variants("BRK-B") == {"BRK-B", "BRK.B"}
    assert chk._ticker_variants("AAPL") == {"AAPL"}


def test_match_against_delisted_finds_exact_code():
    records = [{"Code": "ALTR", "Name": "Altera", "Type": "Common Stock", "Isin": "US0214411003"}]
    out = chk.match_against_delisted(["ALTR", "MISSING"], records)
    assert out["ALTR"]["Name"] == "Altera"
    assert out["MISSING"] is None


def test_match_against_delisted_matches_dot_dash_variant():
    records = [{"Code": "BRK.B", "Name": "Berkshire Hathaway"}]
    out = chk.match_against_delisted(["BRK-B"], records)
    assert out["BRK-B"]["Name"] == "Berkshire Hathaway"
