"""RS26 구역 ③ 0단계 가드 — 봉인 구간 줄은 읽지 않고, 구역 밖 종목은 열지 않는다 (네트워크 없음)."""

from __future__ import annotations

import pytest

from scripts import rs26_zone3_check as z


def _write(path, rows, head=",close,div,split"):
    path.write_text(head + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_stops_reading_at_first_row_after_2021(tmp_path):
    p = _write(tmp_path / "X.csv", ["2011-05-31,1,0,0", "2011-06-01,2,0,0", "2021-12-31,3,0,0", "2022-01-03,999,0,0", "garbage"])
    s, beyond = z.read_closes_guarded("X", p, {"X"})
    assert beyond and list(s.values) == [2.0, 3.0] and s.index.max().year == 2021


def test_tiingo_raw_close_column(tmp_path):
    p = _write(tmp_path / "T.csv", ["2012-01-03,10.5,100,1.0,0.0"], head="date,raw_close,volume,split_factor,div_cash")
    s, beyond = z.read_closes_guarded("T", p, {"T"})
    assert not beyond and s.iloc[0] == 10.5


def test_ticker_outside_zone_is_refused(tmp_path):
    p = _write(tmp_path / "AAPL.csv", ["2012-01-03,1,0,0"])
    with pytest.raises(z.ZoneGuardError):
        z.read_closes_guarded("AAPL", p, {"X"})


def test_unsorted_file_is_refused(tmp_path):
    p = _write(tmp_path / "U.csv", ["2012-01-04,1,0,0", "2012-01-03,1,0,0"])
    with pytest.raises(z.ZoneGuardError):
        z.read_closes_guarded("U", p, {"U"})
