"""봉인 구간(2022-01-01 이후) 데이터 접근 차단 (AI 펀드 F2, docs/design/fund_sim.md 2.2 "봉인 구간").

봉인 구간은 최종 검증(F6) 때 딱 한 번만 연다. 그 전까지는 로더가 이 구간 데이터를
절대 조용히 돌려주면 안 된다 — unseal=True를 명시적으로 넘기지 않으면 항상
SealedDataError로 멈춘다. 순수 함수(날짜 비교만) — core/ 규칙대로 네트워크·파일·
현재 시각에 접근하지 않는다.
"""

from __future__ import annotations

from datetime import date


class SealedDataError(RuntimeError):
    """봉인 구간(seal_date 이후) 데이터를 unseal 없이 요청했을 때 낸다."""


def enforce_not_sealed(end: date, seal_date: date | None, unseal: bool = False) -> None:
    """end가 seal_date를 넘으면(그리고 unseal이 아니면) SealedDataError를 낸다.

    입력: end(요청한 데이터 구간의 끝 날짜), seal_date(config.yaml의 backtest.seal_date —
         None이면 이 호출에서는 봉인을 확인하지 않는다, 호출부가 결정), unseal(True면
         봉인 구간이어도 통과시킨다 — F6 최종 검증 전용, 기본은 항상 차단)
    출력: 없음. 통과하면 아무 일도 하지 않는다.
    """
    if seal_date is None or unseal:
        return
    if end > seal_date:
        raise SealedDataError(
            f"요청한 구간 끝({end.isoformat()})이 봉인 기준일({seal_date.isoformat()}) 이후입니다 — "
            "unseal=True 없이는 봉인 구간 데이터를 불러올 수 없습니다."
        )
