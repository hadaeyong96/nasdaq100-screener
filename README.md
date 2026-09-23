# 나스닥 100 MACD 스크리너

나스닥 100 종목의 일봉을 받아 MACD 중심 1:2:6 분할 전략 신호를 판정하고,
텔레그램 일일 브리핑으로 보내는 스크리너. 전략 기준 문서는 `docs/strategy_v3.md`,
작업 원칙은 `CLAUDE.md`를 따른다. 지금은 P2(신호 판정 + 상태 전이 + 수량 계산)
단계까지 구현돼 있다 — 텔레그램 발송·스케줄·구글 드라이브 연동은 아직 없다.

## 실행 방법

```bash
pip install -r requirements.txt
python scripts/p1_report.py          # 지표 점검용 리포트
python -m engine.daily --replay      # 첫 실행: 상태를 되돌려 만들고 오늘 신호 산출
python -m engine.daily               # 이후 매일 실행
```

- `python -m engine.daily`는 `data/state.db`(SQLite)에 종목별 상태를 저장하고,
  `outputs/signals_YYYY-MM-DD.{md,csv}`에 오늘의 매수·매도·손절 신호와 경고를 남긴다.
- 실제 체결가·수량은 `data/fills.csv`(커밋되지 않음, `data/fills.example.csv` 참고)에
  적어 두면 다음 실행 때 추천값을 덮어쓴다.
- `--dry-run`을 붙이면 DB에 쓰지 않고 결과만 확인할 수 있다.

- `config.yaml`을 복사하지 않고 그대로 읽는다. 지표 설정·가정값은 이 파일에서 바꾼다.
- 실행하면 나스닥 100 구성 종목의 일봉을 받아(`data/cache/*.parquet`에 캐시) 지표를
  계산하고, `outputs/rsi_bottom10.{csv,md}`와 `outputs/parity_check.{csv,md}`를 만든다.
- 캐시가 오늘 확정된 거래일까지 있으면 네트워크를 다시 타지 않는다. 강제로 새로
  받으려면 `data/cache/` 안의 해당 종목 parquet 파일을 지운다.
- 테스트는 네트워크 없이 돈다: `pytest`

## `.env` 설정

`.env.example`을 복사해 `.env`로 저장하고 필요한 값만 채운다. `.env`는 커밋되지
않는다(`.gitignore`).

```bash
cp .env.example .env
```

## 학교·회사 네트워크에서 인증서 오류가 날 때

`python scripts/p1_report.py` 또는 `python -m data.prices` 실행 중 아래와 비슷한
오류가 나면:

```
Cookie/crumb fetch failed (CertificateVerifyError), continuing without crumb
$AAPL: possibly delisted; no price data found
```

yfinance가 기본으로 쓰는 `curl_cffi`(브라우저 TLS 지문 위장)가 학교·회사 네트워크의
TLS 검사(프록시)와 충돌하는 것이다. `.env`에 아래 한 줄을 추가하면 일반 `requests`로
우회한다.

```
YF_DISABLE_CURL_CFFI=1
```

그래도 `requests` 요청 자체가 `SSLCertVerificationError: self-signed certificate in
certificate chain`으로 실패하면(구성 종목을 받는 `data/universe.py`의 위키백과 요청
등), 그 네트워크의 프록시 인증서를 파이썬이 신뢰하지 못하는 것이다. 이때는
`pip install pip-system-certs`로 OS(윈도우) 인증서 저장소를 그대로 쓰게 하면 대개
해결된다.

## 구조

`CLAUDE.md`의 폴더 구조 원칙을 따른다: `data/`(입출력), `core/`(순수 지표 함수),
`engine/`·`notify/`·`store/`(다음 단계), `scripts/`(점검 스크립트), `tests/`.
