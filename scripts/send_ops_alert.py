"""예약 작업(scripts/run_daily.ps1)이 실패했을 때 텔레그램 오류 알림을 보내는 진입점.

notify.telegram.notify_ops_error를 그대로 부른다 — 새 발송 로직을 만들지 않는다.
.env 값은 여기서도 절대 출력하지 않는다.

실행: python scripts/send_ops_alert.py "짧은 오류 메시지"
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):  # 윈도우 콘솔 cp949 UnicodeEncodeError 방지
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from notify.telegram import notify_ops_error  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        print('사용법: python scripts/send_ops_alert.py "오류 메시지"')
        sys.exit(2)
    ok = notify_ops_error(sys.argv[1])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
