"""구글 드라이브 상태 저장 — 라이브 어드바이저 1단계
(docs/design/live_advisor.md 5번).

GitHub Actions 러너는 실행마다 새 가상머신이라 data/state.db·
data/paper_state.db가 실행 사이에 안 남는다. 서비스 계정은 자체 저장
용량이 없어 새 파일을 만들 수 없으므로, **사용자가 미리** 구글 드라이브에
만들어 서비스 계정에 편집 권한으로 공유해 둔 빈 파일 2개(state.db용,
paper_state.db용)의 내용만 덮어쓴다(Drive API의 update media — 새 파일
생성이 아니다) — 소유자(사용자)의 저장 용량을 쓰므로 서비스 계정 용량
문제가 없다.

실행 순서: 실행 시작 — download_from_drive(file_id, local_path)로 그 파일
ID의 내용을 로컬 DB 경로로 내려받는다(엔진은 지금과 동일하게 로컬 파일로
동작) → 엔진 실행 → 실행 끝 — upload_to_drive(file_id, local_path)로 같은
파일 ID에 덮어쓴다.

인증은 data/sheets.py와 같은 서비스 계정(GOOGLE_SERVICE_ACCOUNT_JSON)을
재사용한다. 인증 정보의 값은 어떤 경우에도 출력·로그하지 않는다
(CLAUDE.md 보안).
"""

from __future__ import annotations

from pathlib import Path

from data.sheets import load_credentials_info

_SCOPES = ["https://www.googleapis.com/auth/drive"]


def get_service():
    """서비스 계정 인증으로 구글 드라이브 API v3 클라이언트를 만든다."""
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build

    info = load_credentials_info()
    creds = Credentials.from_service_account_info(info, scopes=_SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def download_from_drive(file_id: str, local_path: Path, service=None) -> None:
    """file_id의 내용을 local_path로 내려받는다 (실행 시작 시 상태 DB 복원).

    service를 주면(테스트용) 그 서비스 객체를 쓰고, 없으면 get_service()로 새로
    인증한다. 드라이브 파일이 비어 있으면(사용자가 아직 아무 내용도 안 넣은 새
    빈 파일 — 첫 실행) local_path를 만들지 않는다: 그 경우 store.db.connect()가
    그 경로에 새 빈 DB를 만든다.
    """
    if service is None:
        service = get_service()
    local_path = Path(local_path)

    content = service.files().get_media(fileId=file_id).execute()
    if not content:
        return
    if isinstance(content, str):
        content = content.encode("utf-8")

    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(content)


def upload_to_drive(file_id: str, local_path: Path, service=None) -> None:
    """local_path의 내용을 file_id 파일에 덮어쓴다 (update media — 새로 만들기
    아님). local_path가 없으면(예: 이번 실행에서 그 모드의 DB가 아예 안 만들어짐)
    아무것도 하지 않는다.
    """
    from googleapiclient.http import MediaFileUpload

    if service is None:
        service = get_service()
    local_path = Path(local_path)
    if not local_path.exists():
        return

    media = MediaFileUpload(str(local_path), resumable=False)
    service.files().update(fileId=file_id, media_body=media).execute()
