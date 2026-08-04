# 천안시 의료취약지역 지도 - FastAPI 버전

기존 React/Next 프로젝트의 지도, 지역 선택, 가중치 비교, 취약순위, 비교 대시보드와 계산 로직은 그대로 유지하고 FastAPI가 화면을 제공하도록 구성했습니다.

## Windows 실행

1. Python 3.10 이상 설치
2. `run.bat` 더블클릭
3. 브라우저에서 `http://127.0.0.1:8000` 접속

## 명령어 실행

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

- 웹 화면: `http://127.0.0.1:8000`
- FastAPI 문서: `http://127.0.0.1:8000/docs`
- 상태 확인: `http://127.0.0.1:8000/api/health`

## 폴더 구조

- `app/main.py`: FastAPI 서버
- `app/static/`: 실행용 웹 화면과 번들
- `frontend_source/`: 업로드된 원본 React/Next 소스 전체
- `build_frontend.js`: 원본 TS/TSX에서 실행 번들을 재생성하는 스크립트

## 참고

원본과 동일하게 지도 배경 타일을 표시하려면 인터넷 연결이 필요합니다. React와 Leaflet도 CDN에서 불러옵니다.
