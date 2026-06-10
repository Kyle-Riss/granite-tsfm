# 전력 거래량 예측 데모 (FastAPI)

서울 KPX 전력거래량 분석·예측 시연 대시보드.

## 로컬 실행

```bash
cd /path/to/granite-tsfm
TRANSFORMERS_OFFLINE=1 .venv/bin/python -m uvicorn demo.app:app --host 127.0.0.1 --port 8001
```

브라우저: http://localhost:8001

## 외부 공유 (Cloudflare Quick Tunnel)

터미널 2개 필요. **서버를 먼저** 띄운 뒤 터널을 실행하세요.

```bash
# 터미널 1 — 서버
TRANSFORMERS_OFFLINE=1 .venv/bin/python -m uvicorn demo.app:app --host 127.0.0.1 --port 8001

# 터미널 2 — 터널 (학교·공용 와이파이에서는 http2 권장)
cloudflared tunnel --url http://127.0.0.1:8001 --protocol http2
```

출력되는 `https://....trycloudflare.com` URL을 공유. 재실행 시 URL이 바뀝니다.

## 사전 요구사항

- `artifacts/seoul/` — TTM 모델, GRU 체크포인트, parquet (로컬에 있어야 예측 탭 동작)
- `.venv` — `pip install -e ".[notebooks]"` 또는 `pyproject.toml` 의존성 설치

## 2025~2027 시뮬레이션

실측은 2024년까지만 보유. 미래 연도는 **다른 과거 연도의 같은 계절 패턴**을 컨텍스트로 사용:

| 미래 | 컨텍스트 소스 |
|------|----------------|
| 2025 | 2024년 실측 |
| 2026 | 2023년 실측 |
| 2027 | 2022년 실측 |
