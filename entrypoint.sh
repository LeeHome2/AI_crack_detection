#!/usr/bin/env bash
# 컨테이너 진입점: RAG 인덱스가 없으면 최초 1회 생성 후 Streamlit 기동.
set -euo pipefail

CHROMA_DIR="knowledge/chroma"

if [ ! -d "$CHROMA_DIR" ] || [ -z "$(ls -A "$CHROMA_DIR" 2>/dev/null)" ]; then
  echo "[entrypoint] RAG 인덱스 없음 → build_index 실행 (provider=${EMBED_PROVIDER:-solar})"
  # 인덱싱 실패(예: UPSTAGE_API_KEY 미설정)해도 앱은 RAG 없이 기동되도록 방어
  if python knowledge/build_index.py; then
    echo "[entrypoint] 인덱스 생성 완료"
  else
    echo "[entrypoint][warn] 인덱스 생성 실패 — RAG 없이 기동 (앱에서 안내 표시)"
  fi
else
  echo "[entrypoint] 기존 RAG 인덱스 재사용: $CHROMA_DIR"
fi

exec streamlit run app.py \
  --server.port "${SERVICE_PORT:-8501}" \
  --server.address 0.0.0.0 \
  --server.headless true
