#!/usr/bin/env bash
# 컨테이너 진입점: RAG 인덱스가 없거나 '버전이 다르면' 재빌드 후 Streamlit 기동.
# 버전 마커(knowledge/KB_VERSION ↔ 볼륨의 .kb_version)로 스키마/메트릭/문서 변경을 감지.
# → 코사인 전환·문서 확충 등 지식베이스 변경이 재배포에서 자동 반영됨(옛 인덱스 잔존 방지).
set -euo pipefail

CHROMA_DIR="knowledge/chroma"
REPO_VER="$(cat knowledge/KB_VERSION 2>/dev/null || echo '?')"
IDX_VER="$(cat "$CHROMA_DIR/.kb_version" 2>/dev/null || echo '')"

if [ ! -d "$CHROMA_DIR" ] || [ -z "$(ls -A "$CHROMA_DIR" 2>/dev/null)" ] || [ "$REPO_VER" != "$IDX_VER" ]; then
  echo "[entrypoint] 인덱스 재빌드 필요 (repo='$REPO_VER' vs 인덱스='$IDX_VER') → build_index (provider=${EMBED_PROVIDER:-solar})"
  # 인덱싱 실패(예: UPSTAGE_API_KEY 미설정)해도 앱은 RAG 없이 기동되도록 방어
  if python knowledge/build_index.py; then
    echo "[entrypoint] 인덱스 재빌드 완료 (버전 $REPO_VER)"
  else
    echo "[entrypoint][warn] 인덱스 빌드 실패 — RAG 없이 기동 (앱에서 안내 표시)"
  fi
else
  echo "[entrypoint] 기존 RAG 인덱스 재사용 (버전 $REPO_VER): $CHROMA_DIR"
fi

exec streamlit run app.py \
  --server.port "${SERVICE_PORT:-8501}" \
  --server.address 0.0.0.0 \
  --server.headless true
