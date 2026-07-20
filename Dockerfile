# AI 균열 안전점검 시스템 — 서버 배포용 컨테이너 (Streamlit · CPU · Solar 임베딩)
# 빌드: docker build -t crack:local .
# - YOLO 추론용 torch/ultralytics는 CPU 휠로 설치 (GPU 학습은 데스크탑 전용)
# - 임베딩은 Upstage Solar API (EMBED_PROVIDER=solar) → 로컬 임베딩 모델 불필요
# - RAG 인덱스는 이미지에 굽지 않고 최초 기동 시 볼륨에 생성 (entrypoint)

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBED_PROVIDER=solar \
    SERVICE_PORT=8501

WORKDIR /app

# OpenCV 런타임 라이브러리(libgl1·glib) + 헬스체크용 curl
# (cv2.imshow는 쓰지 않으므로 GUI 서버 불필요, libgl1만 있으면 동작)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# CPU 전용 torch/torchvision 먼저 설치 → 이후 ultralytics가 CUDA 휠을 끌어오지 않게 고정
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# 앱 소스 + 학습 가중치(best.pt) 복사
COPY . .

RUN chmod +x entrypoint.sh \
    # 비루트 사용자로 실행 (보안)
    && useradd --system --create-home --home-dir /home/crack crack \
    && mkdir -p /app/knowledge/chroma \
    && chown -R crack:crack /app

USER crack

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${SERVICE_PORT}/_stcore/health" || exit 1

# CRLF 방어 위해 bash로 호출 (.gitattributes로 LF 강제도 병행)
ENTRYPOINT ["bash", "entrypoint.sh"]
