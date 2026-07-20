"""
임베딩 제공자 추상화 (embedder.py)
- 인덱스(문서)와 쿼리는 반드시 '같은 벡터공간'이어야 검색이 성립한다.
- provider (config.EMBED_PROVIDER):
    "solar" (기본) — Upstage Solar 임베딩 API. 비대칭 모델:
        문서 = solar-embedding-1-large-passage / 쿼리 = solar-embedding-1-large-query
        (두 모델은 '동일 공간'이라 서로 비교 가능, 4096차원). 로컬 모델·torch 불필요.
    "bge"          — 로컬 BGE-m3 (sentence-transformers). 오프라인·무API. 무겁다(2.3GB).
    "mock"         — 오프라인 코드경로 테스트용 결정적 해싱 임베딩 (품질 X).

핵심: build_index.py 와 rag.py 가 모두 이 모듈만 쓰게 하여 공간 일치를 보장한다.
"""
import hashlib
import config

_bge_model = None


# ────────────────────────────── Solar (Upstage) ──────────────────────────────
def _solar_embed(texts, model):
    import requests
    key = config.UPSTAGE_API_KEY
    if not key:
        raise RuntimeError(
            "UPSTAGE_API_KEY 미설정 — 환경변수로 Solar API 키를 넣거나 EMBED_PROVIDER=bge 사용")
    resp = requests.post(
        config.SOLAR_EMBED_ENDPOINT,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "input": texts},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    # 응답 순서 보장 (index 기준 정렬)
    return [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]


# ────────────────────────────── BGE-m3 (로컬 폴백) ──────────────────────────────
def _bge_embed(texts):
    global _bge_model
    if _bge_model is None:
        from sentence_transformers import SentenceTransformer
        _bge_model = SentenceTransformer(config.EMBED_MODEL)
    return _bge_model.encode(texts).tolist()


# ────────────────────────────── mock (오프라인 테스트) ──────────────────────────────
def _mock_embed(texts):
    """결정적 해싱 임베딩. 문서·쿼리 동일 변환 → 같은 토큰이면 매칭됨(코드경로 검증용)."""
    dim = 64
    out = []
    for t in texts:
        v = [0.0] * dim
        for tok in t.split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % dim] += 1.0
        norm = (sum(x * x for x in v)) ** 0.5 or 1.0
        out.append([x / norm for x in v])
    return out


# ────────────────────────────── 공개 API ──────────────────────────────
def embed_passages(texts):
    """문서(인덱스) 임베딩 — 리스트[list[float]] 반환."""
    p = config.EMBED_PROVIDER
    if p == "solar":
        return _solar_embed(texts, config.SOLAR_EMBED_PASSAGE)
    if p == "bge":
        return _bge_embed(texts)
    return _mock_embed(texts)


def embed_query(text):
    """쿼리 임베딩 — 단일 list[float] 반환."""
    p = config.EMBED_PROVIDER
    if p == "solar":
        return _solar_embed([text], config.SOLAR_EMBED_QUERY)[0]
    if p == "bge":
        return _bge_embed([text])[0]
    return _mock_embed([text])[0]
