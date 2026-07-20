"""
[4] RAG 기준 검색 (rag.py)
- ChromaDB + 한국어 임베딩에서 안전기준 근거 문장 top-k 검색
- 출력은 '점수'가 아니라 '근거 문장'
- 아직 knowledge/build_index.py 로 인덱스를 안 만들었으면 빈 결과 반환(앱에서 안내)
"""
import os
import config
from schemas import CrackFeatures, RiskResult, RagResult, Evidence

_collection = None
_embedder = None


def is_ready():
    return os.path.isdir(config.CHROMA_DIR) and os.listdir(config.CHROMA_DIR)


def _load():
    global _collection
    if _collection is not None:
        return _collection
    if not is_ready():
        return None
    import chromadb
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    _collection = client.get_or_create_collection("safety_standards")
    return _collection


def _load_embedder():
    """쿼리를 인덱스와 동일한 BGE-m3로 임베딩 (공간 일치 필수)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(config.EMBED_MODEL)
    return _embedder


def _build_query(feat: CrackFeatures, risk: RiskResult) -> str:
    """feature/등급에서 검색 쿼리 생성."""
    parts = ["콘크리트 균열 안전점검 기준"]
    if feat.avg_width_px:
        parts.append("균열 폭 허용 기준 0.3mm 조치")
    if risk.grade in ("위험", "긴급"):
        parts.append("정밀안전진단 긴급보수")
    return " ".join(parts)


def search(feat: CrackFeatures, risk: RiskResult) -> RagResult:
    col = _load()
    if col is None:
        return RagResult()   # 인덱스 미구축 -> 빈 결과
    q = _build_query(feat, risk)
    q_emb = _load_embedder().encode([q]).tolist()   # 인덱스와 동일 임베딩으로 쿼리
    res = col.query(query_embeddings=q_emb, n_results=config.RAG_TOP_K)
    out = RagResult()
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        out.evidences.append(Evidence(
            text=doc,
            source=(meta or {}).get("source", "미상"),
            score=round(1 - dist, 3)))
    return out
