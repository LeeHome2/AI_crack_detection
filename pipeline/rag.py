"""
[4] RAG 기준 검색 (rag.py)
- ChromaDB에서 안전기준 근거 문장 top-k 검색. 쿼리 임베딩은 embedder.embed_query 사용
  (인덱스 문서는 embed_passages, 쿼리는 embed_query — 동일 벡터공간 보장)
- 출력은 '점수'가 아니라 '근거 문장' + 실제 출처(기준명·문서종류·발행처·URL)
- [2차 MVP] 복합 결함: 탐지된 결함유형(feat.defects)으로 쿼리를 확장해 결함별 기준을 검색.
- 아직 knowledge/build_index.py 로 인덱스를 안 만들었으면 빈 결과 반환(앱에서 안내)
"""
import os
import config
from pipeline import embedder
from schemas import CrackFeatures, RiskResult, RagResult, Evidence

_collection = None

# 결함유형 → 검색 쿼리에 덧붙일 도메인 키워드(해당 결함의 기준 문서를 끌어오기 위함)
_DEFECT_QUERY = {
    "crack": "콘크리트 균열 허용균열폭 상태평가등급",
    "rebar_exposure": "철근노출 피복 탈락 부식 단면 손실 보수",
    "steel_defect": "강재 손상 부식 상태평가 보수",
    "spalling": "콘크리트 박리 박락 피복 탈락 보수",
    "efflorescence": "백태 누수 수분 침투 방수 상태평가",
    "paint_damage": "도장 손상 방식 코팅 재도장",
}


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


def _build_query(feat: CrackFeatures, risk: RiskResult) -> str:
    """feature/등급/탐지 결함에서 검색 쿼리 생성 (복합 결함 반영)."""
    parts = ["시설물 콘크리트 안전점검 기준"]
    # 균열 채널
    if feat.crack_count:
        parts.append(_DEFECT_QUERY["crack"])
        if feat.avg_width_px:
            parts.append("균열 폭 0.3mm 허용 기준 조치")
    # 복합 결함 채널 — 탐지된 결함유형별 도메인 키워드 추가
    for label in (feat.defects or {}):
        kw = _DEFECT_QUERY.get(label)
        if kw:
            parts.append(kw)
    if risk.grade in ("위험", "긴급"):
        parts.append("정밀안전진단 긴급보수 사용제한")
    return " ".join(parts)


def _format_source(meta: dict) -> str:
    """메타데이터로 표시용 출처 문자열 구성 (파일명 폴백)."""
    title = (meta or {}).get("title", "")
    if not title:
        return (meta or {}).get("source", "미상")
    bits = [b for b in ((meta or {}).get("doc_type", ""),
                        (meta or {}).get("publisher", "")) if b]
    return f"「{title}」" + (f" ({' · '.join(bits)})" if bits else "")


def search(feat: CrackFeatures, risk: RiskResult) -> RagResult:
    col = _load()
    if col is None:
        return RagResult()   # 인덱스 미구축 -> 빈 결과
    q = _build_query(feat, risk)
    q_emb = [embedder.embed_query(q)]   # 쿼리 = query 임베딩 (인덱스 passage와 동일 공간)
    res = col.query(query_embeddings=q_emb, n_results=config.RAG_TOP_K)
    out = RagResult()
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        meta = meta or {}
        out.evidences.append(Evidence(
            text=doc,
            source=_format_source(meta),
            score=round(1 - dist, 3),
            title=meta.get("title", ""),
            doc_type=meta.get("doc_type", ""),
            publisher=meta.get("publisher", ""),
            url=meta.get("url", ""),
            defect=meta.get("defect", "")))
    return out
