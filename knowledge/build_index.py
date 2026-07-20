"""
RAG 지식베이스 구축 (knowledge/build_index.py)
- knowledge/raw/ 의 안전기준 문서(txt)를 청킹 -> 임베딩 -> ChromaDB 저장 (1회 실행)
- 임베딩은 embedder.embed_passages 사용 (provider = solar/bge/mock, config로 선택)
  ※ 문서는 반드시 passage 임베딩. 쿼리(rag.py)는 query 임베딩. 동일 벡터공간.

문서 소스:
  - 국가법령정보센터: 시설물안전법 시행령/시행규칙
  - 국토안전관리원: 안전점검·정밀안전진단 지침
  - 콘크리트 허용균열폭 기준 (0.3mm 분기)
"""
import os
import sys
import glob
import config

# pipeline 패키지의 embedder 사용 (프로젝트 루트를 경로에 추가)
sys.path.insert(0, config.BASE_DIR)
from pipeline import embedder

RAW_DIR = os.path.join(config.BASE_DIR, "knowledge", "raw")


def chunk(text, size=400, overlap=80):
    """문단 청킹 (size자, overlap)."""
    text = " ".join(text.split())
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def main():
    files = glob.glob(os.path.join(RAW_DIR, "*.txt"))
    if not files:
        print(f"[안내] {RAW_DIR} 에 안전기준 txt 문서를 넣은 뒤 다시 실행하세요.")
        print("       (예: 시설물안전법.txt, 정밀안전진단지침.txt, 허용균열폭.txt)")
        return

    import chromadb

    os.makedirs(config.CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    col = client.get_or_create_collection("safety_standards")

    docs, metas, ids, idx = [], [], [], 0
    for f in files:
        src = os.path.basename(f)
        for c in chunk(open(f, encoding="utf-8").read()):
            docs.append(c); metas.append({"source": src}); ids.append(f"c{idx}"); idx += 1

    print(f"[임베딩] provider={config.EMBED_PROVIDER} · {len(docs)}개 청크 임베딩 중...")
    embs = embedder.embed_passages(docs)      # 문서 = passage 임베딩
    col.add(documents=docs, embeddings=embs, metadatas=metas, ids=ids)
    print(f"[완료] {len(files)}개 문서 -> {len(docs)}개 청크 인덱싱 (ChromaDB: {config.CHROMA_DIR})")


if __name__ == "__main__":
    main()
