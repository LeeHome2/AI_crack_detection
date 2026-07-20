"""
RAG 지식베이스 구축 (knowledge/build_index.py) — 스텁
- knowledge/raw/ 의 안전기준 문서(txt)를 청킹 -> 임베딩 -> ChromaDB 저장 (1회 실행)
- 실제 문서 수집 후 완성 예정. 지금은 인터페이스/뼈대만.

문서 소스(예정):
  - 국가법령정보센터: 시설물안전법 시행령/시행규칙
  - 국토안전관리원: 안전점검·정밀안전진단 지침
  - 콘크리트 허용균열폭 기준 (0.3mm 분기)
"""
import os
import glob
import config

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
    from sentence_transformers import SentenceTransformer

    os.makedirs(config.CHROMA_DIR, exist_ok=True)
    embedder = SentenceTransformer(config.EMBED_MODEL)
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    col = client.get_or_create_collection("safety_standards")

    docs, metas, ids, idx = [], [], [], 0
    for f in files:
        src = os.path.basename(f)
        for c in chunk(open(f, encoding="utf-8").read()):
            docs.append(c); metas.append({"source": src}); ids.append(f"c{idx}"); idx += 1

    embs = embedder.encode(docs, show_progress_bar=True).tolist()
    col.add(documents=docs, embeddings=embs, metadatas=metas, ids=ids)
    print(f"[완료] {len(files)}개 문서 -> {len(docs)}개 청크 인덱싱 (ChromaDB: {config.CHROMA_DIR})")


if __name__ == "__main__":
    main()
