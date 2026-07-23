"""
RAG 지식베이스 구축 (knowledge/build_index.py)
- knowledge/raw/ 의 안전기준 문서(txt)를 청킹 -> 임베딩 -> ChromaDB 저장 (1회 실행)
- 임베딩은 embedder.embed_passages 사용 (provider = solar/bge/mock, config로 선택)
  ※ 문서는 반드시 passage 임베딩. 쿼리(rag.py)는 query 임베딩. 동일 벡터공간.

[2차 MVP 재설계] '어디서 왔는지'를 정확히:
  - 각 문서 상단 @meta ... @endmeta 프런트매터에서 기준명·문서종류·발행처·출처URL·결함태그를 파싱해
    모든 청크 메타데이터에 부착 → 근거 인용 시 파일명이 아니라 실제 출처를 제시.
  - 본문은 문단 단위로 청킹(원문 발췌 보존), 하단 '출처:' 블록은 검색 본문에서 분리(메타로만 보관).

문서 소스(요약 정리 + 1차 출처 링크):
  - 국가법령정보센터: 시설물안전법 시행령/시행규칙
  - 국토안전관리원: 안전점검·정밀안전진단 세부지침(상태평가등급)
  - 콘크리트구조 설계기준(허용균열폭 0.3mm 분기) 등
"""
import os
import sys
import glob
import re

# 스크립트를 직접 실행(python knowledge/build_index.py)해도 루트 모듈(config)과
# 패키지(pipeline)를 찾도록 프로젝트 루트를 경로에 추가.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from pipeline import embedder

RAW_DIR = os.path.join(config.BASE_DIR, "knowledge", "raw")

_META_RE = re.compile(r"@meta\s*(.*?)\s*@endmeta", re.DOTALL)
# 프런트매터 키 → Evidence/메타 필드
_META_KEYS = {
    "기준명": "title", "문서종류": "doc_type", "발행처": "publisher",
    "출처": "url", "결함유형": "defect",
}


def parse_front_matter(text):
    """@meta ... @endmeta 블록을 파싱해 (메타dict, 프런트매터 제거된 본문) 반환.
    프런트매터가 없으면 빈 메타 + 원문 그대로(하위호환).
    """
    meta = {"title": "", "doc_type": "", "publisher": "", "url": "", "defect": ""}
    m = _META_RE.search(text)
    if not m:
        return meta, text
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        field = _META_KEYS.get(k.strip())
        if field:
            meta[field] = v.strip()
    body = text[:m.start()] + text[m.end():]
    return meta, body


def split_body(body):
    """본문에서 하단 '출처:' 블록을 분리(검색 본문 오염 방지). (본문, 출처텍스트) 반환."""
    idx = body.rfind("\n출처:")
    if idx == -1:
        idx = body.rfind("출처:")
    if idx != -1:
        return body[:idx].strip(), body[idx:].strip()
    return body.strip(), ""


def chunk(text, size=400, overlap=80):
    """문단 우선 청킹. 문단(빈 줄) 단위로 나누되, 너무 길면 size/overlap 슬라이딩으로 세분.
    원문 발췌(문단)를 최대한 보존해 인용 신뢰도를 높인다.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out = []
    for p in paras:
        p = " ".join(p.split())
        if len(p) <= size:
            out.append(p)
        else:
            i = 0
            while i < len(p):
                out.append(p[i:i + size])
                i += size - overlap
    return out


def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.txt")))
    if not files:
        print(f"[안내] {RAW_DIR} 에 안전기준 txt 문서를 넣은 뒤 다시 실행하세요.")
        return

    import chromadb

    os.makedirs(config.CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    # 재구축: 스키마(메타)가 바뀌므로 기존 컬렉션을 지우고 새로 만든다.
    try:
        client.delete_collection("safety_standards")
    except Exception:
        pass
    # 거리 메트릭은 코사인. ChromaDB 기본은 L2(제곱 유클리드)라 정규화 안 된 Solar 벡터에서
    # 1-distance 가 음수로 나와 RAG_MATCH_MIN_SCORE(코사인 관측 기준) 게이팅이 무의미해진다.
    col = client.get_or_create_collection(
        "safety_standards", metadata={"hnsw:space": "cosine"})

    docs, metas, ids, idx = [], [], [], 0
    no_meta = []
    for f in files:
        src = os.path.basename(f)
        raw = open(f, encoding="utf-8").read()
        meta, body = parse_front_matter(raw)
        body, _cite = split_body(body)
        if not meta["title"]:
            no_meta.append(src)
        base_meta = {"source": src, **meta}
        for c in chunk(body):
            docs.append(c)
            metas.append(dict(base_meta))
            ids.append(f"c{idx}")
            idx += 1

    print(f"[임베딩] provider={config.EMBED_PROVIDER} · {len(files)}개 문서 → {len(docs)}개 청크 임베딩 중...")
    embs = embedder.embed_passages(docs)      # 문서 = passage 임베딩
    col.add(documents=docs, embeddings=embs, metadatas=metas, ids=ids)
    print(f"[완료] {len(files)}개 문서 → {len(docs)}개 청크 인덱싱 (ChromaDB: {config.CHROMA_DIR})")
    # 인덱스 버전 마커 기록 → 배포 시 repo의 KB_VERSION 과 비교해 불일치면 entrypoint 가 자동 재빌드
    # (옛 인덱스가 볼륨에 남아 스키마/메트릭 수정이 무효화되는 것을 방지)
    _ver_src = os.path.join(config.BASE_DIR, "knowledge", "KB_VERSION")
    try:
        _ver = open(_ver_src, encoding="utf-8").read().strip()
    except Exception:
        _ver = ""
    if _ver:
        with open(os.path.join(config.CHROMA_DIR, ".kb_version"), "w", encoding="utf-8") as vf:
            vf.write(_ver)
        print(f"[버전] 인덱스 버전 마커 기록: {_ver}")
    if no_meta:
        print(f"※ 프런트매터(@meta) 없는 문서(파일명으로만 인용됨): {no_meta}")


if __name__ == "__main__":
    main()
