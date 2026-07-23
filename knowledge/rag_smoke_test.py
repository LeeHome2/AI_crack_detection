r"""
RAG 스모크 테스트 (knowledge/rag_smoke_test.py)
- build_index.py 로 인덱스를 만든 뒤 실행해, 17문서(복합 결함 포함)가 제대로 검색되는지 확인.
- 결함별 대표 쿼리를 날려 top-k 근거의 '출처(기준명)·유사도'를 출력하고, 기대 문서가 잡히는지 체크.
- Solar/BGE 임베딩이 닿는 환경(로컬/배포)에서 실행. (mock 임베딩이면 랜덤이라 무의미)

실행:
  (venv) D:\crack_detection> python knowledge/build_index.py      # 먼저 인덱스 구축
  (venv) D:\crack_detection> python knowledge/rag_smoke_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from pipeline import embedder, rag

# (쿼리, 기대 결함태그) — 해당 결함 문서가 top-k에 잡히면 통과
CASES = [
    ("철근이 드러나고 녹이 슬어 콘크리트가 떨어져 나감", "rebar_exposure"),
    ("콘크리트 표면이 들뜨고 박락되어 떨어짐", "spalling"),
    ("벽면에 흰색 백태가 생기고 물이 새어 누수 흔적", "efflorescence"),
    ("강재 부재가 부식되어 단면이 손실됨", "steel_defect"),
    ("강교 도장이 벗겨지고 방식 코팅 손상", "paint_damage"),
    ("균열 폭 0.3mm 초과 허용기준 보수", "crack"),
    ("여러 결함이 함께 발생한 복합 열화 종합판정", "general"),
]
TOP_K = 3


def main():
    if not rag.is_ready():
        print("[중단] 인덱스 없음 → 먼저 python knowledge/build_index.py 실행")
        return
    col = rag._load()
    print(f"[정보] provider={config.EMBED_PROVIDER} · 컬렉션 문서수 확인")
    passed = 0
    for q, want in CASES:
        emb = [embedder.embed_query(q)]
        res = col.query(query_embeddings=emb, n_results=TOP_K)
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        titles = [(m or {}).get("title", "?") for m in metas]
        defects = [(m or {}).get("defect", "") for m in metas]
        hit = want in defects
        passed += hit
        print(f"\nQ: {q}")
        print(f"   기대 결함태그: {want}  →  {'✅ 잡힘' if hit else '⚠️ top-k에 없음'}")
        for t, d, dist in zip(titles, defects, dists):
            print(f"   · [{d or '-'}] {t}  (유사도 {round(1 - dist, 3)})")
    print(f"\n===== 결과: {passed}/{len(CASES)} 통과 =====")
    if passed < len(CASES):
        print("※ 미통과 쿼리는 해당 결함 문서의 표현/키워드를 보강하거나 RAG_TOP_K를 늘려 재확인.")


if __name__ == "__main__":
    main()
