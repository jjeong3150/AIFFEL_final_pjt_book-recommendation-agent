import os
from FlagEmbedding import FlagReranker

DRIVE_CACHE_DIR = "/content/drive/MyDrive/aiffel_final_pjt/models"


class LocalReranker:
    """
    BAAI/bge-reranker-v2-m3 기반 로컬 리랭커.

    Google Drive 캐시 경로가 존재하면 로컬에서 로드,
    없으면 HuggingFace에서 다운로드 후 Drive에 저장.

    Usage:
        reranker = LocalReranker("BAAI/bge-reranker-v2-m3")
        ranked_books = reranker.rerank(query=summary, books=merged_payloads, top_n=10)
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool = True):
        model_dir  = model_name.replace("/", "_")
        cache_path = os.path.join(DRIVE_CACHE_DIR, model_dir)

        # 1차 시도: Drive 캐시에서 로드
        try:
            self.model = FlagReranker(cache_path, use_fp16=use_fp16)
            print(f"[LocalReranker] 캐시 로드: {cache_path}")

        # Drive 미마운트 or 캐시 없음 → HuggingFace 다운로드
        except Exception as e:
            print(f"[LocalReranker] 캐시 로드 실패 ({e}), HuggingFace에서 다운로드")
            self.model = FlagReranker(model_name, use_fp16=use_fp16)

            # Drive에 저장 가능한 경우에만 캐싱 시도
            try:
                os.makedirs(cache_path, exist_ok=True)
                self.model.model.save_pretrained(cache_path)
                self.model.tokenizer.save_pretrained(cache_path)
                print(f"[LocalReranker] 캐시 저장 완료: {cache_path}")
            except Exception as e:
                print(f"[LocalReranker] 캐시 저장 실패 ({e}), Drive 마운트 여부 확인 필요")

    def rerank(self, query: str, books: list, top_n: int = 10) -> list:
        """
        query 기준으로 후보 도서를 재정렬.

        Args:
            query  : 리랭킹 기준 쿼리 (원본 summary 권장 — 전체 맥락 앵커)
            books  : payload 딕셔너리 리스트 {"title", "book_intro", "isbn", ...}
            top_n  : 반환할 상위 도서 수

        Returns:
            재정렬된 payload 딕셔너리 리스트 (상위 top_n개)
        """
        if not books:
            return []

        # (query, book_intro) 쌍 구성 — book_intro 없으면 title로 대체
        pairs = [
            [query, b.get("book_intro") or b.get("title", "")]
            for b in books
        ]

        scores = self.model.compute_score(pairs, normalize=True)  # 0~1 정규화

        ranked = sorted(
            zip(scores, books),
            key=lambda x: x[0],
            reverse=True,
        )

        print("\n[Reranking 결과]")
        for score, b in ranked[:top_n]:
            print(f"  score: {score:.4f} | {b.get('title')}")

        return [b for _, b in ranked[:top_n]]
