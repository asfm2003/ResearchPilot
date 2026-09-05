"""
Stage 8: Cross-encoder reranking.
Takes the ~30 candidates RRF already fused and re-scores them by
actually reading (query, chunk) together instead of comparing vectors.
"""
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self):
        self.model = CrossEncoder(RERANKER_MODEL)

    def rerank(self, query: str, chunks: list[dict], top_k: int = 6) -> list[dict]:
        if not chunks:
            return []

        pairs = [[query, c["text"]] for c in chunks]
        scores = self.model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        ranked = sorted(chunks, key=lambda c: -c["rerank_score"])
        return ranked[:top_k]