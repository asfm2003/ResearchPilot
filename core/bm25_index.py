"""
Stage 6: BM25 lexical search - catches exact terms/identifiers
(model names, dataset names) that embeddings sometimes blur.
"""
import re
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self):
        self.chunks: list[dict] = []
        self.bm25: BM25Okapi | None = None

    def build(self, chunks: list[dict]):
        self.chunks = chunks
        tokenized = [_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.chunks, scores), key=lambda x: -x[1])[:top_k]
        return [{**c, "score": float(s)} for c, s in ranked if s > 0]
