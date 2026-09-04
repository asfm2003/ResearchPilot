"""
Stage 7: Hybrid retrieval via Reciprocal Rank Fusion (RRF).
RRF combines RANK POSITIONS (not raw scores) from two different
retrievers, so we don't have to normalize incomparable score scales.
"""
from .vector_store import VectorStore
from .bm25_index import BM25Index

RRF_K = 60  # standard constant from the original RRF paper


def rrf_fuse(dense: list[dict], sparse: list[dict], top_k: int = 8) -> list[dict]:
    scores: dict[str, float] = {}
    lookup: dict[str, dict] = {}
    for rank_list in (dense, sparse):
        for rank, item in enumerate(rank_list):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            lookup[cid] = item
    fused_ids = sorted(scores, key=lambda k: -scores[k])[:top_k]
    return [{**lookup[cid], "rrf_score": scores[cid]} for cid in fused_ids]


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25_index: BM25Index):
        self.vs = vector_store
        self.bm25 = bm25_index

    def retrieve(self, query: str, top_k: int = 8, candidate_pool: int = 30) -> list[dict]:
        dense = self.vs.search(query, top_k=candidate_pool)
        sparse = self.bm25.search(query, top_k=candidate_pool)
        return rrf_fuse(dense, sparse, top_k=top_k)
