"""
Stage 7: Hybrid retrieval via Weighted Reciprocal Rank Fusion.
Dense and BM25 rankings are fused by POSITION (not raw score, since
those scales aren't comparable) - but weighted, since BM25 can
introduce keyword-overlap noise that outranks a genuinely correct
dense hit (see eval/README finding).
"""
from .vector_store import VectorStore
from .bm25_index import BM25Index

RRF_K = 60
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3


def rrf_fuse(dense: list[dict], sparse: list[dict], top_k: int = 8) -> list[dict]:
    scores: dict[str, float] = {}
    lookup: dict[str, dict] = {}

    for rank, item in enumerate(dense):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + DENSE_WEIGHT / (RRF_K + rank + 1)
        lookup[cid] = item

    for rank, item in enumerate(sparse):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + SPARSE_WEIGHT / (RRF_K + rank + 1)
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