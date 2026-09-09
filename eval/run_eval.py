"""
Stage 18: Evaluation harness.
Runs every question in questions.json through three retrieval variants
and reports Recall@K for each, plus abstention accuracy separately.
"""
import json
from core.vector_store import VectorStore
from core.bm25_index import BM25Index
from core.retriever import HybridRetriever
from core.reranker import Reranker
from core.generator import answer_question

TOP_K = 6


def load_questions(path="eval/questions.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_at_k(retrieved_ids: list[str], expected_ids: list[str]) -> int:
    """1 if ANY expected chunk appears in retrieved, else 0."""
    return int(any(eid in retrieved_ids for eid in expected_ids))


def build_indexes():
    vs = VectorStore()
    raw = vs.collection.get(include=["documents", "metadatas"])
    all_chunks = [{"chunk_id": cid, "text": doc, **meta}
                  for cid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])]
    bm25 = BM25Index()
    bm25.build(all_chunks)
    return vs, bm25


def run_eval():
    questions = load_questions()
    vs, bm25 = build_indexes()
    retriever = HybridRetriever(vs, bm25)
    reranker = Reranker()

    retrieval_qs = [q for q in questions if q["expected_chunk_ids"]]
    abstention_qs = [q for q in questions if not q["expected_chunk_ids"]]

    scores = {"dense": [], "hybrid": [], "reranked": []}

    for q in retrieval_qs:
        question, expected = q["question"], q["expected_chunk_ids"]

        dense_results = vs.search(question, top_k=TOP_K)
        dense_ids = [r["chunk_id"] for r in dense_results]
        scores["dense"].append(recall_at_k(dense_ids, expected))

        hybrid_results = retriever.retrieve(question, top_k=TOP_K)
        hybrid_ids = [r["chunk_id"] for r in hybrid_results]
        scores["hybrid"].append(recall_at_k(hybrid_ids, expected))

        candidates = retriever.retrieve(question, top_k=20)
        reranked_results = reranker.rerank(question, candidates, top_k=TOP_K)
        reranked_ids = [r["chunk_id"] for r in reranked_results]
        scores["reranked"].append(recall_at_k(reranked_ids, expected))

        print(f"\n[{question}]")
        print(f"  expected: {expected}")
        print(f"  dense   : {'HIT' if scores['dense'][-1] else 'MISS'} -> {dense_ids}")
        print(f"  hybrid  : {'HIT' if scores['hybrid'][-1] else 'MISS'} -> {hybrid_ids}")
        print(f"  reranked: {'HIT' if scores['reranked'][-1] else 'MISS'} -> {reranked_ids}")

    print(f"\n=== RETRIEVAL: Recall@{TOP_K} over {len(retrieval_qs)} questions ===")
    for method, results in scores.items():
        pct = 100 * sum(results) / len(results)
        print(f"{method:10s}: {pct:.1f}%  ({sum(results)}/{len(results)})")

    print(f"\n=== ABSTENTION: {len(abstention_qs)} questions ===")
    correct_abstentions = 0
    for q in abstention_qs:
        candidates = retriever.retrieve(q["question"], top_k=20)
        top_chunks = reranker.rerank(q["question"], candidates, top_k=TOP_K)
        result = answer_question(q["question"], top_chunks)
        abstained = "couldn't find sufficient evidence" in result["answer"].lower()
        correct_abstentions += int(abstained)
        print(f"  [{'PASS' if abstained else 'FAIL'}] {q['question']}")

    if abstention_qs:
        pct = 100 * correct_abstentions / len(abstention_qs)
        print(f"\nAbstention accuracy: {pct:.1f}%  ({correct_abstentions}/{len(abstention_qs)})")


if __name__ == "__main__":
    run_eval()