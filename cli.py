"""
Fastest way to test the pipeline end-to-end without a web server.
Usage:
    python pipeline.py          # ingest papers once
    python cli.py "What datasets were used?"
"""
import sys
from core.vector_store import VectorStore
from core.bm25_index import BM25Index
from core.retriever import HybridRetriever
from core.generator import answer_question
from pipeline import ingest_all
from core.reranker import Reranker


'''def main():
    if len(sys.argv) < 2:
        print('Usage: python cli.py "your question"')
        return

    question = sys.argv[1]

    vs = VectorStore()
    if vs.count() == 0:
        print("Index empty - ingesting papers now...")
        chunks = ingest_all()
        vs.add_chunks(chunks)
    else:
        chunks = vs.collection.get()["metadatas"]  # not used further; BM25 needs full chunk text
        chunks = None

    bm25 = BM25Index()
    if chunks is None:
        # rebuild BM25 from Chroma's stored docs (simplest path for day-1 CLI)
        raw = vs.collection.get(include=["documents", "metadatas"])
        chunks = [{"chunk_id": cid, "text": doc, **meta}
                  for cid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])]
    bm25.build(chunks)

    retriever = HybridRetriever(vs, bm25)
    top_chunks = retriever.retrieve(question, top_k=6)

    result = answer_question(question, top_chunks)
    print("\n=== ANSWER ===")
    print(result["answer"])
    print("\n=== SOURCES ===")
    for s in result["sources"]:
        print(f"[{s['ref']}] {s['paper']} | {s['section']} | p.{s['page']}")


if __name__ == "__main__":
    main() '''

def main():
    if len(sys.argv) < 2:
        print('Usage: python cli.py "your question"')
        return

    question = sys.argv[1]

    vs = VectorStore()
    if vs.count() == 0:
        print("Index empty - ingesting papers now...")
        chunks = ingest_all()
        vs.add_chunks(chunks)

    raw = vs.collection.get(include=["documents", "metadatas"])
    all_chunks = [{"chunk_id": cid, "text": doc, **meta}
                  for cid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])]

    bm25 = BM25Index()
    bm25.build(all_chunks)

    retriever = HybridRetriever(vs, bm25)
    candidates = retriever.retrieve(question, top_k=20)   # widen the pool

    reranker = Reranker()
    top_chunks = reranker.rerank(question, candidates, top_k=6)   # then narrow

    result = answer_question(question, top_chunks)
    print("\n=== ANSWER ===")
    print(result["answer"])
    print("\n=== SOURCES ===")
    for s in result["sources"]:
        print(f"[{s['ref']}] {s['paper']} | {s['section']} | p.{s['page']}")

if __name__ == "__main__":
    main()