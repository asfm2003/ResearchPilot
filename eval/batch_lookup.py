"""
Runs find_chunk-style lookups for a batch of questions at once,
so you don't have to invoke the CLI one question at a time.
"""
from core.vector_store import VectorStore

QUESTIONS = [
    "what accuracy did the alzheimer speech model achieve",
    "real-time inference latency speech diagnosis",
    "transformer attention mechanism regime detection",
    "experimental results in-context learning",
    "what evaluation metrics do the cardiac and alzheimer papers use",
    "which papers use time series data",
    # add your two custom questions here once you've written them:
    # "your harder factual question",
    # "your own pick question",
]

vs = VectorStore()

for q in QUESTIONS:
    print(f"\n{'='*80}\nQUESTION: {q}\n{'='*80}")
    results = vs.search(q, top_k=5)
    for r in results:
        print(f"chunk_id: {r['chunk_id']}")
        print(f"  paper: {r['title']}")
        print(f"  page:  {r['page']}")
        print(f"  text:  {r['text'][:150]}")