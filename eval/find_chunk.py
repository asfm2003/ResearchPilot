"""
Quick lookup helper for building eval/questions.json.
Usage: python eval/find_chunk.py "your question"
Prints top 5 candidate chunks so you can pick the right chunk_id by eye.
"""
import sys
from core.vector_store import VectorStore

query = sys.argv[1]
vs = VectorStore()
results = vs.search(query, top_k=5)

for r in results:
    print(f"\nchunk_id: {r['chunk_id']}")
    print(f"paper:    {r['title']}")
    print(f"page:     {r['page']}")
    print(f"text:     {r['text'][:150]}")