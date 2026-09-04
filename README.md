# ResearchPilot — 4-Day Build Plan

Scope for 4 days: **M1+M2+M3 fully, M4 fully, thin slice of M5/M6** —
a real hybrid-RAG research assistant with citations, not a toy.
Skipped for now (add later): reranker model, structured extraction,
literature graph, Ragas eval, Docker. The architecture is already
laid out so you can bolt those on afterward.

## Setup (do this first, ~15 min)
```bash
cd research_pilot
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
mkdir -p data/papers
# drop 5-10 PDFs into data/papers/
```

## Day 1 — Ingestion + chunking (M1+M2)
Files: `ingestion/pdf_parser.py`, `ingestion/chunker.py`
- Run `python ingestion/pdf_parser.py data/papers/<one>.pdf` — confirm
  title + page count extract correctly.
- Read `chunker.py` — understand why we chunk **within page boundaries**
  and tag `paper_id/page/section` on every chunk. This metadata is what
  makes citations possible later; don't skip understanding it.
- **Your task:** tune `CHUNK_SIZE_CHARS` — print a few chunks, check
  they're not cutting sentences mid-word in weird spots.

## Day 2 — Embeddings + vector search + first answer (M3, half of M4)
Files: `core/vector_store.py`, `pipeline.py`, `cli.py`
```bash
python pipeline.py                       # ingest + index everything
python cli.py "What datasets were used?"
```
- You now have working semantic RAG with page-level citations.
- **Your task:** ask 5 real questions about your own papers, read the
  `[n]` citations against the actual PDF pages — check the model isn't
  citing the wrong page. Fix chunking if it is.

## Day 3 — Hybrid retrieval + grounded generation (rest of M4, M5-lite)
Files: `core/bm25_index.py`, `core/retriever.py`, `core/generator.py`
- These are already wired into `cli.py`/`app.py`. Read `retriever.py`
  closely — this is the RRF fusion math from the roadmap doc, implemented
  from scratch (no black-box library).
- **Your task:** compare answers with `HybridRetriever` vs dense-only
  (`vs.search()` alone) on a query with an exact model/dataset name like
  "ADReSSo" or "SPECTER2". Hybrid should win — that's the point of BM25.
- Also test the abstention behavior: ask something not in your papers,
  confirm it says it can't find evidence instead of hallucinating.

## Day 4 — API + polish + writeup (M7-lite)
Files: `app.py`
```bash
uvicorn app:app --reload
# POST /papers/upload  (multipart file)
# POST /query          {"question": "..."}
# GET  /papers
```
- Test all three endpoints with `curl` or the FastAPI `/docs` page.
- Write your CV bullet using the template in the roadmap doc (Section
  "What I want the final CV project to look like").
- Push to GitHub with this README as-is — it doubles as your project log.

## What to build next (post-4-days, in order)
1. Cross-encoder reranker (`sentence-transformers` CrossEncoder,
   `cross-encoder/ms-marco-MiniLM-L-6-v2` — drop-in, no new infra)
2. Structured extraction pass (datasets/methods/metrics as JSON per paper)
3. Small eval set (20 Q&A pairs) + Recall@K, then Ragas
4. Semantic Scholar / OpenAlex for literature discovery beyond uploads

## Architecture (current state)
```
PDF → pymupdf4llm → page-aware chunks → MiniLM embeddings → Chroma
                                       ↘ BM25 (rank_bm25) ↗
                                    RRF fusion → top-k → Claude
                                    → cited, grounded answer
```
