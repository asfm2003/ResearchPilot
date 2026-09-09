# ResearchPilot

**An evidence-grounded RAG platform for scientific literature analysis.**
Ask questions across your research papers and get answers with inline
citations pointing to the exact paper, section, and page — grounded
entirely in the provided evidence, with correct refusal when the
evidence isn't there.

Built by Abdullah Sajid.

---

## What it does

- Upload research PDFs (via Streamlit UI or FastAPI)
- Ask natural-language questions: "What datasets were used?",
  "Compare the methodologies in these papers", "What are the
  limitations discussed?"
- Get answers with `[n]`-style citations mapped to real
  paper/section/page metadata
- Correctly abstains ("I couldn't find sufficient evidence...") on
  out-of-scope questions instead of hallucinating from general
  knowledge

## Architecture

```
PDF → pymupdf4llm (page-aware markdown extraction)
    → structure-aware chunker (page + section tagged)
    → MiniLM embeddings (sentence-transformers, local, free)
    → Chroma (persistent local vector store)
    → rank_bm25 (lexical/keyword index)
    → Reciprocal Rank Fusion (hand-implemented, RRF_K=60)
    → cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
    → Gemini (gemini-3.6-flash) with citation-forcing system prompt
    → answer with [n] citations + abstention on insufficient evidence
    → served via FastAPI (/papers/upload, /query, /papers)
      and a Streamlit UI (upload + ask, one page)
```

## Setup

```bash
git clone https://github.com/asfm2003/ResearchPilot.git
cd ResearchPilot
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com),
then create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_key_here
```

Drop PDFs into `data/papers/`, then:
```bash
python pipeline.py              # ingest + index everything
python cli.py "your question"   # terminal Q&A
```

Or run the full apps:
```bash
uvicorn app:app --reload        # API at localhost:8000/docs
streamlit run streamlit_app.py  # UI at localhost:8501
```

## Evaluation

18-question benchmark (14 retrieval, 4 abstention) across three
retrieval configurations:

| Method             | Recall@6 |
|--------------------|----------|
| Dense only         | 92.9%    |
| Hybrid (RRF)       | 78.6%    |
| Hybrid + Reranked  | 78.6%    |

Abstention accuracy: 100% (4/4), including two adversarial cases
designed to be plausible-sounding but genuinely unanswerable from
the corpus (e.g. asking about FDA approval status when no paper
discusses regulatory status).

**Finding:** dense-only retrieval outperformed both hybrid variants
on this corpus. Root cause, confirmed across multiple failing
questions: RRF can rank a chunk that scores moderately in *both*
dense and BM25 above a chunk that scores excellently in one source
but is entirely absent from the other's candidate pool. Weighting
dense higher (0.7/0.3) fixed one specific failure mode (a noisy
BM25 false-positive outranking a correct dense hit) but did not
address this second, more fundamental one. At n=14 this is a
consistent pattern, not noise - but the corpus is small (~190
chunks across 5 papers); RRF's documented advantages typically
emerge at much larger scale, where BM25's exact-term matching
becomes essential for precision (e.g. distinguishing "TabPFN" from
semantically-similar-but-wrong terms) rather than a source of noise.
For this project's scale, dense-only or a higher pre-rerank
candidate pool (top-50+ instead of top-20) would likely outperform
naive RRF - a concrete next experiment, not yet run.

Run the eval yourself:
```bash
python -m eval.run_eval
```

## Known limitations (v2)

- **Chunking is character-based, not sentence-aware** — chunks can
  start/end mid-sentence. Acceptable for LLM consumption, not ideal
  for human readability of intermediate output.
- **Section detection is a keyword heuristic** checking only the first
  120 characters of a chunk — mid-page chunks often default to "Body"
  rather than their true section.
- **Title extraction** is a simple heuristic (first substantial line
  of page 1) and can merge title + author lines on some PDF layouts.
- Filenames become `paper_id`s directly, so inconsistent filenames
  (e.g. browser-appended ` (1)`, ` (2)` on duplicate downloads)
  propagate into chunk IDs and citations.

## Project history

- **v1**: PDF parsing → structure-aware chunking → dense embeddings
  (Chroma) → BM25 → hand-implemented RRF fusion → Gemini generation
  with citation-forcing prompt and abstention → FastAPI wrapper
- **v2**: cross-encoder reranking, Streamlit UI, 10-question Recall@K
  evaluation harness with per-question diagnostics

## Roadmap (not yet built)

1. Structured extraction (datasets/methods/metrics as JSON per paper)
2. Weighted/confidence-aware RRF, or larger pre-rerank candidate pool
3. Semantic Scholar / OpenAlex integration for literature discovery
   beyond uploaded PDFs
4. Larger eval set (50+ questions) with Ragas metrics (faithfulness,
   context precision/recall)
5. Docker + deployment

## Stack

Python · FastAPI · Streamlit · Chroma · sentence-transformers ·
rank_bm25 · Gemini (`gemini-3.6-flash`) · pymupdf4llm