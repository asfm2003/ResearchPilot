"""
Glues Stage 1-7 together. Run once per new batch of papers.
"""
import os
import glob
from ingestion.pdf_parser import extract_paper
from ingestion.chunker import chunk_paper, chunk_to_dict
from core.vector_store import VectorStore
from core.bm25_index import BM25Index

PAPERS_DIR = "data/papers"


def ingest_all(papers_dir: str = PAPERS_DIR) -> list[dict]:
    """Parse + chunk every PDF in the folder. Returns all chunk dicts."""
    all_chunks = []
    pdf_paths = sorted(glob.glob(os.path.join(papers_dir, "*.pdf")))
    if not pdf_paths:
        print(f"No PDFs found in {papers_dir}. Drop some in and re-run.")
        return []

    for path in pdf_paths:
        paper_id = os.path.splitext(os.path.basename(path))[0]
        print(f"Parsing {paper_id}...")
        paper = extract_paper(path, paper_id)
        chunks = [chunk_to_dict(c) for c in chunk_paper(paper)]
        print(f"  -> {len(chunks)} chunks, title: {paper.title}")
        all_chunks.extend(chunks)

    return all_chunks


def build_indexes(chunks: list[dict]) -> tuple[VectorStore, BM25Index]:
    vs = VectorStore()
    vs.add_chunks(chunks)
    bm25 = BM25Index()
    bm25.build(chunks)
    print(f"Indexed {vs.count()} chunks in vector store + BM25.")
    return vs, bm25


if __name__ == "__main__":
    chunks = ingest_all()
    build_indexes(chunks)
