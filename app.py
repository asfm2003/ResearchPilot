"""
Stage 19/24 (scoped for 4 days): minimal FastAPI wrapper.
Run: uvicorn app:app --reload
"""
import os
import shutil
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from ingestion.pdf_parser import extract_paper
from ingestion.chunker import chunk_paper, chunk_to_dict
from core.vector_store import VectorStore
from core.bm25_index import BM25Index
from core.retriever import HybridRetriever
from core.generator import answer_question

app = FastAPI(title="ResearchPilot")
vs = VectorStore()
bm25 = BM25Index()
ALL_CHUNKS: list[dict] = []


class Query(BaseModel):
    question: str
    top_k: int = 6


def _rebuild_bm25():
    raw = vs.collection.get(include=["documents", "metadatas"])
    global ALL_CHUNKS
    ALL_CHUNKS = [{"chunk_id": cid, "text": doc, **meta}
                  for cid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])]
    bm25.build(ALL_CHUNKS)


@app.on_event("startup")
def startup():
    if vs.count() > 0:
        _rebuild_bm25()


@app.post("/papers/upload")
async def upload_paper(file: UploadFile = File(...)):
    os.makedirs("data/papers", exist_ok=True)
    dest = os.path.join("data/papers", file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    paper_id = os.path.splitext(file.filename)[0]
    paper = extract_paper(dest, paper_id)
    chunks = [chunk_to_dict(c) for c in chunk_paper(paper)]
    vs.add_chunks(chunks)
    _rebuild_bm25()

    return {"paper_id": paper_id, "title": paper.title, "chunks_indexed": len(chunks)}


@app.post("/query")
async def query(q: Query):
    if vs.count() == 0:
        return {"answer": "No papers indexed yet. Upload some first.", "sources": []}
    retriever = HybridRetriever(vs, bm25)
    top_chunks = retriever.retrieve(q.question, top_k=q.top_k)
    return answer_question(q.question, top_chunks)


@app.get("/papers")
async def list_papers():
    raw = vs.collection.get(include=["metadatas"])
    seen = {}
    for meta in raw["metadatas"]:
        seen[meta["paper_id"]] = meta["title"]
    return [{"paper_id": pid, "title": t} for pid, t in seen.items()]
