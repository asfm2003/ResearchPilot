import os
import shutil
import streamlit as st

from ingestion.pdf_parser import extract_paper
from ingestion.chunker import chunk_paper, chunk_to_dict
from core.vector_store import VectorStore
from core.bm25_index import BM25Index

from core.bm25_index import BM25Index
from core.retriever import HybridRetriever
from core.reranker import Reranker
from core.generator import answer_question

st.set_page_config(page_title="ResearchPilot", layout="wide")
st.title("📄 ResearchPilot")
st.caption("Evidence-grounded research assistant — ask questions, get cited answers.")


@st.cache_resource
def get_vector_store():
    return VectorStore()


vs = get_vector_store()

uploaded_files = st.file_uploader(
    "Upload research papers (PDF)", type="pdf", accept_multiple_files=True
)

if uploaded_files and st.button("Ingest papers"):
    os.makedirs("data/papers", exist_ok=True)
    progress = st.progress(0, text="Starting...")

    for i, file in enumerate(uploaded_files):
        dest = os.path.join("data/papers", file.name)
        with open(dest, "wb") as f:
            shutil.copyfileobj(file, f)

        paper_id = os.path.splitext(file.name)[0]
        progress.progress((i) / len(uploaded_files), text=f"Parsing {paper_id}...")
        paper = extract_paper(dest, paper_id)
        chunks = [chunk_to_dict(c) for c in chunk_paper(paper)]
        vs.add_chunks(chunks)
        progress.progress((i + 1) / len(uploaded_files), text=f"Indexed {paper_id}")

    st.success(f"Ingested {len(uploaded_files)} paper(s). Total chunks in index: {vs.count()}")

st.divider()
st.write(f"**Currently indexed:** {vs.count()} chunks")


@st.cache_resource
def get_reranker():
    return Reranker()


st.divider()
st.subheader("Ask a question")

question = st.text_input("Your question")

if question and st.button("Ask"):
    with st.spinner("Retrieving evidence and generating answer..."):
        raw = vs.collection.get(include=["documents", "metadatas"])
        all_chunks = [{"chunk_id": cid, "text": doc, **meta}
                      for cid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])]

        bm25 = BM25Index()
        bm25.build(all_chunks)
        retriever = HybridRetriever(vs, bm25)
        candidates = retriever.retrieve(question, top_k=20)

        reranker = get_reranker()
        top_chunks = reranker.rerank(question, candidates, top_k=6)

        result = answer_question(question, top_chunks)

    st.markdown("### Answer")
    st.write(result["answer"])

    st.markdown("### Sources")
    for s in result["sources"]:
        st.write(f"**[{s['ref']}]** {s['paper']} — {s['section']}, page {s['page']}")