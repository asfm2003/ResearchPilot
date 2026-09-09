import os
import re
import shutil
import streamlit as st

from ingestion.pdf_parser import extract_paper
from ingestion.chunker import chunk_paper, chunk_to_dict
from core.vector_store import VectorStore
from core.bm25_index import BM25Index
from core.retriever import HybridRetriever
from core.reranker import Reranker
from core.generator import answer_question


def clean_text(text: str) -> str:
    """Strip stray HTML/XML tags that sometimes leak into extracted
    titles or author strings (e.g. <sup>1</sup> affiliation markers
    the parser picked up), and collapse repeated whitespace."""
    if not text:
        return text
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Page setup ────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchPilot",
    page_icon="📄",  # Streamlit requires this for the browser tab; not shown in-app
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Icons (inline SVG, single-weight line icons — no emoji) ───────
ICON_LIBRARY = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>"""
ICON_UPLOAD = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12"/><path d="M7 8l5-5 5 5"/><path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/></svg>"""
ICON_DOC = """<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M9 13h6M9 17h6M9 9h1"/></svg>"""
ICON_ARROW = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/><path d="M13 6l6 6-6 6"/></svg>"""

# ── Theme ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600&display=swap');

:root {
    --paper:      #FDFCF9;
    --paper-dim:  #F7F5EF;
    --ink:        #1C1F26;
    --ink-soft:   #514C42;
    --muted:      #8A8375;
    --line:       #E5E1D6;
    --evidence:   #2B5D4F;   /* deep archival green — primary */
    --evidence-dark: #1F463B;
    --citation:   #B75A34;   /* burnt sienna — citation / accent */
    --citation-bg:#FBEEE7;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Force every native text element to the ink palette — without this,
   Streamlit's dark-theme defaults (white text) go invisible on our
   white background wherever we haven't explicitly restyled something. */
.stApp, .stApp p, .stApp span, .stApp label, .stApp li,
.stApp h1, .stApp h2, .stApp h3,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
[data-testid="stWidgetLabel"] p,
details summary, details summary span, details p, details li {
    color: var(--ink) !important;
}
[data-testid="stCaptionContainer"] p { color: var(--muted) !important; font-size: 0.85rem !important; }

.stApp { background: var(--paper); }
.main .block-container { padding-top: 3rem; max-width: 880px; }

/* Kill Streamlit chrome noise */
#MainMenu, footer, header { visibility: hidden; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--paper-dim);
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .block-container { padding-top: 2.5rem; }

.side-heading {
    display: flex; align-items: center; gap: 8px;
    font-family: 'Inter', sans-serif; font-weight: 600; font-size: 0.95rem;
    color: var(--ink); margin-bottom: 14px;
}
.side-heading svg { color: var(--evidence); flex-shrink: 0; }

.stat-row { display: flex; gap: 10px; }
.stat-box {
    flex: 1; background: var(--paper); border: 1px solid var(--line);
    border-radius: 4px; padding: 14px 10px; text-align: center;
}
.stat-box .num {
    font-family: 'Source Serif 4', serif; font-size: 1.7rem; font-weight: 600;
    color: var(--evidence); line-height: 1;
}
.stat-box .label {
    font-size: 0.72rem; color: var(--muted); margin-top: 4px;
    letter-spacing: 0.02em;
}

/* ---------- Main: hero ---------- */
.hero-row { display: flex; align-items: flex-start; gap: 16px; margin-bottom: 6px; }
.hero-row .icon { color: var(--evidence); margin-top: 6px; flex-shrink: 0; }
.hero-title {
    font-family: 'Source Serif 4', serif; font-weight: 700;
    font-size: 2.6rem; color: var(--ink); line-height: 1.05; margin: 0;
}
.hero-sub {
    font-size: 1rem; color: var(--ink-soft); margin: 10px 0 2.2rem 50px;
    max-width: 60ch; line-height: 1.5;
}

/* ---------- Question input ---------- */
div[data-testid="stTextInput"] input {
    background: var(--paper) !important;
    border: 1.5px solid var(--line) !important;
    border-radius: 6px !important;
    padding: 14px 16px !important;
    font-size: 1rem !important;
    color: var(--ink) !important;
    box-shadow: none !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: var(--evidence) !important;
    box-shadow: 0 0 0 3px rgba(43, 93, 79, 0.12) !important;
}
div[data-testid="stTextInput"] input::placeholder { color: var(--muted); }

/* ---------- Buttons ---------- */
.stButton button {
    background: var(--evidence) !important;
    color: var(--paper) !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.55rem 1.3rem !important;
    font-weight: 500 !important;
    font-size: 0.92rem !important;
    transition: background 0.15s ease;
}
.stButton button:hover { background: var(--evidence-dark) !important; }
.stButton button:focus { box-shadow: 0 0 0 3px rgba(43, 93, 79, 0.25) !important; }

section[data-testid="stSidebar"] .stButton button {
    background: var(--paper) !important;
    color: var(--ink) !important;
    border: 1.5px solid var(--line) !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    border-color: var(--evidence) !important;
    color: var(--evidence) !important;
    background: var(--paper) !important;
}

/* ---------- Answer ---------- */
.answer-box {
    background: var(--paper); border: 1px solid var(--line);
    border-top: 3px solid var(--evidence); border-radius: 8px;
    padding: 28px 32px 22px; margin-top: 4px;
    box-shadow: 0 1px 3px rgba(28, 31, 38, 0.04);
}
.answer-label {
    display: flex; align-items: center; gap: 8px;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.03em;
    color: var(--evidence); text-transform: uppercase; margin-bottom: 12px;
}
.answer-box [data-testid="stMarkdownContainer"] p,
.answer-box [data-testid="stMarkdownContainer"] li {
    font-family: 'Source Serif 4', serif !important; font-size: 1.06rem !important;
    color: var(--ink) !important; line-height: 1.7 !important;
}
.answer-box [data-testid="stMarkdownContainer"] strong { color: var(--evidence-dark) !important; }

/* ---------- Sources ---------- */
.sources-label {
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.03em;
    color: var(--citation); text-transform: uppercase; margin: 28px 0 12px;
}
.source-card {
    display: flex; gap: 12px; align-items: baseline;
    background: var(--paper); border: 1px solid var(--line); border-left: 3px solid var(--citation);
    border-radius: 4px; padding: 12px 16px; margin-bottom: 8px; font-size: 0.92rem;
    color: var(--ink-soft); transition: box-shadow 0.15s ease, border-left-color 0.15s ease;
}
.source-card:hover {
    box-shadow: 0 2px 8px rgba(183, 90, 52, 0.12);
    border-left-color: var(--citation);
}
.source-ref {
    font-family: 'Source Serif 4', serif; font-weight: 700; color: var(--citation);
    flex-shrink: 0;
}

/* ---------- Empty state ---------- */
.empty-hint {
    display: flex; align-items: center; gap: 10px;
    color: var(--muted); font-size: 0.9rem;
    border: 1px dashed var(--line); border-radius: 6px;
    padding: 14px 18px; margin-top: 4px;
}

/* ---------- Divider ---------- */
hr { border-color: var(--line) !important; margin: 1.6rem 0 !important; }

/* ---------- File uploader ---------- */
/* Broad net: target every element whose testid contains "FileUploader" so this
   survives Streamlit DOM/version differences, and force backgrounds white
   at every nesting level (dropzone, inner section, instructions wrapper). */
[data-testid*="FileUploader"],
[data-testid*="FileUploader"] section,
[data-testid*="FileUploader"] div {
    background: var(--paper) !important;
}
div[data-testid="stFileUploaderDropzone"] {
    border: 1.5px dashed var(--evidence) !important;
    border-radius: 8px !important;
    padding: 6px !important;
    transition: border-color 0.15s ease, background 0.15s ease;
}
div[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--evidence-dark) !important;
    background: var(--paper-dim) !important;
}
[data-testid*="FileUploader"] svg { color: var(--evidence) !important; fill: var(--evidence) !important; }
[data-testid*="FileUploader"] p,
[data-testid*="FileUploader"] span,
[data-testid*="FileUploader"] small,
[data-testid*="FileUploader"] div { color: var(--evidence-dark) !important; }
[data-testid*="FileUploader"] small { color: var(--muted) !important; }

/* the actual "Browse files" button rendered inside the dropzone */
[data-testid*="FileUploader"] button {
    background: var(--paper) !important;
    border: 1.5px solid var(--evidence) !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
}
[data-testid*="FileUploader"] button:hover {
    background: var(--evidence) !important;
}
[data-testid*="FileUploader"] button p,
[data-testid*="FileUploader"] button span,
[data-testid*="FileUploader"] button svg { color: var(--evidence) !important; fill: var(--evidence) !important; }
[data-testid*="FileUploader"] button:hover p,
[data-testid*="FileUploader"] button:hover span,
[data-testid*="FileUploader"] button:hover svg { color: var(--paper) !important; fill: var(--paper) !important; }

/* uploaded-file chip that appears after selecting a file */
div[data-testid="stFileUploaderFile"] {
    background: var(--paper-dim) !important; border: 1px solid var(--line) !important;
    border-radius: 4px !important;
}
div[data-testid="stFileUploaderFile"] span, div[data-testid="stFileUploaderFile"] small,
div[data-testid="stFileUploaderFile"] p { color: var(--ink) !important; }

/* ---------- Expander ---------- */
details, [data-testid="stExpander"] {
    background: var(--paper) !important; border: 1px solid var(--line) !important;
    border-radius: 6px !important;
}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span { color: var(--ink) !important; font-weight: 500 !important; }
[data-testid="stExpander"] svg { color: var(--evidence) !important; }

/* ---------- Spinner ---------- */
[data-testid="stSpinner"] > div { color: var(--evidence) !important; }
[data-testid="stSpinner"] svg { color: var(--evidence) !important; }

/* ---------- Alerts (info/warning/success) ---------- */
div[data-testid="stAlert"] { border-radius: 6px !important; }
div[data-testid="stAlert"] p { color: var(--ink) !important; }
</style>
""", unsafe_allow_html=True)


# ── Cached resources (loaded once per session) ───────────────────
@st.cache_resource
def get_vector_store():
    return VectorStore()


@st.cache_resource
def get_reranker():
    return Reranker()


vs = get_vector_store()
reranker = get_reranker()


def rebuild_bm25():
    raw = vs.collection.get(include=["documents", "metadatas"])
    all_chunks = [{"chunk_id": cid, "text": doc, **meta}
                  for cid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])]
    bm25 = BM25Index()
    bm25.build(all_chunks)
    return bm25, all_chunks


# ── Sidebar: upload + corpus stats ───────────────────────────────
with st.sidebar:
    st.markdown(f'<div class="side-heading">{ICON_LIBRARY} Your Papers</div>', unsafe_allow_html=True)

    raw = vs.collection.get(include=["metadatas"])
    paper_titles = sorted(set(clean_text(m["title"]) for m in raw["metadatas"])) if raw["metadatas"] else []

    st.markdown(
        f'<div class="stat-row">'
        f'<div class="stat-box"><div class="num">{len(paper_titles)}</div><div class="label">Papers</div></div>'
        f'<div class="stat-box"><div class="num">{vs.count()}</div><div class="label">Chunks</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    with st.expander(f"View indexed papers ({len(paper_titles)})", expanded=False):
        for t in paper_titles:
            st.markdown(f"- {t}")

    st.divider()
    st.markdown(f'<div class="side-heading">{ICON_UPLOAD} Upload New Papers</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop PDFs here", type="pdf", accept_multiple_files=True, label_visibility="collapsed"
    )

    if uploaded_files and st.button("Ingest papers", use_container_width=True, type="primary"):
        os.makedirs("data/papers", exist_ok=True)
        progress = st.progress(0, text="Starting...")

        for i, file in enumerate(uploaded_files):
            dest = os.path.join("data/papers", file.name)
            with open(dest, "wb") as f:
                shutil.copyfileobj(file, f)

            paper_id = os.path.splitext(file.name)[0]
            progress.progress(i / len(uploaded_files), text=f"Parsing {paper_id}...")
            paper = extract_paper(dest, paper_id)
            chunks = [chunk_to_dict(c) for c in chunk_paper(paper)]
            vs.add_chunks(chunks)
            progress.progress((i + 1) / len(uploaded_files), text=f"Indexed {paper_id}")

        st.success(f"Ingested {len(uploaded_files)} paper(s)")
        st.rerun()

    st.divider()
    st.caption("Hybrid dense + BM25 retrieval, cross-encoder reranking, "
               "and citation-grounded generation via Gemini.")


# ── Main area: title + Q&A ───────────────────────────────────────
st.markdown(
    f'<div class="hero-row"><div class="icon">{ICON_DOC}</div>'
    f'<h1 class="hero-title">ResearchPilot</h1></div>'
    f'<div class="hero-sub">Evidence-grounded research assistant — every claim is traceable '
    f'to a page and section.</div>',
    unsafe_allow_html=True,
)

question = st.text_input(
    "Ask a question",
    placeholder="e.g. Compare the methodologies used across these papers",
    label_visibility="collapsed",
)
ask_clicked = st.button("Ask ResearchPilot", type="primary")

if not ask_clicked:
    st.markdown(
        f'<div class="empty-hint">{ICON_ARROW} Ask about a method, a result, or how two papers '
        f'compare — answers come with page-level citations.</div>',
        unsafe_allow_html=True,
    )

if ask_clicked and question:
    if vs.count() == 0:
        st.warning("No papers indexed yet — upload some in the sidebar first.")
    else:
        with st.spinner("Retrieving evidence and generating a grounded answer..."):
            bm25, _ = rebuild_bm25()
            retriever = HybridRetriever(vs, bm25)
            candidates = retriever.retrieve(question, top_k=20)
            top_chunks = reranker.rerank(question, candidates, top_k=6)
            result = answer_question(question, top_chunks)

        # Opened/closed as separate calls on purpose: this lets Streamlit run its
        # normal markdown parser on result["answer"] (so **bold**, lists, and
        # citation brackets render correctly) while still wrapping it in our
        # styled card via the surrounding raw HTML.
        st.markdown('<div class="answer-box"><div class="answer-label">Answer</div>', unsafe_allow_html=True)
        st.markdown(result["answer"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sources-label">Sources</div>', unsafe_allow_html=True)
        for s in result["sources"]:
            st.markdown(
                f'<div class="source-card">'
                f'<span class="source-ref">[{s["ref"]}]</span>'
                f'<span>{clean_text(s["paper"])} &nbsp;·&nbsp; {clean_text(s["section"])}, page {s["page"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

elif ask_clicked and not question:
    st.info("Type a question above first.")