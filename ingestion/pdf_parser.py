"""
Stage 1: PDF -> structured text.
Uses pymupdf4llm to get markdown with page metadata preserved.
"""
import pymupdf4llm
import re
from dataclasses import dataclass, field


@dataclass
class Paper:
    paper_id: str
    title: str
    raw_markdown: str
    pages: list = field(default_factory=list)  # list of {page, text}


SECTION_HEADERS = [
    "abstract", "introduction", "related work", "background",
    "methodology", "methods", "dataset", "datasets", "experiments",
    "experimental setup", "results", "discussion", "limitations",
    "conclusion", "references",
]


def extract_paper(pdf_path: str, paper_id: str) -> Paper:
    """Extract page-aware markdown chunks from a PDF."""
    page_data = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    # page_data is a list of dicts: {"text": ..., "metadata": {"page": n, ...}}
    pages = [{"page": p["metadata"].get("page", i + 1), "text": p["text"]}
             for i, p in enumerate(page_data)]
    full_text = "\n\n".join(p["text"] for p in pages)
    title = _guess_title(pages[0]["text"] if pages else "")
    return Paper(paper_id=paper_id, title=title, raw_markdown=full_text, pages=pages)


def _guess_title(first_page_text: str) -> str:
    """First non-empty markdown heading/line on page 1 is usually the title."""
    for line in first_page_text.splitlines():
        line = line.strip("# ").strip()
        if len(line) > 8 and not line.lower().startswith("arxiv"):
            return line
    return "Untitled"


def detect_section(text_chunk: str) -> str:
    """Cheap heuristic: does this chunk start under a known section heading?"""
    lowered = text_chunk.lower()
    for header in SECTION_HEADERS:
        if re.search(rf"\b{re.escape(header)}\b", lowered[:120]):
            return header.title()
    return "Body"


if __name__ == "__main__":
    import sys
    p = extract_paper(sys.argv[1], "test_paper")
    print(f"Title: {p.title}")
    print(f"Pages: {len(p.pages)}")
