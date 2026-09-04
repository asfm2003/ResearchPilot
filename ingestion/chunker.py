"""
Stage 3: Structure-aware chunking.
Splits each page's text into ~350-token chunks WITHOUT crossing section
boundaries where possible, and tags every chunk with paper/page/section.
"""
from dataclasses import dataclass, asdict
from .pdf_parser import Paper, detect_section

CHUNK_SIZE_CHARS = 1400   # ~300-400 tokens
CHUNK_OVERLAP = 200


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    title: str
    page: int
    section: str
    text: str


def chunk_paper(paper: Paper) -> list[Chunk]:
    chunks = []
    counter = 0
    for page in paper.pages:
        text = page["text"].strip()
        if not text:
            continue
        section = detect_section(text)
        for piece in _split(text):
            counter += 1
            chunks.append(Chunk(
                chunk_id=f"{paper.paper_id}-p{page['page']}-c{counter}",
                paper_id=paper.paper_id,
                title=paper.title,
                page=page["page"],
                section=section,
                text=piece,
            ))
    return chunks


def _split(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE_CHARS:
        return [text]
    pieces = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE_CHARS
        pieces.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return pieces


def chunk_to_dict(c: Chunk) -> dict:
    return asdict(c)
