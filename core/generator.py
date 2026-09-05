"""
Stage 11/17: Grounded generation + citation + abstention.
The prompt forces the model to (a) answer only from provided evidence,
(b) tag every claim with [n] pointing at a source chunk, and
(c) abstain if evidence is insufficient.
"""
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types


MODEL = "gemini-3.6-flash"

client = genai.Client()
SYSTEM_PROMPT = """You are ResearchPilot, a research assistant that answers ONLY from the
numbered evidence chunks given to you. Rules:
1. Every factual claim must end with a citation like [1] or [2][4] referencing chunk numbers.
2. Never use outside knowledge - if the evidence doesn't support an answer, say:
   "I couldn't find sufficient evidence in the provided papers to answer this."
3. Be concise and direct. No filler.
"""


def build_evidence_block(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] (Paper: {c['title']} | Section: {c['section']} | Page: {c['page']})\n{c['text']}"
        )
    return "\n\n".join(lines)


def answer_question(question: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {"answer": "No evidence retrieved - upload papers first.", "sources": []}

    evidence = build_evidence_block(chunks)
    prompt = f"Evidence:\n{evidence}\n\nQuestion: {question}"

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=800,
        ),
    )
    answer_text = response.text

    sources = [{
        "ref": i + 1, "paper": c["title"], "section": c["section"],
        "page": c["page"], "chunk_id": c["chunk_id"],
    } for i, c in enumerate(chunks)]

    return {"answer": answer_text, "sources": sources}
