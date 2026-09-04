"""
Stage 5: Vector store.
Chroma persists to disk (no Docker/server needed) - good for 4-day scope.
Embeddings come from a local sentence-transformers model (free, offline).
"""
import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, good enough to start
COLLECTION_NAME = "research_pilot"


class VectorStore:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME)
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

    def add_chunks(self, chunks: list[dict]):
        """chunks: list of dicts with chunk_id, text, paper_id, title, page, section."""
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()
        self.collection.add(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{
                "paper_id": c["paper_id"], "title": c["title"],
                "page": c["page"], "section": c["section"],
            } for c in chunks],
        )

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        q_emb = self.embedder.encode([query]).tolist()
        res = self.collection.query(query_embeddings=q_emb, n_results=top_k)
        out = []
        for i in range(len(res["ids"][0])):
            out.append({
                "chunk_id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "score": 1 - res["distances"][0][i],  # cosine distance -> similarity
                **res["metadatas"][0][i],
            })
        return out

    def count(self) -> int:
        return self.collection.count()
