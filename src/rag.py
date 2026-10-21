"""RAG engine: chunk, embed, store, retrieve."""
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from src.config import PROJECT_ROOT

# Where ChromaDB stores its data on disk
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "it_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class RAGEngine:
    """Local RAG using ChromaDB + sentence-transformers."""

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        # Persistent client — stores vectors on disk, survives restarts
        self.client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

        # Embedding function — uses sentence-transformers locally
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        # Get or create a collection (think: a table in a regular DB)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedder,  # type: ignore[arg-type]
        )

    # ──────────────────────────────────────────────────────────
    # INDEXING (done once, ahead of time)
    # ──────────────────────────────────────────────────────────

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> list[str]:
        """Split text into overlapping chunks.

        Overlap helps preserve context at chunk boundaries,
        so a sentence split mid-way is still findable.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start = end - overlap
        return [c for c in chunks if c]  # drop empty

    def ingest_document(self, file_path: Path, doc_id_prefix: str = "doc") -> int:
        """Load a text file, chunk it, embed it, store it."""
        text = file_path.read_text(encoding="utf-8")
        chunks = self.chunk_text(text)

        # Chroma needs: ids, documents (the text), optional metadata
        ids = [f"{doc_id_prefix}_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": file_path.name, "chunk_index": i} for i in range(len(chunks))
        ]

        self.collection.add(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,  # type: ignore[arg-type]
        )

        return len(chunks)

    def reset_collection(self) -> None:
        """Wipe the collection and start fresh."""
        self.client.delete_collection(name=self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedder,  # type: ignore[arg-type]
        )

    # ──────────────────────────────────────────────────────────
    # RETRIEVAL (done at query time)
    # ──────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 2) -> dict:
        """Find the top-K most relevant chunks for a query.

        Returns the chunks + retrieval latency metric.
        """
        start = time.perf_counter()

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        retrieval_time = time.perf_counter() - start

        # Chroma returns a dict with lists wrapped in another list (for batching).
        # Fields can be None in the type stubs — default to empty wrappers for safety.
        documents_batch = results.get("documents") or [[]]
        distances_batch = results.get("distances") or [[]]
        metadatas_batch = results.get("metadatas") or [[]]

        return {
            "chunks": list(documents_batch[0]),
            "distances": list(distances_batch[0]),
            "metadatas": list(metadatas_batch[0]),
            "retrieval_time": retrieval_time,
            "query": query,
        }

    def count(self) -> int:
        """How many chunks are stored."""
        return self.collection.count()