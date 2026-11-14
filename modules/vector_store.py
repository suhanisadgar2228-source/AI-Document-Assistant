
import faiss
import numpy as np

from modules.embeddings import generate_embeddings


class VectorStore:
    """In-memory FAISS vector database for document chunks."""

    def __init__(self):
        self.index = None
        self.chunks = []
        self.dimension = None

    def build(self, chunks):
        """Generate embeddings and build the FAISS search index."""

        if not chunks:
            raise ValueError("There are no text chunks to index.")

        # Remove empty chunks
        valid_chunks = [
            chunk for chunk in chunks
            if chunk.get("text", "").strip()
        ]

        if not valid_chunks:
            raise ValueError("No valid text chunks were found.")

        texts = [chunk["text"] for chunk in valid_chunks]

        embeddings = np.asarray(
            generate_embeddings(texts),
            dtype=np.float32,
        )

        # Validate embedding dimensions
        if embeddings.ndim != 2 or embeddings.shape[0] != len(valid_chunks):
            raise ValueError("Invalid embedding dimensions.")

        if not np.all(np.isfinite(embeddings)):
            raise ValueError("Embeddings contain invalid values.")

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        self.dimension = embeddings.shape[1]

        # Build FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)

        self.index.add(
            np.ascontiguousarray(embeddings)
        )

        self.chunks = valid_chunks

        return self.index.ntotal

    def search(self, question, top_k=5, min_score=None):
        """Retrieve the most relevant document chunks."""

        if self.index is None or not self.chunks:
            return []

        if not question or not question.strip():
            return []

        if top_k <= 0:
            return []

        # Generate question embedding
        query_vector = np.asarray(
            generate_embeddings([question.strip()]),
            dtype=np.float32,
        )

        if query_vector.ndim != 2:
            raise ValueError("Invalid question embedding.")

        if query_vector.shape[1] != self.dimension:
            raise ValueError("Question embedding dimension mismatch.")

        if not np.all(np.isfinite(query_vector)):
            raise ValueError("Question embedding contains invalid values.")

        # Normalize query for cosine similarity
        faiss.normalize_L2(query_vector)

        # Retrieve matching chunks
        k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(
            np.ascontiguousarray(query_vector),
            k,
        )

        results = []

        for score, chunk_index in zip(scores[0], indices[0]):

            if chunk_index < 0:
                continue

            score = float(score)

            # Optional relevance filtering
            if min_score is not None and score < min_score:
                continue

            item = self.chunks[int(chunk_index)].copy()
            item["score"] = round(score, 4)

            results.append(item)

        return results

    def clear(self):
        """Clear the vector database."""

        self.index = None
        self.chunks = []
        self.dimension = None
