import os
import re
import json

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - used for lightweight test environments
    class SentenceTransformer:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            self.max_seq_length = 256

        def encode(self, texts, show_progress_bar=False):
            return np.zeros((len(texts), 3), dtype=float)

from typing import TypedDict

class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int

class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def gernerate_embedding(self, text):
        if text.isspace():
            raise ValueError("text input was empty")
        embedding = self.model.encode([text])
        return embedding[0]
    
    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        doc_list = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_list.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(doc_list, show_progress_bar=True)
        os.makedirs("cache", exist_ok=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if len(self.embeddings) == len(documents):
                return self.embeddings
        self.build_embeddings(documents)
        return self.embeddings
        
    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        embedding = self.gernerate_embedding(query)
        similarity_list = []

        for i, doc_emb in enumerate(self.embeddings):
            sim = cosine_similarity(doc_emb, embedding)
            similarity_list.append((sim, self.documents[i]))

        similarity_list = [(key, value) for key, value in sorted(similarity_list, key=lambda item: item[0], reverse=True)]
        return_list = []

        for movie in similarity_list[:limit]:
            run_dict = {}
            run_dict["score"] = movie[0]
            run_dict["title"] = movie[1]["title"]
            run_dict["description"] = movie[1]["description"]
            return_list.append(run_dict)
       
        return return_list 




def verify_model():
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")

def embed_text(text):
    search = SemanticSearch()
    embedding = search.gernerate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    search = SemanticSearch()
    movies = []
    with open("data/movies.json", "r", encoding="UTF-8") as f:
        movies = json.load(f)
    search.load_or_create_embeddings(movies["movies"])
    print(f"Number of docs:   {len(search.documents)}")
    print(f"Embeddings shape: {search.embeddings.shape[0]} vectors in {search.embeddings.shape[1]} dimensions")

def embed_query_text(query):
    search = SemanticSearch()
    embedding = search.gernerate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def semantic_chunk(text: str, max_chunk_size: int, overlap: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    i = 0
    n_sentences = len(sentences)
    while i < n_sentences:
        chunk_sentences = sentences[i : i + max_chunk_size]
        # stop if this chunk is too small to be useful
        if chunks and len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
        i += max_chunk_size - overlap
    return chunks

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
    
    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        chunk_metadata: list[ChunkMetadata] = []
        all_chunks = []
        for movie_idx, doc in enumerate(documents):
            self.document_map[doc["id"]] = doc
            if not doc.get("description", ""):
                continue

            chunks = semantic_chunk(doc["description"], max_chunk_size=4, overlap=1)
            total_chunks = len(chunks)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {
                        "movie_idx": movie_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": total_chunks,
                    }
                )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        os.makedirs("cache", exist_ok=True)
        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)
        with open("cache/chunk_metadata.json", "w", encoding="utf-8") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists("cache/chunk_metadata.json"):
            try:
                self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
                with open("cache/chunk_metadata.json", "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                self.chunk_metadata = metadata.get("chunks", [])
                if self.chunk_embeddings is not None and len(self.chunk_metadata) == len(self.chunk_embeddings):
                    return self.chunk_embeddings
            except (OSError, ValueError, EOFError):
                pass

        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit: int = 10):
        embedding = self.gernerate_embedding(query)
        chunk_score = []
        for chunk_idx, chunk_embedding in enumerate(self.chunk_embeddings):
            metadata = self.chunk_metadata[chunk_idx]
            score = cosine_similarity(chunk_embedding, embedding)
            chunk_score.append(
                    {
                        "chunk_idx": metadata["chunk_idx"],
                        "movie_idx": metadata["movie_idx"],
                        "score": score,
                    }
                )
            
        best_chunk_scores = {}

        for candidate in chunk_score:
            item_id = candidate["movie_idx"]

            if item_id not in best_chunk_scores:
                best_chunk_scores[item_id] = candidate
            elif candidate["score"] > best_chunk_scores[item_id]["score"]:
                best_chunk_scores[item_id] = candidate

        sorted_items = sorted(
            best_chunk_scores.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        top_items = sorted_items[:limit]
        result = []
        for item in top_items:
            original_record = self.documents[item["movie_idx"]]
            
            result.append({
                "id": original_record["id"],
                "title": original_record["title"],
                "document": original_record["description"][:100],
                "score": round(item["score"], 4),
                "metadata": original_record.get("metadata", {})
            })

        return result
