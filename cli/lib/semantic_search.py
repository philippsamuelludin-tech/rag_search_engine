import os
from sentence_transformers import SentenceTransformer
import numpy as np
import json

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
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
        for doc in documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if len(self.embeddings) == len(documents):
                return self.embeddings
        else:
            self.build_embeddings(documents)
            return self.embeddings
        
    def search(self, query, limit):
        if self.embeddings == None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        embedding = self.gernerate_embedding(query)
        similarity_list = []
        for doc_emb in self.embeddings:
            sim = cosine_similarity(doc_emb, embedding)
        similarity_list = [(key, value) for key, value in sorted(similarity_list, key=lambda item: item[1], reverse=True)]




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