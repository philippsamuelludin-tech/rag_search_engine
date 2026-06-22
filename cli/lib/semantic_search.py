from sentence_transformers import SentenceTransformer
import numpy as np

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
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings

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
