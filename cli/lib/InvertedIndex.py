from lib.load_movies import load_movies
from lib.keyword_search import tokenize_text
from collections import defaultdict, Counter
from pickle import dump, load
import os
import math
from lib.constants import *



class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)

    def __add_document(self, doc_id, text):
        tokenizedText = tokenize_text(text)
        for token in tokenizedText:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term):
        return sorted(self.index[term])
    
    def build(self):
        movies = load_movies()
        for movie in movies["movies"]:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie

    def save(self):
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as f:
            dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            dump(self.docmap, f)
        with open("cache/term_frequencies.pkl", "wb") as f:
            dump(self.term_frequencies, f)

    def load(self):
        with open("cache/index.pkl", "rb") as f:
            self.index = load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = load(f)
        with open("cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = load(f)
    
    def get_tf(self, doc_id, term):
        docIDCounter = self.term_frequencies[doc_id]
        return docIDCounter[term]
    
    def get_bm25_idf(self, term: str) -> float:
        return math.log((len(self.docmap) - len(self.index[term]) + 0.5) / (len(self.index[term]) + 0.5) + 1)
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1):
        tf = self.get_tf(doc_id, term)
        sat_tf = (tf * (k1 + 1)) / (tf + k1)
        return sat_tf