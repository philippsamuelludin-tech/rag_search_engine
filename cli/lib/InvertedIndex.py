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
        self.doc_lengths = {}
        self.doc_lengths_path = os.path.join("cache", "doc_lengths.pkl")

    def __add_document(self, doc_id, text):
        tokenizedText = tokenize_text(text)
        tokenAmount = len(tokenizedText)
        for token in tokenizedText:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1
            self.doc_lengths[doc_id] = tokenAmount

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
        with open("cache/doc_lengths.pkl", "wb") as f:
            dump(self.doc_lengths, f)

    def load(self):
        with open("cache/index.pkl", "rb") as f:
            self.index = load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = load(f)
        with open("cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = load(f)
        with open("cache/doc_lengths.pkl", "rb") as f:
            self.doc_lengths = load(f)

    def get_tf(self, doc_id, term):
        docIDCounter = self.term_frequencies[doc_id]
        return docIDCounter[term]
    
    def get_bm25_idf(self, term: str) -> float:
        return math.log((len(self.docmap) - len(self.index[term]) + 0.5) / (len(self.index[term]) + 0.5) + 1)
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        tf = self.get_tf(doc_id, term)
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return tf_component
    
    def __get_avg_doc_length(self) -> float:
        if len(self.docmap) == 0:
            return 0.0
        count = 0
        for doc in self.docmap:
            count += self.doc_lengths[doc]
        count = count / len(self.docmap)
        return count
    
    def bm25(self, doc_id, term):
        idf = self.get_bm25_idf(term)
        tf = self.get_bm25_tf(doc_id, term)
        return idf * tf
    
    def bm25_search(self, query, limit=5):
        tokens = tokenize_text(query)
        scores = {}
        for doc in self.docmap:
            running_score = 0
            for token in tokens:
                running_score += self.bm25(doc, token)
            scores[doc] = running_score
        scores = [(key, value) for key, value in sorted(scores.items(), key=lambda item: item[1], reverse=True)]
        return_score = []
        for i in range(limit):
            return_score.append(scores[i])
        return return_score