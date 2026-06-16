import json
from lib.preprocess_text import preprocess_text

def load_stopwords():
    with open("data/stopwords.txt", "r", encoding="utf-8") as f:
        stopwords = f.read().splitlines()
        for i, word in enumerate(stopwords):
            stopwords[i] = preprocess_text(word)
    return stopwords