from lib.InvertedIndex import InvertedIndex
from lib.get_idf import get_idf


def get_tfidf(args, docID, term):
    invIdx = InvertedIndex()
    invIdx.load()
    tf = invIdx.get_tf(docID, term)
    idf = get_idf(term, args)
    tfidf = tf * idf
    print(f"TF-IDF score of '{args.term}' in document '{args.documentID}': {tfidf:.2f}")