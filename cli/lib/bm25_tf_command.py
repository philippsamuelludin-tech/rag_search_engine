from lib.keyword_search import tokenizeSingleTerm
from lib.InvertedIndex import InvertedIndex
from lib.constants import *


def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B):
    invIdx = InvertedIndex()
    invIdx.load()
    term = tokenizeSingleTerm(term)
    bm25_tf = invIdx.get_bm25_tf(doc_id, term, k1)
    return bm25_tf