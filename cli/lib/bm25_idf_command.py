from lib.keyword_search import tokenizeSingleTerm
from lib.InvertedIndex import InvertedIndex


def bm25_idf_command(term: str) -> float:
    invIdx = InvertedIndex()
    invIdx.load()
    term = tokenizeSingleTerm(term)
    return invIdx.get_bm25_idf(term)