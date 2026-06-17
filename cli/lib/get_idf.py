import math

from lib.keyword_search import tokenizeSingleTerm
from lib.InvertedIndex import InvertedIndex


def get_idf(term, args):
    invIdx = InvertedIndex()
    invIdx.load()
    tokenTerm = tokenizeSingleTerm(term)
    idf = math.log((len(invIdx.docmap) + 1) / (len(invIdx.index[tokenTerm]) + 1 ))
    return idf
    