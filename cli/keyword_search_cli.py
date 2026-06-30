#!/usr/bin/env python3

import argparse
from lib.bm25_tf_command import bm25_tf_command
from lib.constants import *
from lib.bm25_idf_command import bm25_idf_command
from lib.get_tfidf import get_tfidf
from lib.get_idf import get_idf
from lib.search_command import search_command
from lib.build_command import build_command
from lib.keyword_search import tokenizeSingleTerm
from lib.search_utils import load_movies
from lib.load_stopwords import load_stopwords
from lib.InvertedIndex import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    subparsers.add_parser("build", help="Build the inverted index")
    idf_parser = subparsers.add_parser("idf", help="Returns the Inverse Document Frequency")
    idf_parser.add_argument("term", type=str, help="the term to search for")
    tfidf_parser = subparsers.add_parser("tfidf", help="get the tf-idf of a term in a doc")
    tfidf_parser.add_argument("documentID", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="the term")
    tf_parser = subparsers.add_parser("tf", help="Return the term frequency")
    tf_parser.add_argument("documentID", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="the term")
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            search_command(args)
            
        case "build":
            build_command()

        case "tf":
            singleTerm = tokenizeSingleTerm(args.term)
            invIdx = InvertedIndex()
            invIdx.load()
            print(invIdx.get_tf(args.documentID, singleTerm))
            
        case "idf":
            idf = get_idf(args.term, args)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            get_tfidf(args, args.documentID, args.term)

        case "bm25idf":
            bm25 = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25:.2f}")

        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case "bm25search":
            invIdx = InvertedIndex()
            invIdx.load()
            results = invIdx.bm25_search(args.query)
            for i in range(len(results)):
                print(f"{i+1}. ({results[i][0]}){invIdx.docmap[results[i][0]]['title']} - Score: {results[i][1]:.2f}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()