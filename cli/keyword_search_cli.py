#!/usr/bin/env python3

import argparse
from lib.get_tfidf import get_tfidf
from lib.get_idf import get_idf
from lib.search_command import search_command
from lib.build_command import build_command
from lib.keyword_search import tokenizeSingleTerm
from lib.load_movies import load_movies
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

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()