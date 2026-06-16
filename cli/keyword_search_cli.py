#!/usr/bin/env python3

import argparse
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
            
            

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()