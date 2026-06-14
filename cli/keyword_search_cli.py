#!/usr/bin/env python3

import argparse
import json
from keyword_search import has_matching_token, tokenize_text

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            query = args.query
            print(f"Searching for: {query}")

            with open("data/movies.json", "r", encoding="utf-8") as f:
                movies = json.load(f)

            list_of_movies_with_query = []

            tokenizedQuery = tokenize_text(query)

            for movie in movies["movies"]:
                tokenizedMovie = tokenize_text(movie["title"])
                if has_matching_token(tokenizedQuery, tokenizedMovie) == True:
                    list_of_movies_with_query.append(movie["title"])

            for i, movie in enumerate(list_of_movies_with_query[:5], 1):
                print(f"{i}. {movie}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()