#!/usr/bin/env python3

import argparse
import json

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            with open("/data/movies.json", "r", encoding="utf-8") as f:
                movies = json.load(f)
            list_of_movies_with_query = []
            for movie in movies["movies"]:
                if movie.title.contains(args.query):
                    list_of_movies_with_query.append(movie.title)
            for movie in list_of_movies_with_query:

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()