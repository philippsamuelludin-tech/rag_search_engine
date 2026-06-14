#!/usr/bin/env python3

import argparse
import json
import string

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            query = args.query
            transTable = str.maketrans("", "", string.punctuation)
            print(f"Searching for: {query}")

            with open("data/movies.json", "r", encoding="utf-8") as f:
                movies = json.load(f)

            list_of_movies_with_query = []

            for movie in movies["movies"]:

                if query.lower().translate(transTable) in movie["title"].lower().translate(transTable):
                    list_of_movies_with_query.append(movie["title"])

            i = 1
            for movie in list_of_movies_with_query:
                if i < 6:
                    print(f"{i}. {movie}")
                    i += 1

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()