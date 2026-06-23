#!/usr/bin/env python3

import argparse

from lib.semantic_search import *
import json

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search = subparsers.add_parser("search", help="search shit")
    search.add_argument("query", type=str, help="the query to search for")
    search.add_argument("--limit", type=int, default=5, help="How many results")
    verfiy_parser = subparsers.add_parser("verify", help="Verify the model loaded")
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verfiys the generated embeddings")
    embed_query_parser = subparsers.add_parser("embed_query", help="embed the query")
    embed_query_parser.add_argument("query", type=str, help="the query to embed")
    embed_parser = subparsers.add_parser("embed_text", help="Embeds a input text")
    embed_parser.add_argument("text", type=str, help="The text to embed")
    args = parser.parse_args()

    match args.command:

        case "search":
            search = SemanticSearch()
            movies = []
            with open("data/movies.json", "r", encoding="UTF-8") as f:
                movies = json.load(f)
            search.load_or_create_embeddings(movies["movies"])
            docs = search.search(args.query, args.limit)
            counter = 1
            for doc in docs:
                print(f"{counter}. {doc["title"]} (score: {doc["score"]:.4f})\n   {doc["description"]}")
                counter += 1

        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embed_query":
            embed_query_text(args.query)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()