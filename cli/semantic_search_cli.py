#!/usr/bin/env python3

import argparse

from lib.semantic_search import *

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verfiy_parser = subparsers.add_parser("verify", help="Verify the model loaded")
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verfiys the generated embeddings")
    embed_query_parser = subparsers.add_parser("embed_query", help="embed the query")
    embed_query_parser.add_argument("query", type=str, help="the query to embed")
    embed_parser = subparsers.add_parser("embed_text", help="Embeds a input text")
    embed_parser.add_argument("text", type=str, help="The text to embed")
    args = parser.parse_args()

    match args.command:

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