#!/usr/bin/env python3

import argparse
import json
import re


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
    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Generate chunked embeddings for movie descriptions")
    chunk_paser = subparsers.add_parser("chunk", help="fixed-size chunking to split long text into smaller pieces for embedding.")
    chunk_paser.add_argument("text", type=str, help="the text to chunk")
    chunk_paser.add_argument("--chunk-size", type=int, default=200, help="the size of a single chunk")
    chunk_paser.add_argument("--overlap", type=int, help="How much overlap between the chunks")
    semantic_chunk1 = subparsers.add_parser("semantic_chunk", help="chunks longer sentances")
    semantic_chunk1.add_argument("text", type=str, help="the text to chunk")
    semantic_chunk1.add_argument("--max-chunk-size", type=int, default=4, help="max chunk size")
    semantic_chunk1.add_argument("--overlap", type=int, default=0, help="overlap between chunks")
    args = parser.parse_args()

    match args.command:

        case "search":
            from lib.semantic_search import SemanticSearch
            search = SemanticSearch()
            movies = []
            with open("data/movies.json", "r", encoding="UTF-8") as f:
                movies = json.load(f)
            search.load_or_create_embeddings(movies["movies"])
            docs = search.search(args.query, args.limit)
            counter = 1
            for doc in docs:
                print(f"{counter}. {doc['title']} (score: {doc['score']:.4f})\n   {doc['description']}")
                counter += 1

        case "verify":
            from lib.semantic_search import verify_model
            verify_model()

        case "embed_text":
            from lib.semantic_search import embed_text
            embed_text(args.text)

        case "verify_embeddings":
            from lib.semantic_search import verify_embeddings
            verify_embeddings()

        case "embed_query":
            from lib.semantic_search import embed_query_text
            embed_query_text(args.query)

        case "embed_chunks":
            from lib.semantic_search import ChunkedSemanticSearch
            search = ChunkedSemanticSearch()
            with open("data/movies.json", "r", encoding="UTF-8") as f:
                movies = json.load(f)
            embeddings = search.load_or_create_chunk_embeddings(movies["movies"])
            print(f"Generated {len(embeddings)} chunked embeddings")

        case "chunk":
            text_split = args.text.split()
            chunks = []
            for i in range(0, len(text_split), args.chunk_size):
                if args.overlap > 0 and i != 0:
                    chunk_words = text_split[i-args.overlap:i + args.chunk_size]
                else:
                    chunk_words = text_split[i:i + args.chunk_size]
                chunk_str = " ".join(chunk_words)
                chunks.append(chunk_str)
            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks):
                print(f"{i+1}. {chunk}")

        case "semantic_chunk":
            from lib.semantic_search import semantic_chunk
            chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks):
                print(f"{i+1}. {chunk}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()