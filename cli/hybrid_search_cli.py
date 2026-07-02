import argparse
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparsers.add_parser("normalize", help="Get normalized Values of the input")
    normalize_parser.add_argument("values", type=float, nargs="*", help="The values to normlize")
    weighted_search_parser = subparsers.add_parser("weighted-search", help="Weighted Search")
    weighted_search_parser.add_argument("query", type=str, help="The query to search for")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5)
    weighted_search_parser.add_argument("--limit", type=int, default=5)
    rrf_parser = subparsers.add_parser("rrf-search", help="Calculate the RRF-Score")
    rrf_parser.add_argument("query", type=str, help="The query to search")
    rrf_parser.add_argument("-k", default=60, type=int, help="The weight when calculating results")
    rrf_parser.add_argument("--limit", default=5, type=int, help="The Limit of result returned")
    rrf_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite"], help="Query enhancement method")

    args = parser.parse_args()

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    model = "openrouter/free"
    

    match args.command:

        case "normalize":
            from lib.search_utils import normalize
            result = normalize(args.values)
            for res in result:
                print(f"* {res:.4f}")

        case "weighted-search":
            from lib.hybrid_search import HybridSearch
            from lib.search_utils import load_movies
            movies = load_movies()
            h_search = HybridSearch(movies)
            result = h_search.weighted_search(args.query, args.alpha, args.limit)
            for i, res in enumerate(result, 1):
                print(f"{i}. {res['title']}")
                print(f"  Hybrid Score: {res['score']:.3f}")
                print(f"  BM25: {res['metadata']['bm25_score']:.3f}, Semantic: {res['metadata']['semantic_score']:.3f}")
                print(f"  {res['document']}")

        case "rrf-search":
            query_to_search = args.query

            if args.enhance == "spell":
                prompt = f"""Fix any spelling errors in the user-provided movie search query below.
                Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                Preserve punctuation and capitalization unless a change is required for a typo fix.
                If there are no spelling errors, or if you're unsure, output the original query unchanged.
                Output only the final query text, nothing else.
                User query: "{query_to_search}"
                """

                response = client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}]
                )
                assert response.usage is not None
                enhanced_query = response.choices[0].message.content.strip()
                if enhanced_query == query_to_search:
                    print("Query not enhanced")
                else:
                    print(f"Enhanced query ({args.enhance}): '{query_to_search}' -> '{enhanced_query}'\n")
                
                query_to_search = enhanced_query

            elif args.enhance == "rewrite":
                prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions (horror = scary, animation = cartoon)
                - Keep the rewritten query concise (under 10 words)
                - It should be a Google-style search query, specific enough to yield relevant results
                - Don't use boolean logic

                Examples:
                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                If you cannot improve the query, output the original unchanged.
                Output only the rewritten query text, nothing else.

                User query: "{query_to_search}"
                """

                response = client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}]
                )
                assert response.usage is not None
                enhanced_query = response.choices[0].message.content.strip()
                if enhanced_query == query_to_search:
                    print("Query not enhanced")
                else:
                    print(f"Enhanced query ({args.enhance}): '{query_to_search}' -> '{enhanced_query}'\n")
                
                query_to_search = enhanced_query
                
            

            from lib.hybrid_search import HybridSearch
            from lib.search_utils import load_movies
            movies = load_movies()
            rf_search = HybridSearch(movies)
            result = rf_search.rrf_search(query_to_search, args.k, args.limit)
            for i, res in enumerate(result, 1):
                print(f"{i}. {res['title']}")
                print(f"  RRF Score: {res['score']:.3f}")
                print(f"  BM25 Rank: {res['metadata']['bm25_rank']:.3f}, Semantic Rank: {res['metadata']['semantic_rank']:.3f}")
                print(f"  {res['document']}")


        case _:
            parser.print_help()

if __name__ == "__main__":
    main()