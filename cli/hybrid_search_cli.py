import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparsers.add_parser("normalize", help="Get normalized Values of the input")
    normalize_parser.add_argument("values", type=float, nargs="*", help="The values to normlize")
    weighted_search_parser = subparsers.add_parser("weighted-search", help="Weighted Search")
    weighted_search_parser.add_argument("query", type=str, help="The query to search for")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5)
    weighted_search_parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

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
            print(
                f"Weighted Hybrid Search Results for '{result['query']}' (alpha={result['alpha']}):"
            )
            print(
                f"  Alpha {result['alpha']}: {int(result['alpha'] * 100)}% Keyword, {int((1 - result['alpha']) * 100)}% Semantic"
            )
            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                print(f"   Hybrid Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                if "bm25_score" in metadata and "semantic_score" in metadata:
                    print(
                        f"   BM25: {metadata['bm25_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}"
                    )
                print(f"   {res['document'][:100]}...")
                print()


        case _:
            parser.print_help()

if __name__ == "__main__":
    main()