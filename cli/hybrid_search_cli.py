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
    rrf_parser = subparsers.add_parser("rrf-search", help="Calculate the RRF-Score")
    rrf_parser.add_argument("query", type=str, help="The query to search")
    rrf_parser.add_argument("-k", default=60, type=int, help="The weight when calculating results")
    rrf_parser.add_argument("--limit", default=5, type=int, help="The Limit of result returned")

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
            for i, res in enumerate(result, 1):
                print(f"{i}. {res['title']}")
                print(f"  Hybrid Score: {res['score']:.3f}")
                print(f"  BM25: {res['metadata']['bm25_score']:.3f}, Semantic: {res['metadata']['semantic_score']:.3f}")
                print(f"  {res['document']}")

        case "rrf-search":
            from lib.hybrid_search import HybridSearch
            from lib.search_utils import load_movies
            movies = load_movies()
            rf_search = HybridSearch(movies)
            result = rf_search.rrf_search(args.query, args.k, args.limit)
            for i, res in enumerate(result, 1):
                print(f"{i}. {res['title']}")
                print(f"  RRF Score: {res['score']:.3f}")
                print(f"  BM25 Rank: {res['metadata']['bm25_rank']:.3f}, Semantic Rank: {res['metadata']['semantic_rank']:.3f}")
                print(f"  {res['document']}")


        case _:
            parser.print_help()

if __name__ == "__main__":
    main()