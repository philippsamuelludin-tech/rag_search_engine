import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparsers.add_parser("normalize", help="Get normalized Values of the input")
    normalize_parser.add_argument("values", type=float, nargs="*", help="The values to normlize")

    args = parser.parse_args()

    match args.command:

        case "normalize":
            if args.values == []:
                pass
            else:
                minList = min(args.values)
                maxList = max(args.values)
                if minList == maxList:
                    print(1.0)
                else:
                    for value in args.values:
                        print(f"* {(value - minList) / (maxList - minList):.4f}")



        case _:
            parser.print_help()

if __name__ == "__main__":
    main()