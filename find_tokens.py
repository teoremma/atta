import argparse

from transformers import AutoTokenizer


def _display_token(token: str) -> str:
    return token.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find vocab tokens that contain specified substrings.",
    )
    parser.add_argument("model_id", help="Model id or local path to load the tokenizer.")
    parser.add_argument("strings", nargs="+", help="One or more substrings to match.")
    parser.add_argument(
        "--case-insensitive",
        action="store_true",
        help="Case-insensitive substring matching.",
    )
    parser.add_argument(
        "--include-special",
        action="store_true",
        help="Include special tokens in the search.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    vocab = tokenizer.get_vocab()
    special_tokens = set(tokenizer.all_special_tokens)

    if args.case_insensitive:
        query_strings = [s.lower() for s in args.strings]
    else:
        query_strings = args.strings

    matches = []
    for token, token_id in vocab.items():
        if not args.include_special and token in special_tokens:
            continue
        haystack = token.lower() if args.case_insensitive else token
        if any(q in haystack for q in query_strings):
            matches.append((token_id, token))

    for token_id, token in sorted(matches, key=lambda item: item[0]):
        print(f"{token_id}\t{_display_token(token)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
