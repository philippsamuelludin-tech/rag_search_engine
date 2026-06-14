import string

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    return [token for token in text.split() if token]

def has_matching_token(query: list[str], movieTitle: list[str]) -> bool:
    for queryWord in query:
        for movieWord in movieTitle:
            if queryWord in movieWord:
                return True
    return False