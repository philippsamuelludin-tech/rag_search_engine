import string
from lib.load_stopwords import load_stopwords
from nltk.stem import PorterStemmer
from lib.preprocess_text import preprocess_text

stopwords = load_stopwords()
stemmer = PorterStemmer()

def tokenize_text(text: str) -> list[str]:

    text = preprocess_text(text)
    tokenizedTextCopy = [token for token in text.split() if token]
    tokenizedText = []
    
    for token in tokenizedTextCopy:
        if token not in stopwords:
            tokenizedText.append(stemmer.stem(token))
        
    return tokenizedText

def has_matching_token(query: list[str], movieTitle: list[str]) -> bool:
    for queryWord in query:
        for movieWord in movieTitle:
            if queryWord in movieWord:
                return True
    return False

def tokenizeSingleTerm(term):
    tokenTerm = tokenize_text(term)
    if len(tokenTerm) != 1:
        raise Exception("More than one word")
    else:
        return tokenTerm[0]
    