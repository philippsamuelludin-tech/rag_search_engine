from lib.InvertedIndex import InvertedIndex
from lib.keyword_search import tokenize_text



def search_command(args):
    invIdx = InvertedIndex()
    invIdx.load()

    query = args.query

    tokenizedQuery = tokenize_text(query)

    count = 0
    listOfMovies = []

    for token in tokenizedQuery:
        docID = invIdx.get_documents(token)
        for ID in docID:
            fullMovieDict = invIdx.docmap[ID]
            if fullMovieDict != None:
                count += 1
                listOfMovies.append(fullMovieDict)
            if count >= 5:
                break
    for i, movie in enumerate(listOfMovies):
        if i < 5:
            print(f"{i+1}. {movie["title"]}")
        else:
            return        

