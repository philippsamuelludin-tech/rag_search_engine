import json

def load_movies():
    with open("data/movies.json", "r", encoding="utf-8") as f:
        movies = json.load(f)
    return movies