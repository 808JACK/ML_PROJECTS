# Movie Recommender System

## What's This About?
Have you ever been unable to decide on your next movie? This project uses content-based filtering to recommend movies that fit your tastes. 
It analyzes features such as genres, plot descriptions, actors, and much more to give you sensible suggestions.

## The Dataset
I used the TMDB Movie [TMDB Movie Metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata), which is full of some cool movie details—like casts, genres, and keywords. It's great to play around with a recommender system for a movie.

## How It Works:
- **It Understands Movies:** The system takes in movie data and extracts the features.
- **Vectorization:** These features are turned into numerical representations through techniques like TF-IDF or Count Vectorization so that they can be compared mathematically.
- **Similarity:** It calculates similarities of movies using cosine similarity, etc.
- **Recommendation:** According to your interest, it recommends movies you'll probably watch next!

## Tools:
1. **Python** for everything under the hood.
2. **Pandas** to play with the data.
3. **Scikit-learn** to calculate similarity.
4. **NLTK** for cleaning and text processing.
