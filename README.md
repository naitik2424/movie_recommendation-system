# Movie Recommendation System

Collaborative filtering based movie recommendation system 
built using the MovieLens dataset.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn (Cosine Similarity)
- Google Colab

## Dataset
MovieLens Small Dataset — 100,836 ratings, 9,742 movies, 610 users

## How it works
1. Load and merge ratings + movies data
2. Build User-Item Matrix (610 x 9719)
3. Calculate Cosine Similarity between users
4. Find top similar users
5. Recommend unseen movies based on weighted ratings

## Model Performance
RMSE: 0.9754 on 500 test samples

## Sample Output
User 1 Recommendations:
1. Terminator 2: Judgment Day (1991)
2. Aliens (1986)
3. Sixth Sense, The (1999)
4. Hunt for Red October, The (1990)
5. Godfather, The (1972)
