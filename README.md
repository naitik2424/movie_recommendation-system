# 🎬 Movie Recommendation System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://naitik2424-movie-recommendation-system-app-wxvxbn.streamlit.app/)

![App Banner](https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=1000&auto=format&fit=crop)

**🚀 Live Demo:** [Click here to try the web app!](https://naitik2424-movie-recommendation-system-app-wxvxbn.streamlit.app/)

A sleek, interactive, collaborative filtering-based movie recommendation system built using the MovieLens dataset.

## 🛠️ Tech Stack
- **Frontend & Deployment:** Streamlit
- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (Cosine Similarity)

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
