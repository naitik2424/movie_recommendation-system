import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Set page config for a premium look
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main {
        background-color: #0f1116;
        color: #e2e8f0;
    }
    .stApp {
        background: radial-gradient(circle, #1a1c24 0%, #0f1116 100%);
    }
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #ffffff;
    }
    .movie-card {
        background-color: #1e2230;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2d3142;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .movie-card:hover {
        transform: translateY(-2px);
        border-color: #4f46e5;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #10b981;
    }
    .highlight {
        color: #6366f1;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Define dataset paths
DATA_DIR = "ml-latest-small"
ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ZIP_FILE = "ml-latest-small.zip"

# Cache data downloading and extraction
@st.cache_resource
def download_and_extract_data():
    if not os.path.exists(DATA_DIR):
        with st.spinner("Downloading MovieLens Dataset (ml-latest-small)..."):
            urllib.request.urlretrieve(ZIP_URL, ZIP_FILE)
            with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
                zip_ref.extractall(".")
            # Clean up the zip file
            if os.path.exists(ZIP_FILE):
                os.remove(ZIP_FILE)
    return DATA_DIR

# Cache data loading and preprocessing
@st.cache_data
def load_and_preprocess_data(data_dir):
    ratings = pd.read_csv(os.path.join(data_dir, 'ratings.csv'))
    movies = pd.read_csv(os.path.join(data_dir, 'movies.csv'))
    
    # Merge datasets to get titles
    df = pd.merge(ratings, movies, on='movieId')
    df_clean = df.drop(columns=['timestamp', 'movieId'])
    
    # Build User-Movie Matrix
    user_movie_matrix = df_clean.pivot_table(
        index='userId',
        columns='title',
        values='rating'
    )
    matrix_filled = user_movie_matrix.fillna(0)
    
    # Compute Cosine Similarity between users
    user_similarity = cosine_similarity(matrix_filled)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_movie_matrix.index,
        columns=user_movie_matrix.index
    )
    
    return user_movie_matrix, user_similarity_df, df_clean

# Initialize application data
data_dir = download_and_extract_data()
user_movie_matrix, user_similarity_df, raw_df = load_and_preprocess_data(data_dir)

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.image("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=300&auto=format&fit=crop", use_container_width=True)
st.sidebar.title("🎬 Recommender Controls")

# Input User Selection
total_users = int(user_movie_matrix.index.max())
selected_user = st.sidebar.number_input(
    "Enter User ID",
    min_value=1,
    max_value=total_users,
    value=1,
    step=1,
    help=f"Select a User ID between 1 and {total_users} to generate personalized recommendations."
)

# Number of Recommendations
num_recs = st.sidebar.slider(
    "Number of Recommendations",
    min_value=3,
    max_value=15,
    value=5,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Info")
st.sidebar.info(
    f"📊 **Users:** {user_movie_matrix.shape[0]}\n\n"
    f"🎬 **Movies:** {user_movie_matrix.shape[1]}\n\n"
    f"⭐ **Ratings:** {len(raw_df)}"
)

# --- MAIN APP INTERFACE ---
st.title("🎬 Movie Recommendation System")
st.caption("A Collaborative Filtering engine powered by Cosine Similarity using the MovieLens Dataset.")

# Recommendation calculation logic
def recommend_movies(user_id, num_recommendations=5):
    # Find similar users
    similar_users = user_similarity_df[user_id].drop(user_id)
    # Get top 10 similar users
    top_users = similar_users.sort_values(ascending=False).head(10)
    
    # Get watched movies by current user
    watched = user_movie_matrix.loc[user_id]
    watched_movies = watched[watched > 0].index.tolist()
    
    scores = {}
    # Accumulate similarity-weighted ratings of unseen movies
    for similar_user, similarity in top_users.items():
        ratings_su = user_movie_matrix.loc[similar_user]
        unseen = ratings_su[~ratings_su.index.isin(watched_movies)]
        for movie, rating in unseen.items():
            if rating > 0:
                if movie not in scores:
                    scores[movie] = []
                scores[movie].append(similarity * rating)
                
    # Average the weighted scores
    averaged_scores = {movie: np.mean(val) for movie, val in scores.items()}
    recommended = sorted(averaged_scores.items(), key=lambda x: x[1], reverse=True)
    return recommended[:num_recommendations], top_users

# Calculate recommendations
recommendations, top_similar_users = recommend_movies(selected_user, num_recs)

# Layout: Split into tabs
tab1, tab2, tab3 = st.tabs([
    "🎯 Personalized Recommendations", 
    "📜 User Rating History", 
    "👥 Similar Users Detail"
])

# TAB 1: Recommendations
with tab1:
    st.subheader(f"Top {num_recs} Movie Recommendations for User {selected_user}")
    
    if len(recommendations) == 0:
        st.warning("No recommendations could be generated. Try selecting a user with more rating history.")
    else:
        for idx, (movie, score) in enumerate(recommendations, 1):
            st.markdown(f"""
            <div class="movie-card">
                <h3>{idx}. {movie}</h3>
                <p>Recommendation Confidence Score: <span class="metric-value">{score:.2f}</span></p>
            </div>
            """, unsafe_allow_html=True)

# TAB 2: User Rating History
with tab2:
    st.subheader(f"Movies Watched and Rated by User {selected_user}")
    
    user_ratings = user_movie_matrix.loc[selected_user]
    watched_history = user_ratings[user_ratings > 0].sort_values(ascending=False)
    
    if len(watched_history) == 0:
        st.info("This user has not rated any movies yet.")
    else:
        st.write(f"Showing all {len(watched_history)} movies rated by User {selected_user}:")
        
        # Display as a dataframe with a beautiful gradient
        history_df = pd.DataFrame({
            "Movie Title": watched_history.index,
            "Rating Given": watched_history.values
        })
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

# TAB 3: Similar Users Detail
with tab3:
    st.subheader("Top 10 Most Similar Users")
    st.write(f"These are the users whose historical rating profiles are closest to User {selected_user}.")
    
    similar_users_df = pd.DataFrame({
        "Similar User ID": top_similar_users.index,
        "Cosine Similarity Score": top_similar_users.values
    })
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(
            similar_users_df,
            use_container_width=True,
            hide_index=True
        )
    with col2:
        # Explanatory metric
        avg_similarity = top_similar_users.mean()
        st.metric(
            label="Average Similarity of Top 10 Neighbors",
            value=f"{avg_similarity:.2%}",
            delta=None
        )
        st.info("💡 **Why do we need similar users?**\n"
                "User-based collaborative filtering works by finding peers who share similar tastes. "
                "The ratings from these similar users are used to predict what movies our target user might enjoy.")

