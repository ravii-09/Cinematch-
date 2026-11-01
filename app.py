import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CineMatch | Movie Recommender", page_icon="🎬", layout="wide")

@st.cache_data
def load_movies():
    data = pd.read_csv("movies.csv")
    data["genres"] = data["genres"].fillna("")
    for col in ["overview", "keywords", "director", "cast"]:
        data[col] = data[col].fillna("")
    return data

movies = load_movies()
@st.cache_data(ttl=86400)
def get_movie_poster(title, year):

    # Read the secret token from Streamlit
    try:
        token = st.secrets["TMDB_READ_ACCESS_TOKEN"]
    except (KeyError, FileNotFoundError):
        return None

    url = "https://api.themoviedb.org/3/search/movie"

    headers = {
        "Authorization": f"Bearer {token}",
        "accept": "application/json"
    }

    params = {
        "query": title,
        "year": int(year),
        "include_adult": False
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])

        if results:
            poster_path = results[0].get("poster_path")

            if poster_path:
                return (
                    "https://image.tmdb.org/t/p/w500"
                    + poster_path
                )

        return None

    except requests.RequestException:
        return None

@st.cache_resource
def build_recommender(data):
    # Combine useful movie details so the algorithm can compare their content.
    data = data.copy()
    data["combined_features"] = (
        data["genres"].str.replace("|", " ", regex=False) + " " +
        data["overview"] + " " + data["keywords"] + " " +
        data["director"] + " " + data["cast"]
    )
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(data["combined_features"])
    similarity = cosine_similarity(matrix, matrix)
    return similarityim

similarity = build_recommender(movies)

st.markdown("""
<style>
.main {background-color:#10131c;}
.block-container {padding-top:2rem; padding-bottom:3rem;}
.hero {padding:28px 30px; border-radius:18px; background:linear-gradient(120deg,#242b42,#4a2348); margin-bottom:22px;}
.hero h1 {font-size:2.5rem; margin:0; color:white;}
.hero p {color:#e5e7eb; margin:8px 0 0 0;}
.movie-card {background:#1b2030; border:1px solid #343b50; padding:16px; border-radius:14px; min-height:190px;}
.movie-card h4 {margin:6px 0 8px 0;}
.small-muted {color:#b8bfd0; font-size:0.88rem;}
div.stButton > button {border-radius:10px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🎬 CineMatch</h1>
  <p>Discover your next favourite movie — recommendations based on what you enjoy.</p>
</div>
""", unsafe_allow_html=True)

def movie_card(movie):

    st.markdown(f"### {movie['title']}")

    # Fetch the movie poster
    poster_url = get_movie_poster(
        movie["title"],
        movie["year"]
    )

    # Display the poster
    if poster_url:
        st.image(
            poster_url,
            use_container_width=True
        )
    else:
        st.info("Poster not available")

    # Display movie information
    st.write("Release Year:", int(movie["year"]))

    st.write(
        "⭐ Rating:",
        movie["rating"],
        "/10"
    )

    st.write(
        "Genre:",
        movie["genres"].replace("|", ", ")
    )

    st.write("Description:")
    st.write(movie["overview"])

def recommend_by_title(title, count=5):
    index = movies.index[movies["title"] == title][0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda item: item[1], reverse=True)
    # Skip the selected movie itself.
    results = [i for i, score in scores if i != index][:count]
    return movies.iloc[results]

tab1, tab2, tab3, tab4 = st.tabs(["✨ Recommendations", "🔎 Explore Movies", "🏆 Top Rated", "ℹ️ About Project"])

with tab1:
    st.subheader("Find movies similar to a movie you like")
    selected = st.selectbox("Choose a movie", movies["title"].sort_values().tolist())
    number = st.slider("Number of recommendations", 3, 10, 5)
    if st.button("Recommend Movies", type="primary"):
        results = recommend_by_title(selected, number)
        st.success(f"Because you selected **{selected}**, here are some similar movies:")
        cols = st.columns(3)
        for idx, (_, movie) in enumerate(results.iterrows()):
            with cols[idx % 3]:
                movie_card(movie)

    st.divider()
    st.subheader("Recommendations by genre")
    all_genres = sorted({g for value in movies["genres"] for g in value.split("|") if g})
    chosen_genres = st.multiselect("Select one or more genres", all_genres, default=["Action"])
    min_rating = st.slider("Minimum rating", 0.0, 10.0, 7.0, 0.1)
    if st.button("Find Movies by Genre"):
        filtered = movies[movies["rating"] >= min_rating]
        if chosen_genres:
            filtered = filtered[filtered["genres"].apply(
                lambda value: any(g in value.split("|") for g in chosen_genres)
            )]
        filtered = filtered.sort_values("rating", ascending=False).head(9)
        if filtered.empty:
            st.info("No movies match these filters. Try another genre or rating.")
        else:
            cols = st.columns(3)
            for idx, (_, movie) in enumerate(filtered.iterrows()):
                with cols[idx % 3]:
                    movie_card(movie)

with tab2:
    st.subheader("Explore the movie collection")
    query = st.text_input("Search by movie title, actor, director or keyword")
    genre_filter = st.selectbox("Filter by genre", ["All"] + all_genres)
    view = movies.copy()
    if query:
        searchable = view[["title", "director", "cast", "overview", "keywords"]].astype(str).agg(" ".join, axis=1)
        view = view[searchable.str.contains(query, case=False, na=False)]
    if genre_filter != "All":
        view = view[view["genres"].apply(lambda x: genre_filter in x.split("|"))]
    st.caption(f"{len(view)} movie(s) found")
    if view.empty:
        st.warning("No matching movies found.")
    else:
        cols = st.columns(3)
        for idx, (_, movie) in enumerate(view.head(18).iterrows()):
            with cols[idx % 3]:
                movie_card(movie)

with tab3:
    st.subheader("Highest-rated movies")
    top = movies.sort_values("rating", ascending=False).head(10)
    for rank, (_, movie) in enumerate(top.iterrows(), start=1):
        c1, c2, c3 = st.columns([0.7, 4, 1])
        c1.markdown(f"### #{rank}")
        c2.markdown(f"**{movie['title']} ({int(movie['year'])})**")
        c2.caption(movie["genres"].replace("|", " • "))
        c3.metric("Rating", f"{movie['rating']}/10")

with tab4:
    st.subheader("About CineMatch")
    st.write("CineMatch is a Python-based movie recommendation application. It uses content-based filtering to suggest movies with similar genres, descriptions, keywords, directors and cast members.")
    st.markdown("**How it works**")
    st.markdown("""
    1. Movie information is loaded from a CSV dataset using Pandas.
    2. Important movie details are combined into one text description.
    3. TF-IDF converts the text into numerical vectors.
    4. Cosine Similarity compares the vectors and calculates how similar the movies are.
    5. The system displays the most similar movies, excluding the selected movie.
    """)
    st.info("This is a classroom demonstration dataset containing 30 sample movie records. It can be expanded by adding more rows to movies.csv.")
    st.markdown("**Limitations:** Recommendations depend on the information in the dataset. This version does not learn from individual user watch history or live streaming-platform data.")

st.sidebar.title("CineMatch")
st.sidebar.write("**Python Movie Recommendation System**")
st.sidebar.markdown("---")
st.sidebar.write(f"🎞️ Movies in dataset: **{len(movies)}**")
st.sidebar.write(f"🎭 Genres available: **{len(all_genres)}**")
st.sidebar.markdown("---")
st.sidebar.caption("Built with Python • Pandas • Scikit-learn • Streamlit")
