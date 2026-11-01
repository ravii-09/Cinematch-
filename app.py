
import streamlit as st
import pandas as pd
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide"
)


# --------------------------------------------------
# LOAD MOVIE DATASET
# --------------------------------------------------

@st.cache_data
def load_movies():

    data = pd.read_csv("movies.csv")

    data["genres"] = data["genres"].fillna("")

    for col in ["overview", "keywords", "director", "cast"]:
        data[col] = data[col].fillna("")

    return data


movies = load_movies()


# --------------------------------------------------
# TMDB API - FETCH MOVIE POSTERS
# --------------------------------------------------

@st.cache_data(ttl=86400)
def get_movie_poster(title, year):

    # Read API token from Streamlit Secrets
    try:
        token = st.secrets.get("TMDB_READ_ACCESS_TOKEN", "")
    except Exception:
        return None

    if not token:
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


# --------------------------------------------------
# BUILD RECOMMENDATION SYSTEM
# --------------------------------------------------

@st.cache_resource
def build_recommender(data):

    data = data.copy()

    data["combined_features"] = (
        data["genres"].str.replace("|", " ", regex=False)
        + " "
        + data["overview"]
        + " "
        + data["keywords"]
        + " "
        + data["director"]
        + " "
        + data["cast"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(
        data["combined_features"]
    )

    similarity = cosine_similarity(
        matrix,
        matrix
    )

    return similarity


similarity = build_recommender(movies)


# --------------------------------------------------
# GENRE LIST
# --------------------------------------------------

all_genres = sorted({

    genre

    for value in movies["genres"]

    for genre in value.split("|")

    if genre

})


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #10131c;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 28px 30px;
    border-radius: 18px;
    background: linear-gradient(120deg, #242b42, #4a2348);
    margin-bottom: 22px;
}

.hero h1 {
    font-size: 2.5rem;
    margin: 0;
    color: white;
}

.hero p {
    color: #e5e7eb;
    margin: 8px 0 0 0;
}

div.stButton > button {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

st.markdown("""
<div class="hero">

    🎬 CineMatch

    Discover your next favourite movie —
    recommendations based on what you enjoy.
    

</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DISPLAY MOVIE DETAILS AND POSTERS
# --------------------------------------------------

def movie_card(movie):

    st.markdown(f"### {movie['title']}")

    poster_url = get_movie_poster(
        movie["title"],
        movie["year"]
    )

    if poster_url:

        st.image(
            poster_url,
            use_container_width=True
        )

    else:

        st.info("Poster not available")

    st.write(
        "Release Year:",
        int(movie["year"])
    )

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


# --------------------------------------------------
# RECOMMEND MOVIES BY TITLE
# --------------------------------------------------

def recommend_by_title(title, count=5):

    index = movies.index[
        movies["title"] == title
    ][0]

    scores = list(
        enumerate(similarity[index])
    )

    scores = sorted(
        scores,
        key=lambda item: item[1],
        reverse=True
    )

    results = [
        i for i, score in scores
        if i != index
    ][:count]

    return movies.iloc[results]


# --------------------------------------------------
# APPLICATION TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([

    "✨ Recommendations",
    "🔎 Explore Movies",
    "🏆 Top Rated",
    "ℹ️ About Project"

])


# --------------------------------------------------
# TAB 1 - RECOMMENDATIONS
# --------------------------------------------------

with tab1:

    st.subheader(
        "Find movies similar to a movie you like"
    )

    selected = st.selectbox(
        "Choose a movie",
        movies["title"].sort_values().tolist()
    )

    number = st.slider(
        "Number of recommendations",
        3, 10, 5
    )

    if st.button(
        "Recommend Movies",
        type="primary"
    ):

        results = recommend_by_title(
            selected,
            number
        )

        st.success(
            f"Because you selected {selected}, "
            "here are some similar movies:"
        )

        cols = st.columns(3)

        for idx, (_, movie) in enumerate(
            results.iterrows()
        ):

            with cols[idx % 3]:

                movie_card(movie)

    st.divider()

    st.subheader(
        "Recommendations by genre"
    )

    default_genres = (
        ["Action"] if "Action" in all_genres else []
    )

    chosen_genres = st.multiselect(
        "Select one or more genres",
        all_genres,
        default=default_genres
    )

    min_rating = st.slider(
        "Minimum rating",
        0.0, 10.0, 7.0, 0.1
    )

    if st.button("Find Movies by Genre"):

        filtered = movies[
            movies["rating"] >= min_rating
        ]

        if chosen_genres:

            filtered = filtered[
                filtered["genres"].apply(
                    lambda value: any(
                        genre in value.split("|")
                        for genre in chosen_genres
                    )
                )
            ]

        filtered = filtered.sort_values(
            "rating",
            ascending=False
        ).head(9)

        if filtered.empty:

            st.info(
                "No movies match these filters. "
                "Try another genre or rating."
            )

        else:

            cols = st.columns(3)

            for idx, (_, movie) in enumerate(
                filtered.iterrows()
            ):

                with cols[idx % 3]:

                    movie_card(movie)


# --------------------------------------------------
# TAB 2 - EXPLORE MOVIES
# --------------------------------------------------

with tab2:

    st.subheader(
        "Explore the movie collection"
    )

    query = st.text_input(
        "Search by movie title, actor, director or keyword"
    )

    genre_filter = st.selectbox(
        "Filter by genre",
        ["All"] + all_genres
    )

    view = movies.copy()

    if query:

        searchable = view[
            [
                "title",
                "director",
                "cast",
                "overview",
                "keywords"
            ]
        ].astype(str).agg(" ".join, axis=1)

        view = view[
            searchable.str.contains(
                query,
                case=False,
                na=False,
                regex=False
            )
        ]

    if genre_filter != "All":

        view = view[
            view["genres"].apply(
                lambda value:
                genre_filter in value.split("|")
            )
        ]

    st.caption(
        f"{len(view)} movie(s) found"
    )

    if view.empty:

        st.warning(
            "No matching movies found."
        )

    else:

        cols = st.columns(3)

        for idx, (_, movie) in enumerate(
            view.head(18).iterrows()
        ):

            with cols[idx % 3]:

                movie_card(movie)


# --------------------------------------------------
# TAB 3 - TOP RATED MOVIES
# --------------------------------------------------

with tab3:

    st.subheader(
        "Highest-rated movies"
    )

    top = movies.sort_values(
        "rating",
        ascending=False
    ).head(10)

    cols = st.columns(3)

    for idx, (_, movie) in enumerate(
        top.iterrows()
    ):

        with cols[idx % 3]:

            movie_card(movie)


# --------------------------------------------------
# TAB 4 - ABOUT PROJECT
# --------------------------------------------------

with tab4:

    st.subheader("About CineMatch")

    st.write(
        "CineMatch is a Python-based movie "
        "recommendation application. It uses "
        "content-based filtering to suggest movies "
        "with similar genres, descriptions, "
        "keywords, directors and cast members."
    )

    st.markdown("**How it works**")

    st.markdown("""
    1. Movie information is loaded from a CSV dataset using Pandas.

    2. Important movie details are combined into one text description.

    3. TF-IDF converts the text into numerical vectors.

    4. Cosine Similarity compares the vectors and calculates how similar the movies are.

    5. The system displays the most similar movies, excluding the selected movie.
    """)

    st.info(
        "This is a classroom demonstration dataset "
        "containing 30 sample movie records. "
        "It can be expanded by adding more rows "
        "to movies.csv."
    )

    st.markdown(
        "**Limitations:** Recommendations depend "
        "on the information in the dataset. "
        "This version does not learn from individual "
        "user watch history or live streaming-platform data."
    )

    st.divider()

    st.markdown("**Movie Poster Credits**")

    st.write(
        "This product uses the TMDB API "
        "but is not endorsed or certified by TMDB."
    )

    st.markdown(
        "[TMDB API Information](https://www.themoviedb.org/)"
    )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("CineMatch")

st.sidebar.write(
    "**Python Movie Recommendation System**"
)

st.sidebar.markdown("---")

st.sidebar.write(
    f"🎞️ Movies in dataset: **{len(movies)}**"
)

st.sidebar.write(
    f"🎭 Genres available: **{len(all_genres)}**"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Built with Python • Pandas • "
    "Scikit-learn • Streamlit • TMDB API"
)
