# CineMatch - Movie Recommendation System

A beginner-friendly BCA Python project with a Streamlit interface and content-based movie recommendations.

## Features
- Recommend similar movies from a selected title
- Browse/search movies by title, actor, director or keyword
- Filter movies by genre and minimum rating
- View top-rated movies
- Movie descriptions, genres, release years and ratings

## Requirements
- Python 3.9 or newer
- Internet connection for the first-time package installation

## Setup and run

1. Extract the ZIP file.
2. Open the extracted folder in VS Code or another terminal.
3. Create and activate a virtual environment (optional but recommended).

   Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Install libraries:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:
   ```bash
   streamlit run app.py
   ```

6. Streamlit will open the application in your browser. If it does not, open the local URL shown in the terminal (usually http://localhost:8501).

## Project files
- `app.py` - main application and recommendation logic
- `movies.csv` - sample movie dataset
- `requirements.txt` - required Python libraries

## How recommendation works
The program combines genres, overview, keywords, director and cast into a text profile for every movie. `TfidfVectorizer` converts these profiles into TF-IDF vectors. `cosine_similarity` calculates similarity between every pair. When a user selects a movie, the program sorts other movies by similarity and displays the closest matches.

## How to add movies
Add a new row to `movies.csv`. Keep the same columns:
`movie_id,title,year,genres,rating,director,cast,overview,keywords`

Separate multiple genres, cast members and keywords with `|`. Keep the movie ID unique.

## Important note
The included 30-row dataset is sample data for a classroom demonstration, not a complete or live movie database. Ratings and descriptions are provided as sample project data. Expand or replace the CSV with a dataset your teacher approves before claiming the system covers all movies.
