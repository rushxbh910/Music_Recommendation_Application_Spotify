# 🎵 Spotify Music Recommendation Application

A modular, production-ready, end‑to‑end data prep, model training, and real‑time recommendations via **FastAPI** and an optional **Streamlit** UI. 
It uses feature scaling, dimensionality reduction (PCA), **KMeans** clustering, and **cosine similarity** to recommend tracks. 
If a track isn’t found locally, it can fall back to the **Spotify Web API** (via Spotipy) when credentials are present.

## 🚀 Quickstart

```bash
# 1) Create a virtual env (optional)
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) Add Spotify credentials
cp .env.example .env
# set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env

# 4) Train model on the sample dataset
python scripts/train.py --data data/sample_tracks.csv --model_dir models

# 5) Launch API
uvicorn api.main:app --reload

# 6) (Optional) Streamlit UI
streamlit run app/streamlit_app.py
```

## 📁 Project Structure

```
spotify_recommender_project/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   └── sample_tracks.csv
├── models/
│   ├── model.joblib
│   └── feature_spec.json
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── features.py
│   ├── model.py
│   ├── recommender.py
│   ├── spotify_api.py
│   └── utils.py
├── api/
│   └── main.py
├── app/
│   └── streamlit_app.py
├── scripts/
│   ├── train.py
│   └── build_index.py
├── tests/
│   └── test_recommender.py
└── notebooks/
    └── Spotify_Music_Recommendation_Application.ipynb
```

## 🧠 How it works (high-level)

1. **Ingest**: Load CSV of track audio features (can be extended to multiple files).
2. **Transform**: Standardize features → optional PCA.
3. **Cluster**: Fit **KMeans** to learn coarse similarity buckets.
4. **Index**: Persist the scaler, PCA, KMeans for reuse; precompute normalized vectors.
5. **Recommend**: For one or more seed songs, compute a seed embedding and return the **top‑K nearest** by cosine similarity (optionally respecting clusters).
6. **Fallback to Spotify**: If seed not found locally and credentials exist, query Spotify API for audio features.

## 🧪 Sample request

```
GET http://localhost:8000/recommend?title=Antidote&year=2015&k=5
```

## 🛡️ Notes

- The sample data is tiny (for demo only). Replace `data/sample_tracks.csv` with your dataset.
- The API and UI will automatically use saved artifacts from `models/` if present.
- Put your notebook into `notebooks/` for reference (not required to run the app).


---

## 🧩 Using your uploaded datasets

CSVs added to `data/`:

- `data.csv`, `data_by_artist.csv`, `data_by_genres.csv`, `data_by_year.csv`, `data_w_genres.csv`

You can either:
1) **Pick any CSV directly in the Streamlit UI**, or
2) **Normalize/merge multiple files** into one canonical file:
   ```bash
   python scripts/ingest.py --inputs data/data.csv data/data_by_artist.csv data/data_by_genres.csv data/data_by_year.csv data/data_w_genres.csv \
                           --output data/tracks_canonical.csv
   python scripts/train.py --data data/tracks_canonical.csv --model_dir models
   ```
The ingestion step tolerates different column names and tries to canonicalize to the schema used by this app.
