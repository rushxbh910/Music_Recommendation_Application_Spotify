
import streamlit as st
from pathlib import Path
from src.recommender import load_artifacts, recommend

st.set_page_config(page_title="Spotify Recommender", layout="centered")

st.title("🎵 Spotify Music Recommender")
st.caption("Choose a dataset, type a song and optional year; get top‑K similar tracks by audio features.")

data_dir = Path("data")
options = sorted([p.name for p in data_dir.glob("*.csv")])
chosen = st.selectbox("Dataset", options, index=options.index("sample_tracks.csv") if "sample_tracks.csv" in options else 0)

art = load_artifacts(model_dir=Path("models"), data_path=data_dir/chosen)

with st.form("rec_form"):
    title = st.text_input("Song title", value="Antidote")
    year = st.number_input("Year (optional)", value=2015, step=1)
    k = st.slider("Top‑K", 1, 20, 5)
    submit = st.form_submit_button("Recommend")

if submit:
    seeds = [{"title": title, "year": int(year)}] if title else []
    recs = recommend(art, seeds, k=k)
    if not recs:
        st.warning("No recommendations found (check title/year or add Spotify credentials).")
    else:
        st.subheader("Recommendations")
        for r in recs:
            st.write(f"**{r['title']}** — {r['artist']} ({int(r['year'])})  ")
            st.text(f"score: {r['score']:.4f}")
