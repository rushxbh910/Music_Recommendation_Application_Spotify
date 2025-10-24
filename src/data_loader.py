
from pathlib import Path
import pandas as pd

ESSENTIAL_COLS = [
    'title','artist','year','danceability','energy','key','loudness','mode',
    'speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms'
]

def _canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    # Lightweight canonicalization mirror (for direct use without scripts/ingest.py)
    rename_map = {
        'name':'title','track_name':'title','song':'title','track':'title',
        'artists':'artist','artist_name':'artist','track_artist':'artist',
        'release_year':'year'
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
    if 'year' not in df.columns:
        for c in ['release_date','album_release_date']:
            if c in df.columns:
                years = pd.to_datetime(df[c], errors='coerce').dt.year
                df['year'] = years.fillna(0).astype(int)
                break

    for col in ESSENTIAL_COLS:
        if col not in df.columns:
            df[col] = None

    df['title'] = df['title'].astype(str)
    df['artist'] = df['artist'].astype(str)
    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)

    numeric = [c for c in ESSENTIAL_COLS if c not in ['title','artist','year']]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    return df[ESSENTIAL_COLS]

def load_tracks_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Try strict check
    missing = [c for c in ESSENTIAL_COLS if c not in df.columns]
    if missing:
        df = _canonicalize(df)
    # final check
    missing = [c for c in ESSENTIAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns even after canonicalization: {missing}")
    df['title'] = df['title'].astype(str).str.strip()
    df['artist'] = df['artist'].astype(str).str.strip()
    df['year'] = df['year'].astype(int)
    return df
