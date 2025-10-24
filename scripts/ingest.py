
import os, sys
# Ensure project root is on sys.path when executing from anywhere
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import argparse
from pathlib import Path
import pandas as pd

REQUIRED = [
    'title','artist','year','danceability','energy','key','loudness','mode',
    'speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms'
]

MAPPINGS = [
    # common Spotify/Kaggle schemas -> our canonical names
    {
        'title': ['title','name','track_name','song','track'],
        'artist': ['artist','artists','artist_name','track_artist'],
        'year': ['year','release_year'],
        'danceability': ['danceability'],
        'energy': ['energy'],
        'key': ['key'],
        'loudness': ['loudness'],
        'mode': ['mode'],
        'speechiness': ['speechiness'],
        'acousticness': ['acousticness'],
        'instrumentalness': ['instrumentalness'],
        'liveness': ['liveness'],
        'valence': ['valence'],
        'tempo': ['tempo'],
        'duration_ms': ['duration_ms','duration','duration_ms.1','duration_ms_x','duration_ms_y']
    }
]

def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    # build rename dict by first match in candidates
    rename = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for target, candidates in MAPPINGS[0].items():
        for cand in candidates:
            if cand.lower() in lower_cols:
                rename[lower_cols[cand.lower()]] = target
                break
    df = df.rename(columns=rename)

    # derive year from release_date if needed
    if 'year' not in df.columns:
        for c in ['release_date','album_release_date']:
            if c in df.columns:
                try:
                    years = pd.to_datetime(df[c], errors='coerce').dt.year
                    df['year'] = years.fillna(0).astype(int)
                    break
                except Exception:
                    pass

    # ensure present cols; drop rows missing many essentials
    for col in REQUIRED:
        if col not in df.columns:
            df[col] = None

    # coerce types
    df['title'] = df['title'].astype(str)
    df['artist'] = df['artist'].astype(str)
    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)

    num_cols = [c for c in REQUIRED if c not in ['title','artist','year']]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # basic cleaning
    df = df.dropna(subset=['title','artist'])
    # keep only required columns in order
    df = df[REQUIRED]
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True, help='One or more CSV paths')
    ap.add_argument('--output', required=True, help='Output CSV path (canonical schema)')
    args = ap.parse_args()

    frames = []
    for p in args.inputs:
        df = pd.read_csv(p)
        frames.append(canonicalize(df))
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"✅ Wrote canonical dataset with {len(out):,} rows to {args.output}")

if __name__ == "__main__":
    main()
