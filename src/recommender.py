from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from .data_loader import load_tracks_csv
from .features import FeaturePipeline, FeatureSpec, FEATURE_COLS
from .model import TrackClusteringModel, ModelSpec
from .utils import cosine_similarity_matrix, topk_indices
from . import spotify_api

@dataclass
class Artifacts:
    feature_pipeline: FeaturePipeline
    model: TrackClusteringModel
    data: pd.DataFrame
    embeddings: np.ndarray  # normalized feature vectors for retrieval

def _normalize_rows(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

def load_artifacts(model_dir: str | Path, data_path: str | Path) -> Artifacts:
    # load data
    df = load_tracks_csv(data_path)

    # load feature pipeline and model
    spec = FeatureSpec()
    feat = FeaturePipeline(spec)
    # Try to load fitted scaler/PCA if present
    scaler_p = Path(model_dir)/'scaler.joblib'
    pca_p = Path(model_dir)/'pca.joblib'
    if scaler_p.exists():
        feat.scaler = joblib.load(scaler_p)
    if pca_p.exists() and feat.pca is not None:
        feat.pca = joblib.load(pca_p)
    X = feat.transform(df) if scaler_p.exists() else feat.fit_transform(df)

    # model
    model = TrackClusteringModel(ModelSpec())
    model_path = Path(model_dir)/'model.joblib'
    if model_path.exists():
        model.load(model_dir)
    else:
        model.fit(X)
        model.save(model_dir)

    # save/update transformers
    joblib.dump(feat.scaler, Path(model_dir)/'scaler.joblib')
    if feat.pca is not None:
        joblib.dump(feat.pca, Path(model_dir)/'pca.joblib')

    # precompute normalized embeddings for fast cosine similarity
    embeddings = _normalize_rows(X)
    return Artifacts(feature_pipeline=feat, model=model, data=df, embeddings=embeddings)

def _search_local(df: pd.DataFrame, title: str, year: Optional[int]=None) -> Optional[int]:
    title_norm = title.strip().lower()
    cand = df.index[df['title'].str.lower() == title_norm]
    if year is not None:
        cand = [i for i in cand if int(df.loc[i, 'year']) == int(year)]
    if len(cand) > 0:
        return int(cand[0])
    return None

def _fetch_from_spotify(title: str, year: Optional[int]=None) -> Optional[Dict]:
    sp = spotify_api.get_client()
    if sp is None:
        return None
    query = f"track:{title} year:{year}" if year else f"track:{title}"
    res = sp.search(q=query, type='track', limit=1)
    items = res.get('tracks', {}).get('items', [])
    if not items:
        return None
    track = items[0]
    # Fetch audio features
    feats = sp.audio_features([track['id']])[0]
    if feats is None:
        return None
    # Map into our schema
    mapped = { 
        'title': track['name'],
        'artist': ', '.join([a['name'] for a in track['artists']]),
        'year': int(track['album']['release_date'][:4]) if track['album']['release_date'] else 0,
        'danceability': feats['danceability'],
        'energy': feats['energy'],
        'key': feats['key'],
        'loudness': feats['loudness'],
        'mode': feats['mode'],
        'speechiness': feats['speechiness'],
        'acousticness': feats['acousticness'],
        'instrumentalness': feats['instrumentalness'],
        'liveness': feats['liveness'],
        'valence': feats['valence'],
        'tempo': feats['tempo'],
        'duration_ms': feats['duration_ms'] if 'duration_ms' in feats and feats['duration_ms'] else 0
    }
    return mapped

def recommend(art: Artifacts, seeds: List[Dict], k: int=10) -> List[Dict]:
    """Recommend top‑k tracks given seed dicts like {{'title': 'Antidote', 'year': 2015}}."""
    # Build seed vectors
    seed_vecs = []
    for s in seeds:
        idx = _search_local(art.data, s.get('title',''), s.get('year'))
        if idx is None:
            # fallback to Spotify (optional)
            fetched = _fetch_from_spotify(s.get('title',''), s.get('year'))
            if fetched is None:
                continue
            # turn into 1-row DataFrame for transform
            import pandas as pd
            row = pd.DataFrame([fetched])
            X = art.feature_pipeline.transform(row)
            seed_vecs.append(X[0])
        else:
            seed_vecs.append(art.embeddings[idx])  # already normalized

    if not seed_vecs:
        return []

    seed_vecs = np.vstack(seed_vecs)
    # if any seeds came from Spotify transform, normalize them
    seed_vecs = seed_vecs / (np.linalg.norm(seed_vecs, axis=1, keepdims=True) + 1e-12)

    # average the seed vectors
    centroid = seed_vecs.mean(axis=0, keepdims=True)  # 1xd
    sims = (centroid @ art.embeddings.T).ravel()      # cosine similarity to all

    # remove any exact seed matches by title+year
    seed_keys = {(s.get('title','').strip().lower(), int(s.get('year',0))) for s in seeds if 'title' in s and 'year' in s}
    mask = []
    for i, row in art.data.iterrows():
        key = (row['title'].strip().lower(), int(row['year']))
        mask.append(key not in seed_keys)
    sims = np.where(mask, sims, -1.0)

    idxs = np.argsort(-sims)[:k]
    out = []
    for i in idxs:
        r = art.data.iloc[i].to_dict()
        r['score'] = float(sims[i])
        out.append(r)
    return out
