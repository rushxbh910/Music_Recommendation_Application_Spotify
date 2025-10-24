from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
from src.recommender import load_artifacts, recommend

app = FastAPI(title="Spotify Recommender API", version="1.0.0")

# Lazy init (loads on first request)
ART = None

class Seed(BaseModel):
    title: str
    year: Optional[int]=None

def get_artifacts():
    global ART
    if ART is None:
        ART = load_artifacts(model_dir=Path("models"), data_path=Path("data/sample_tracks.csv"))
    return ART

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/recommend")
def recommend_endpoint(title: str = Query(...), year: Optional[int] = Query(None), k: int = Query(10)):
    art = get_artifacts()
    seeds = [{"title": title, "year": year}] if title else []
    recs = recommend(art, seeds, k=k)
    return {"results": recs}
