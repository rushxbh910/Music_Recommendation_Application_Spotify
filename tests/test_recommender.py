from pathlib import Path
from src.recommender import load_artifacts, recommend

def test_basic_recommendations():
    art = load_artifacts(model_dir=Path("models"), data_path=Path("data/sample_tracks.csv"))
    recs = recommend(art, [{"title": "Antidote", "year": 2015}], k=3)
    assert isinstance(recs, list)
    assert 0 < len(recs) <= 3
