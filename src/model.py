from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
import joblib

@dataclass
class ModelSpec:
    n_clusters: int = 12
    random_state: int = 42

class TrackClusteringModel:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.kmeans = KMeans(n_clusters=spec.n_clusters, n_init='auto', random_state=spec.random_state)

    def fit(self, X: np.ndarray):
        self.kmeans.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.kmeans.predict(X)

    def save(self, model_dir: str | Path):
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.kmeans, Path(model_dir)/'model.joblib')

    def load(self, model_dir: str | Path):
        self.kmeans = joblib.load(Path(model_dir)/'model.joblib')
        return self
