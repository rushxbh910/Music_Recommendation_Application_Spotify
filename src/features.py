from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

FEATURE_COLS = [
    'danceability','energy','key','loudness','mode','speechiness',
    'acousticness','instrumentalness','liveness','valence','tempo','duration_ms'
]

@dataclass
class FeatureSpec:
    use_pca: bool = True
    pca_components: int = 8

class FeaturePipeline:
    def __init__(self, spec: FeatureSpec):
        self.spec = spec
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=spec.pca_components) if spec.use_pca else None

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df[FEATURE_COLS].values
        Xs = self.scaler.fit_transform(X)
        if self.pca:
            return self.pca.fit_transform(Xs)
        return Xs

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df[FEATURE_COLS].values
        Xs = self.scaler.transform(X)
        if self.pca:
            return self.pca.transform(Xs)
        return Xs
