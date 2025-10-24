
import os, sys
# Ensure project root is on sys.path when executing from anywhere
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
from pathlib import Path
import json
import pandas as pd
from src.data_loader import load_tracks_csv
from src.features import FeaturePipeline, FeatureSpec, FEATURE_COLS
from src.model import TrackClusteringModel, ModelSpec
import joblib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True, help='Path to CSV with track features')
    ap.add_argument('--model_dir', type=str, default='models', help='Where to save model + transformers')
    ap.add_argument('--n_clusters', type=int, default=12)
    ap.add_argument('--pca_components', type=int, default=8)
    ap.add_argument('--no_pca', action='store_true', help='Disable PCA')
    args = ap.parse_args()

    df = load_tracks_csv(args.data)

    spec = FeatureSpec(use_pca=not args.no_pca, pca_components=args.pca_components)
    feat = FeaturePipeline(spec)
    X = feat.fit_transform(df)

    model = TrackClusteringModel(ModelSpec(n_clusters=args.n_clusters))
    model.fit(X)

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(feat.scaler, Path(args.model_dir)/'scaler.joblib')
    if feat.pca is not None:
        joblib.dump(feat.pca, Path(args.model_dir)/'pca.joblib')
    model.save(args.model_dir)

    # Save feature spec (for info only)
    feature_spec = {
        'use_pca': spec.use_pca,
        'pca_components': spec.pca_components,
        'feature_columns': FEATURE_COLS,
        'n_clusters': args.n_clusters
    }
    (Path(args.model_dir)/'feature_spec.json').write_text(json.dumps(feature_spec, indent=2))
    print('✅ Trained & saved artifacts to', args.model_dir)

if __name__ == '__main__':
    main()
