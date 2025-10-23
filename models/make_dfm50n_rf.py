# -*- coding: utf-8 -*-
"""
Create a DEMO RandomForest model for DFM50N (for single-patient demonstration only).

Usage:
    conda activate dfm50n
    python src/make_demo_dfm50n_rf.py

Outputs:
    - models/dfm50n_rf.pkl
    - models/dfm50n_rf.cols.json
    - models/thresholds.json
"""
from pathlib import Path
import json
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import joblib

FEATURES_9 = [
    "pred_F1105","pred_F2550","pred_F2857",
    "pred_F3792","pred_F3248","pred_F3580",
    "pred_F2165","pred_F93","pred_F1790"
]

def main(out_dir: str = "models", random_state: int = 42) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X, y = make_classification(
        n_samples=400, n_features=9, n_informative=6, n_redundant=0,
        n_clusters_per_class=2, class_sep=1.2, flip_y=0.03, random_state=random_state
    )
    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X, y)
    joblib.dump(clf, out / "dfm50n_rf.pkl")
    (out / "dfm50n_rf.cols.json").write_text(json.dumps({"feature_cols": FEATURES_9}, indent=2))
    (out / "thresholds.json").write_text(json.dumps({"dfm50n": {"youden": 0.25, "note": "demo cutoff"}}, indent=2))
    print("[OK] Demo model saved to models/")

if __name__ == "__main__":
    main()
