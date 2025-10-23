# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
import pandas as pd, joblib

def _norm(c:str)->str:
    return c.replace(" ", "_").replace("FCIB5ON", "FCIB50N")  # 防止历史命名差异

def main(fcib_csv, cols_json, model_pkl, out_csv,
         id_col="patient_id", score_col="DFM50N_score"):
    df = pd.read_csv(fcib_csv).copy()
    df.columns = [_norm(c) for c in df.columns]

    kept_cols = json.loads(Path(cols_json).read_text())["feature_cols"]
    missing = [c for c in kept_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")

    X = df[kept_cols].values
    clf = joblib.load(model_pkl)            # 训练好的 RandomForestClassifier（默认参数）
    prob = clf.predict_proba(X)[:, 1]       # class-1 概率即 DFM50N 分数

    out = df[[id_col]].copy(); out[score_col] = prob
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} (n={len(out)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fcib_csv", required=True)
    ap.add_argument("--cols_json", required=True)  # models/dfm50n_rf.cols.json
    ap.add_argument("--model_pkl", required=True)  # models/dfm50n_rf.pkl
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--id_col", default="patient_id")
    args = ap.parse_args()
    main(**vars(args))
