# -*- coding: utf-8 -*-
import argparse, pandas as pd
from fmcib.run import get_features  # foundation-cancer-image-biomarker==1.0.0

def main(csv_path, out_csv, spatial_size=50, precropped=False):
    df = get_features(
        csv_path=csv_path,
        spatial_size=(spatial_size, spatial_size, spatial_size),
        precropped=precropped
    )
    df.to_csv(out_csv, index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=50)
    ap.add_argument("--precropped", action="store_true")
    args = ap.parse_args()
    main(args.csv, args.out, args.size, args.precropped)
