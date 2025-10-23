# -*- coding: utf-8 -*-
"""
Single-patient DFM50N computation (reproducible, publication-ready)

Pipeline:
  NIfTI (CT + mask)
    -> resample to 1 mm isotropic (tricubic for CT; nearest for mask)
    -> crop 50×50×50 cube centered on the lesion (centroid or DT-peak)
    -> FCIB v1.0.0 feature extraction (precropped=True, spatial_size=50)
    -> select 9 FCIB embeddings
    -> RandomForestClassifier (DEFAULT hyperparameters) -> class-1 probability
    -> save results to JSON/CSV + save cropped NIfTI for audit

Usage (Windows PowerShell):
  conda activate dfm50n
  python .\src\single_patient_dfm50n.py `
    --img .\data\PA_demo.nii.gz `
    --msk .\data\PA_demo_mask.nii.gz `
    --model .\models\dfm50n_rf.pkl `
    --cols  .\models\dfm50n_rf.cols.json `
    --out_dir .\outputs\PA_demo `
    --patient_id PA_demo `
    --youden 0.25 `
    --center dt_peak

Notes:
  - This script assumes FCIB (foundation-cancer-image-biomarker)==1.0.0 is installed.
  - RandomForest model was trained with DEFAULT hyperparameters (sklearn 1.4.*).
  - The threshold (Youden=0.25) is dataset-derived; adjust as needed.

Optional (Windows OpenMP conflict quick-fix; uncomment if needed for demo):
  import os
  os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Tuple

import joblib
import nibabel as nib
import numpy as np
import pandas as pd
from fmcib.run import get_features  # foundation-cancer-image-biomarker==1.0.0
from scipy.ndimage import center_of_mass, distance_transform_edt, label, zoom


# ----------------------------- I/O & checks -----------------------------
def _assert_exists(p: Path, desc: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"{desc} not found: {p}")


def _ensure_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------- geometry utils -----------------------------
def _resample_iso(nii: nib.Nifti1Image, order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample to 1 mm isotropic.
    - order=3: tricubic interpolation (CT)
    - order=0: nearest-neighbor (mask)
    Returns:
      data_resampled, affine_1mm
    """
    data = nii.get_fdata()
    sx, sy, sz = nii.header.get_zooms()[:3]
    if min(sx, sy, sz) <= 0:
        raise ValueError(f"Invalid voxel spacing from header: {(sx, sy, sz)}")
    # scipy.ndimage.zoom expects scale factors = output/input spacing
    data_iso = zoom(data, zoom=(1 / sx, 1 / sy, 1 / sz), order=order)
    aff = nii.affine.copy()
    aff[:3, :3] = np.diag([1.0, 1.0, 1.0])  # set voxel size to 1mm
    return data_iso, aff


def _largest_cc(binary: np.ndarray) -> np.ndarray:
    lab, n = label(binary)
    if n <= 1:
        return binary.astype(np.uint8)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    k = counts.argmax()
    return (lab == k).astype(np.uint8)


def _center_xyz(mask_bin: np.ndarray, method: str = "dt_peak") -> Tuple[float, float, float]:
    if method == "centroid":
        c = center_of_mass(mask_bin)
        return float(c[0]), float(c[1]), float(c[2])
    # dt_peak: point farthest from background (robust to irregular shapes)
    dt = distance_transform_edt(mask_bin > 0)
    z, y, x = np.unravel_index(np.argmax(dt), dt.shape)
    return float(z), float(y), float(x)


def _crop_cube(arr: np.ndarray, center_zyx, size: int = 50) -> np.ndarray:
    zc, yc, xc = [int(round(v)) for v in center_zyx]
    half = size // 2
    z0, z1 = zc - half, zc + half
    y0, y1 = yc - half, yc + half
    x0, x1 = xc - half, xc + half
    pad = [
        max(0, -z0),
        max(0, z1 - arr.shape[0]),
        max(0, -y0),
        max(0, y1 - arr.shape[1]),
        max(0, -x0),
        max(0, x1 - arr.shape[2]),
    ]
    if any(pad):
        arr = np.pad(
            arr,
            ((pad[0], pad[1]), (pad[2], pad[3]), (pad[4], pad[5])),
            mode="constant",
        )
        z0 += pad[0]
        z1 += pad[0]
        y0 += pad[2]
        y1 += pad[2]
        x0 += pad[4]
        x1 += pad[4]
    return arr[z0:z1, y0:y1, x0:x1]


# ----------------------------- main pipeline -----------------------------
def run_single_case(
    img_path: Path,
    msk_path: Path,
    model_pkl: Path,
    cols_json: Path,
    out_dir: Path,
    patient_id: str = "CASE001",
    youden: float = 0.25,
    center: str = "dt_peak",
) -> Path:
    """
    Returns:
      Path to JSON result file.
    """
    t0 = time.time()

    # sanity checks
    _assert_exists(img_path, "CT NIfTI")
    _assert_exists(msk_path, "Mask NIfTI")
    _assert_exists(model_pkl, "RandomForest model (.pkl)")
    _assert_exists(cols_json, "feature list (.json)")
    _ensure_outdir(out_dir)

    # 1) load & resample (1 mm)
    img_nii = nib.load(str(img_path))
    msk_nii = nib.load(str(msk_path))
    img_iso, aff = _resample_iso(img_nii, order=3)  # tricubic for CT
    msk_iso, _ = _resample_iso(msk_nii, order=0)    # nearest for mask

    mask_bin = (msk_iso > 0).astype(np.uint8)
    if mask_bin.max() == 0:
        raise ValueError("Mask is empty after resampling. Please check your mask input.")
    mask_cc = _largest_cc(mask_bin)

    # 2) center & crop 50³
    c = _center_xyz(mask_cc, method=center)
    ct_cube = _crop_cube(img_iso, c, size=50)
    msk_cube = _crop_cube(mask_cc, c, size=50)

    # 3) save cropped for audit
    out_ct = out_dir / "case_ct_1mm_50cube.nii.gz"
    out_mk = out_dir / "case_msk_1mm_50cube.nii.gz"
    nib.save(nib.Nifti1Image(ct_cube.astype(np.float32), aff), str(out_ct))
    nib.save(nib.Nifti1Image(msk_cube.astype(np.uint8), aff), str(out_mk))

    # 4) FCIB extraction via temp CSV (precropped = true)
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp_csv = Path(td) / "inputs.csv"
        with open(tmp_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patient_id", "image_path", "mask_path", "precropped"])
            w.writerow([patient_id, str(out_ct), str(out_mk), "true"])
        df_fcib = get_features(
            csv_path=str(tmp_csv),
            spatial_size=(50, 50, 50),
            precropped=True,
            num_workers=0,
            batch_size=1,
        )

    # 5) pick 9 features (order preserved)
    cols = json.loads(cols_json.read_text())["feature_cols"]
    df_fcib.columns = [c.replace(" ", "_").replace("FCIB5ON", "FCIB50N") for c in df_fcib.columns]
    missing = [c for c in cols if c not in df_fcib.columns]
    if missing:
        raise KeyError(f"Missing expected feature columns: {missing}")
    X = df_fcib[cols].values

    # 6) RF probability
    clf = joblib.load(str(model_pkl))
    if not hasattr(clf, "predict_proba"):
        raise TypeError("Loaded model has no predict_proba(). Please provide a probabilistic classifier.")
    prob = float(clf.predict_proba(X)[0, 1])
    label = "high-risk" if prob >= youden else "low-risk"

    # 7) save results JSON & CSV
    res = {
        "patient_id": patient_id,
        "DFM50N_score": round(prob, 6),
        "threshold_youden": float(youden),
        "risk_label": label,
        "center_method": center,
        "cropped_ct": str(out_ct),
        "cropped_mask": str(out_mk),
        "runtime_sec": round(time.time() - t0, 3),
    }
    out_json = out_dir / "dfm50n_result.json"
    out_csv = out_dir / "dfm50n_result.csv"
    out_json.write_text(json.dumps(res, indent=2))
    pd.DataFrame([res]).to_csv(out_csv, index=False)

    print("\n=== Single-patient DFM50N ===")
    print(json.dumps(res, indent=2))
    print(f"\nSaved:\n  {out_json}\n  {out_csv}\n  {out_ct}\n  {out_mk}")
    return out_json


def main():
    p = argparse.ArgumentParser(
        description="Single-patient DFM50N computation (NIfTI -> FCIB v1.0.0 -> 9 feats -> RF -> score)"
    )
    p.add_argument("--img", required=True, help="Path to CT NIfTI (.nii or .nii.gz)")
    p.add_argument("--msk", required=True, help="Path to binary mask NIfTI (same space as CT)")
    p.add_argument("--model", required=True, help="Path to RandomForest .pkl (DEFAULT hyperparameters)")
    p.add_argument("--cols", required=True, help="Path to dfm50n_rf.cols.json (9 FCIB features)")
    p.add_argument("--out_dir", default="outputs/single_case", help="Output directory")
    p.add_argument("--patient_id", default="CASE001", help="Patient ID for outputs")
    p.add_argument("--youden", type=float, default=0.25, help="Risk cutoff (Youden). Default: 0.25")
    p.add_argument("--center", choices=["dt_peak", "centroid"], default="dt_peak", help="Crop center method")
    args = p.parse_args()

    run_single_case(
        img_path=Path(args.img),
        msk_path=Path(args.msk),
        model_pkl=Path(args.model),
        cols_json=Path(args.cols),
        out_dir=Path(args.out_dir),
        patient_id=args.patient_id,
        youden=args.youden,
        center=args.center,
    )


if __name__ == "__main__":
    main()
