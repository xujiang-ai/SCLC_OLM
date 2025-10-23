# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np, nibabel as nib, pandas as pd
from scipy.ndimage import zoom, center_of_mass, distance_transform_edt, label

def _resample_iso(nii, order):
    data = nii.get_fdata()
    sx, sy, sz = nii.header.get_zooms()[:3]
    out = zoom(data, zoom=(1/sx, 1/sy, 1/sz), order=order)
    aff = nii.affine.copy()
    aff[:3, :3] = np.diag([1.0, 1.0, 1.0])
    return out, aff

def _largest_cc(binary):
    lab, n = label(binary)
    if n <= 1: return binary
    counts = np.bincount(lab.ravel()); counts[0] = 0
    k = counts.argmax()
    return (lab == k).astype(np.uint8)

def _center(mask_bin, method="centroid"):
    if method == "centroid":
        return center_of_mass(mask_bin)
    dt = distance_transform_edt(mask_bin)
    return tuple(float(v) for v in np.unravel_index(np.argmax(dt), dt.shape))

def _crop(arr, c, size=50):
    zc, yc, xc = [int(round(v)) for v in c]; h = size//2
    z0, z1 = zc-h, zc+h; y0, y1 = yc-h, yc+h; x0, x1 = xc-h, xc+h
    pad = [max(0,-z0), max(0,z1-arr.shape[0]), max(0,-y0), max(0,y1-arr.shape[1]), max(0,-x0), max(0,x1-arr.shape[2])]
    if any(pad):
        arr = np.pad(arr, ((pad[0],pad[1]),(pad[2],pad[3]),(pad[4],pad[5])), mode="constant")
        z0 += pad[0]; z1 += pad[0]; y0 += pad[2]; y1 += pad[2]; x0 += pad[4]; x1 += pad[4]
    cube = arr[z0:z1, y0:y1, x0:x1]
    return cube, any(pad)

def main(img, msk, out_img, out_msk, out_qc, cube=50, center="dt_peak"):
    img_nii = nib.load(img); msk_nii = nib.load(msk)
    img_iso, aff = _resample_iso(img_nii, order=3)   # tricubic for CT
    msk_iso, _   = _resample_iso(msk_nii, order=0)   # nearest for mask

    mask_bin = (msk_iso > 0).astype(np.uint8)
    if mask_bin.max() == 0: raise ValueError("Empty mask after resampling.")
    mask_cc = _largest_cc(mask_bin)

    c = _center(mask_cc, method=center)
    img_cube, t1 = _crop(img_iso, c, size=cube)
    msk_cube, t2 = _crop(mask_cc, c, size=cube)

    Path(out_img).parent.mkdir(parents=True, exist_ok=True)
    Path(out_msk).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(img_cube.astype(np.float32), aff), out_img)
    nib.save(nib.Nifti1Image(msk_cube.astype(np.uint8),     aff), out_msk)

    cov = msk_cube.sum()/max(1, mask_cc.sum())
    qc = pd.DataFrame([{
        "image_path": img, "mask_path": msk, "center_method": center,
        "mask_voxels_total": int(mask_cc.sum()),
        "mask_voxels_in_cube": int(msk_cube.sum()),
        "mask_coverage_ratio": round(float(cov), 4),
        "touched_border": bool(t1 or t2)
    }])
    Path(out_qc).parent.mkdir(parents=True, exist_ok=True)
    if Path(out_qc).exists(): qc.to_csv(out_qc, mode="a", header=False, index=False)
    else: qc.to_csv(out_qc, index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--msk", required=True)
    ap.add_argument("--out_img", required=True)
    ap.add_argument("--out_msk", required=True)
    ap.add_argument("--out_qc",  required=True)
    ap.add_argument("--cube", type=int, default=50)
    ap.add_argument("--center", choices=["centroid","dt_peak"], default="dt_peak")
    args = ap.parse_args()
    main(**vars(args))
