🧬 DFM50N-score: Minimal Reproducible Workflow from NIfTI to DFM50N

Purpose
This repository provides a minimal, fully reproducible pipeline to compute the DFM50N score—a foundation-model–based quantitative imaging biomarker for predicting occult lymph-node metastasis (OLM) in resectable cT1–2N0M0 small-cell lung cancer (SCLC).

The workflow runs end-to-end from NIfTI (or DICOM) CT scans with delineated lesion masks to the DFM50N probability score, using the Foundation Cancer Imaging Biomarker (FCIB) model (v1.0.0, Harvard AIM) and a Random Forest classifier trained on nine selected FCIB embeddings.

Data & outputs policy
For privacy and reproducibility policy, the data/ and outputs/ folders are intentionally empty in this public repo. They are kept in Git using placeholder files (e.g., .gitkeep).

Put your own NIfTI files under data/ when running locally.

Runtime results will be written to outputs/.

If you prefer to include small anonymized demos, add data/demo_ct.nii.gz and data/demo_mask.nii.gz yourself.

📁 Repository structure
DFM50N_score/
├─ README.md
├─ environment.yml
├─ data/                         # empty by default (privacy); add your own NIfTI here
│  ├─ demo_ct.nii.gz            
│  └─ demo_mask.nii.gz          
├─ models/
│  ├─ dfm50n_rf.pkl             # RandomForest model (9 FCIB features)
│  ├─ dfm50n_rf.cols.json       # names of the 9 embeddings (order fixed)
│  ├─ thresholds.json           # fixed Youden cutoff (0.25)
│  └─ metadata.json             # version & parameter record
├─ outputs/                     # empty by default; created at runtime
└─ src/
   ├─ single_patient_dfm50n.py  # one-shot NIfTI → FCIB → 9 feats → RF → score
   ├─ step00_crop50cube.py      # 1 mm resample + 50³ cube centered on mask
   ├─ step02_fcib_extract.py    # FCIB feature extraction (v1.0.0)
   └─ step03_dfm50n_score.py    # 9-feature RF probability (DFM50N score)

⚙️ Environment setup
conda env create -f environment.yml
conda activate dfm50n
# Python 3.9; foundation-cancer-image-biomarker == 1.0.0; scikit-learn == 1.4.*; nibabel == 5.*; numpy == 1.26.*

🚀 Quick start (single patient)

If you did not add demo files, replace the paths with your own NIfTI CT and mask under data/.

python src/single_patient_dfm50n.py \
  --img data/demo_ct.nii.gz \
  --msk data/demo_mask.nii.gz \
  --model models/dfm50n_rf.pkl \
  --cols  models/dfm50n_rf.cols.json \
  --youden 0.25 \
  --center dt_peak \
  --out_dir outputs


Windows tip (OpenMP conflict):

set KMP_DUPLICATE_LIB_OK=TRUE && python src\single_patient_dfm50n.py --img data\demo_ct.nii.gz --msk data\demo_mask.nii.gz --model models\dfm50n_rf.pkl --cols models\dfm50n_rf.cols.json --youden 0.25 --center dt_peak --out_dir outputs


Expected outputs:

outputs/dfm50n_result.json

outputs/dfm50n_result.csv

outputs/case_ct_1mm_50cube.nii.gz

outputs/case_msk_1mm_50cube.nii.gz

🔢 DFM50N feature list (9 FCIB embeddings)

models/dfm50n_rf.cols.json

{
  "feature_cols": [
    "pred_1105",
    "pred_2550",
    "pred_2857",
    "pred_3792",
    "pred_3248",
    "pred_3580",
    "pred_2165",
    "pred_93",
    "pred_1790"
  ]
}


Important:

All FCIB features used in this model follow the unified naming pattern pred_<ID> (e.g., pred_1105, pred_2550).

Make sure your FCIB feature table contains these columns; if your FCIB output uses a different prefix (e.g., embedding_), rename columns or adjust dfm50n_rf.cols.json accordingly.

📈 Threshold & interpretation

models/thresholds.json

{
  "dfm50n": {
    "youden": 0.25,
    "note": "Training-derived cutoff; reused unchanged for validation/test/inference."
  }
}


DFM50N ≥ 0.25 → high OLM risk

DFM50N < 0.25 → low OLM risk

🧮 Parameters & Versions
Component	Version	Parameters
Foundation Cancer Imaging Biomarker (FCIB)	1.0.0	Defaults (spatial_size=50, precropped=true)
RandomForestClassifier	scikit-learn 1.4.*	All defaults (n_estimators=100, criterion='gini', max_features='sqrt', etc.)
Preprocessing	—	No imputation or scaling
Threshold	—	Fixed Youden = 0.25
🧰 Common issues & fixes
Problem	Cause	Solution
OpenMP conflict (libomp.dll initialized)	Multiple BLAS runtimes	Add set KMP_DUPLICATE_LIB_OK=TRUE before python
Lambda pickling error (Can't pickle <lambda>)	FCIB multiprocessing on Windows	Force single-process (num_workers=0)
Missing feature columns	FCIB outputs use a different prefix	Ensure feature names start with pred_ (or adjust cols.json)
Large weight file (model_weights.torch)	FCIB auto-downloads pretrained weights	Keep locally; do not commit to GitHub
Timeout (HTTP 504)	Network instability during weight download	Re-run once; weights will be cached locally
🔁 Reproducibility summary

✅ Fixed cube geometry (50×50×50 voxels, 1 mm isotropic)
✅ FCIB v1.0.0, RandomForest default parameters
✅ Unified feature prefix: pred_
✅ No retraining; fixed 9 features + Youden=0.25
✅ Version-pinned (Python 3.9)

🧩 Citation (paper integration)

Methods.
We defined a 50×50×50-voxel cube (1 mm isotropic) centered at the lesion’s centroid to capture the tumor and its immediate peritumoral microenvironment. CT volumes were resampled with tricubic interpolation and masks with nearest-neighbor. The cube was processed using the Foundation Cancer Imaging Biomarker (v1.0.0) to extract image embeddings. Nine FCIB-derived embeddings (prefixed pred_) were retained and used to train a Random Forest classifier (scikit-learn 1.4.*, default hyperparameters). The class-1 probability was defined as the DFM50N score. The operating threshold (Youden = 0.25) was determined once on the training cohort and reused unchanged for validation, test, and deployment.

Code availability.
A fully reproducible pipeline (NIfTI → FCIB v1.0.0 → 9 pred_ features → Random Forest → DFM50N score) is publicly available under the MIT License.

🧠 Acknowledgement

Built upon the Foundation Cancer Imaging Biomarker (FCIB v1.0.0) developed by the Harvard AIM Lab
→ https://github.com/AIM-Harvard/foundation-cancer-image-biomarker

📜 License

MIT License © 2025 Xu Jiang / Peking Union Medical College