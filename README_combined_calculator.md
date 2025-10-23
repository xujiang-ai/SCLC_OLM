# SCLC OLM Calculator

**Web-based tool for predicting occult lymph node metastasis (OLM) in early-stage small-cell lung cancer (SCLC).**

---

## 🔎 Background
Occult lymph node metastasis (OLM) occurs in about one-third of early-stage SCLC patients but is often missed on routine imaging.  
This calculator implements a combined logistic regression model that integrates:
- **Clinical predictors**: smoking history, short-axis diameter, spiculation  
- **DFM50N score**: extracted from a cancer-specific foundation deep learning model  

The model was developed and validated in a multicenter study, aiming to support individualized preoperative nodal risk estimation.

---

## 🌐 Online Access
- Homepage: [GitHub Pages Link](https://YOUR_USERNAME.github.io/SCLC_OLM_calculator/)  
- Calculator: [Direct Link](https://YOUR_USERNAME.github.io/SCLC_OLM_calculator/calculator.html)  

*(replace `YOUR_USERNAME` with your GitHub account name)*

---

## 📖 Usage
1. Open the calculator in a browser.  
2. Enter patient-specific values:  
   - Smoking history (Yes/No)  
   - Tumor short-axis diameter (cm)  
   - Spiculation (Yes/No)  
   - DFM50N score  
3. Click **Calculate** → Outputs:  
   - Predicted probability of OLM (%)  
   - Risk classification (OLM+ / OLM−, default threshold = **0.254**, Youden index)

---

## 📊 Model Formula
Logit(p) = −1.34614 − 0.60818·Smoke1 − 0.33883·Perpendicular − 0.19792·Spiculated1 + 6.22427·DFM50N_score  

p = 1 / (1 + exp(−Logit(p)))  

---

## ⚠ Disclaimer
This tool is provided for **research purposes only**.  
It is not validated for clinical use. Clinical decisions should not rely on this calculator without further prospective validation.

---

## 📚 Citation
If you use this tool in your work, please cite:  

> Jiang X, et al. *Development and validation of a foundation model–based calculator for occult lymph node metastasis prediction in early-stage SCLC*. (Manuscript in preparation / under review, 2025).  
