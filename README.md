# Predicting US Traffic Fatalities: A Machine Learning Approach to Highway Safety

A supervised machine learning project applying Logistic Regression and Random Forest classifiers to the NHTSA Fatality Analysis Reporting System (FARS) dataset to predict which drivers involved in fatal crashes are themselves fatally injured, and to identify the conditions most strongly associated with driver deaths.

---

## Project Overview

Traffic fatalities represent one of the most significant and preventable public health challenges in the United States, claiming over 38,000 lives annually. This project leverages the FARS-derived US Traffic Fatality Records dataset (2015–2016) to build and evaluate predictive classification models capable of identifying high-risk crash conditions based on driver demographics, environmental factors, road characteristics, and behavioral variables.

The project follows the full CRISP-DM lifecycle — from business understanding and data profiling through feature engineering, modeling, and SHAP-based interpretability analysis — and culminates in evidence-based policy recommendations for highway safety planners.

---

## Status: data-coding fixes (November 2026)

After the original submission, several data-coding problems were found and fixed in the notebooks:

- **Alcohol:** `ALC_RES` is stored in thousandths with 995–999 as status codes. The original `ALC_RES >= 0.08` check flagged nearly every driver with any reading — including "test not given" — as alcohol-positive. Now BAC ≥ 0.08 means `80 <= ALC_RES <= 940`.
- **Rural/urban:** now taken from `RUR_URB`. The original flag came from `FUNC_SYS`, which is road class (Interstate … local), not rural/urban.
- **Leakage:** `FATALS` (deaths in the crash) and `ALC_TESTED` are only known after the outcome and were removed as model inputs.
- **Unknown codes:** "not reported" / "unknown" codes (98, 99, etc.) are now treated as missing instead of as real values. `INJ_SEV` 6 and 9 are dropped instead of counted as non-fatal.
- **Logistic regression** now one-hot encodes category codes (state, weather, restraint use…).
- **SHAP** notebook works with current `shap` versions.

**The results, SHAP ranking and findings below are from the original May 2026 submission and have not yet been re-run with these fixes.** Expect them to change.

---

## Key Results (original submission)

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 75.87% | **79.82%** |
| Precision | 74.32% | **77.44%** |
| Recall | 71.34% | **78.17%** |
| F1 Score | 0.7280 | **0.7781** |
| AUC-ROC | 0.8339 | **0.8750** |

The Random Forest model was designated champion, outperforming Logistic Regression on every evaluation metric. The temporal train-validation split (2015 training / 2016 validation) was chosen to simulate real-world generalization to future crash data.

---

## SHAP Feature Importance (original submission)

SHAP (SHapley Additive exPlanations) values were computed on a 2,000-record sample of the 2016 validation dataset using the TreeExplainer method to provide a theoretically rigorous measure of individual feature contributions.

| Rank | Feature | Mean \|SHAP Value\| |
|---|---|---|
| 1 | Restraint Use | 0.1845 |
| 2 | Alcohol Positive (BAC ≥ 0.08) | 0.0988 |
| 3 | Drunk Driver in Crash | 0.0461 |
| 4 | Manner of Collision | 0.0448 |
| 5 | Driver Age | 0.0409 |
| 6 | Rural/Urban Setting | 0.0329 |
| 7 | Hour of Crash | 0.0174 |
| 8 | Total Crash Fatalities | 0.0115 |
| 9 | Road Functional Class | 0.0115 |
| 10 | State | 0.0106 |

Restraint use was the dominant predictor with a SHAP value nearly double that of the second-ranked feature, confirming that seatbelt compliance is the single highest-impact modifiable risk factor for fatal crash outcomes.

---

## Dataset

- **Source:** [NHTSA FARS downloads](https://www.nhtsa.gov/file-downloads?p=nhtsa/downloads/FARS/) (also mirrored on [Kaggle](https://www.kaggle.com/datasets/usdot/nhtsa-traffic-fatalities)); see `data/README.md` for setup
- **Coverage:** 2015–2016 FARS fatal crash census — every record comes from a crash in which at least one person died
- **Records (original run):** 101,562 driver-level records after joining the accident and person tables and keeping drivers; slightly fewer after dropping unknown injury severity
- **Train/Validation Split:** 2015 / 2016 — temporal split to simulate predicting a future year
- **Target Variable:** Was the driver fatally injured? (INJ_SEV 4 = 1, other known levels = 0; codes 6 and 9 dropped)
- **License:** US Government Open Data

---

## Technical Stack

- **Language:** Python 3.x
- **Environment:** Jupyter Notebook
- **Libraries:** pandas, NumPy, scikit-learn, matplotlib, SHAP (see `requirements.txt`)
- **Models:** Logistic Regression (one-hot encoded categories, LBFGS solver, L2 regularization), Random Forest (200 trees, max depth 12)
- **Interpretability:** SHAP TreeExplainer

---

## Repository Structure

```
traffic-fatality-prediction/
│
├── data/
│   ├── README.md               # Where to download FARS and how to name the files
│   └── raw/                    # Raw FARS CSVs go here (not in git)
│
├── notebooks/
│   ├── 01_data_profiling.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_visualization.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_shap_analysis.ipynb
│
├── outputs/
│   └── figures/                # Charts, created when the notebooks run (not in git)
│
├── reports/
│   └── final_report.pdf        # Full capstone report (original submission)
│
├── requirements.txt
└── README.md
```

---

## How to Run

1. Install Python 3.10+ and the libraries: `pip install -r requirements.txt`
2. Download the FARS 2015 and 2016 files into `data/raw/` as described in `data/README.md`.
3. Open Jupyter from the `notebooks/` folder (`jupyter notebook`) and run the notebooks in order, 01 through 05. Notebook 02 creates the prepared datasets the later notebooks use.

---

## Key Findings (original submission — to be re-checked after the fixes above)

1. **Restraint use is the single most important modifiable risk factor.** Its SHAP value of 0.1845 dwarfs every other predictor, confirming that seatbelt compliance programs have the highest return on investment for highway safety agencies.

2. **Alcohol involvement is the second most important predictor cluster.** Combined SHAP contribution of 0.1449 across two alcohol features; approximately 26% of crashes involved at least one legally impaired driver. *(The original alcohol feature was mis-coded; see Status above.)*

3. **Geographic concentration is extreme.** The top five states (TX, CA, FL, GA, NC) account for over 36% of fatally injured drivers, providing a clear basis for risk-stratified federal safety investment.

4. **Temporal risk is non-uniform.** Fatal crashes peak between 3–5 PM and again midnight–3 AM, supporting time-targeted enforcement deployment strategies.

5. **Rural settings carry elevated risk** independent of other factors (SHAP: 0.0329), reflecting higher speeds, fewer divided highways, and longer EMS response times. *(The original rural flag was derived from road class, not the rural/urban field; see Status above.)*

---

## Policy Recommendations (original submission)

- Prioritize federal advocacy for primary enforcement seatbelt laws in the 17 states currently limited to secondary enforcement
- Deploy DUI checkpoints concentrated in the midnight–3 AM window rather than distributed uniformly across overnight hours
- Concentrate federal highway safety investment in the five highest-burden states
- Expand ignition interlock programs targeting repeat alcohol-impaired offenders
- Evaluate rideshare subsidy programs timed to overnight high-risk windows

---

## Course Context

This project was completed as the capstone for **DATA 495: Data Science Capstone** at the University of Maryland Global Campus (UMGC), May 2026.

---

## References

Full references available in the project report. Key sources include:

- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
- Chen et al. (2015). A multinomial logit model-Bayesian network hybrid approach for driver injury severity analyses. *Accident Analysis & Prevention, 80*, 76–88.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30.*
- National Highway Traffic Safety Administration. (2022). FARS Analytical User's Manual, 1975–2021.
