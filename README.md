# Predicting US Traffic Fatalities: A Machine Learning Approach to Highway Safety

A supervised machine learning project applying Logistic Regression and Random Forest classifiers to the NHTSA Fatality Analysis Reporting System (FARS) dataset to predict which drivers involved in fatal crashes are themselves fatally injured, and to identify the conditions most strongly associated with driver deaths.

---

## Project Overview

Traffic fatalities represent one of the most significant and preventable public health challenges in the United States, claiming over 38,000 lives annually. This project leverages the FARS-derived US Traffic Fatality Records dataset (2015–2016) to build and evaluate predictive classification models capable of identifying high-risk crash conditions based on driver demographics, environmental factors, road characteristics, and behavioral variables.

The project follows the full CRISP-DM lifecycle — from business understanding and data profiling through feature engineering, modeling, and SHAP-based interpretability analysis — and culminates in evidence-based policy recommendations for highway safety planners.

---

## Results (corrected, November 2026)

The models were re-run on the official NHTSA FARS 2015–2016 files after fixing several data-coding problems (see Corrections below). Training: 48,533 drivers from 2015. Validation: 51,195 drivers from 2016.

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 76.72% | **77.20%** |
| Precision | **78.06%** | 77.05% |
| Recall | 69.18% | **72.30%** |
| F1 Score | 0.7335 | **0.7460** |
| AUC-ROC | 0.8292 | **0.8434** |

Random Forest remains the champion: it is better on accuracy, recall, F1 and AUC, while Logistic Regression is slightly more precise. The gap between the two models is smaller than originally reported (AUC +0.014 instead of +0.041), and both scores are lower than the original run (Random Forest AUC 0.843 vs. 0.875), mainly because the original run used `FATALS`, a feature that partly revealed the answer.

---

## SHAP Feature Importance (corrected)

SHAP (SHapley Additive exPlanations) values were computed on a 2,000-record random sample of the 2016 validation set using TreeExplainer.

| Rank | Feature | Mean \|SHAP Value\| | Original rank (value) |
|---|---|---|---|
| 1 | Restraint Use | 0.1858 | 1 (0.1845) |
| 2 | Driver Age | 0.0484 | 5 (0.0409) |
| 3 | Alcohol Positive (BAC ≥ 0.08) | 0.0461 | 2 (0.0988) |
| 4 | Rural/Urban Setting | 0.0433 | 6 (0.0329) |
| 5 | Manner of Collision | 0.0420 | 4 (0.0448) |
| 6 | Hour of Crash | 0.0208 | 7 (0.0174) |
| 7 | Drunk Drivers in Crash | 0.0126 | 3 (0.0461) |
| 8 | Light Condition | 0.0097 | — |
| 9 | Road Functional Class | 0.0083 | 9 (0.0115) |
| 10 | State | 0.0071 | 10 (0.0106) |

Restraint use is still the dominant predictor — now nearly four times the second-ranked feature. Alcohol's importance roughly halved once the feature was coded correctly, and rural/urban setting moved up once it came from the actual rural/urban field.

---

## Corrections (November 2026)

| Problem in the original | Fix | Effect |
|---|---|---|
| `ALC_RES` is stored in thousandths, with 995–999 as status codes (996 = "test not given"). `ALC_RES >= 0.08` flagged any reading or status code as positive. | Positive means `80 <= ALC_RES <= 940`. | Drivers flagged as BAC ≥ 0.08 fell from **68.0%** to **13.3%**. |
| Rural/urban was derived from `FUNC_SYS`, which is road class (Interstate … local). | Use `RUR_URB` (1 = rural, 2 = urban). | Rural/urban is now a real rural/urban flag. |
| `FATALS` (deaths in the crash) and `ALC_TESTED` were model inputs, but both are known only after the outcome. Killed drivers were BAC-tested 74.7% of the time vs. 28.8% for survivors. | Removed as model inputs (kept for charts). | Lower, more honest scores. |
| "Unknown" / "not reported" codes (8/9, 98/99, 96–99) were used as real numbers; `INJ_SEV` 6 and 9 counted as non-fatal. | Treated as missing; categorical unknowns get their own category; `INJ_SEV` 6/9 dropped (1,834 drivers). | Cleaner categories. |
| The person file repeats crash-level columns, so merging created `HOUR_x` / `HOUR_y` duplicates. | Merge only accident-only columns. | Notebooks run on the official files. |
| Logistic regression treated state, weather, etc. as numbers. | One-hot encoding for categories. | Fairer baseline. |

The original May 2026 results are kept in git history and in `reports/final_report.pdf`; that report has **not** been updated.

---

## Dataset

- **Source:** [NHTSA FARS downloads](https://www.nhtsa.gov/file-downloads?p=nhtsa/downloads/FARS/) (also mirrored on [Kaggle](https://www.kaggle.com/datasets/usdot/nhtsa-traffic-fatalities)); see `data/README.md` for setup
- **Coverage:** 2015–2016 FARS fatal crash census — every record comes from a crash in which at least one person died
- **Records:** 101,562 drivers after joining the accident and person tables; 99,728 after dropping unknown injury severity (2015: 48,533 / 2016: 51,195)
- **Train/Validation Split:** 2015 / 2016 — temporal split to simulate predicting a future year
- **Fatal rate:** 46.2% of drivers in the data were fatally injured
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

The notebooks are saved with their outputs from the November 2026 run, so results and charts can be read on GitHub without running anything.

---

## Key Findings (corrected)

1. **Restraint use is the strongest predictor by far.** Mean |SHAP| 0.1858, nearly four times the next feature. Of drivers wearing a lap-and-shoulder belt, 27.7% were killed, compared with 78.4% of drivers using no restraint. Drivers with unknown restraint status were in between (40.9%), so the effect comes from known non-use, not from missing data. Motorcyclists (helmet codes) also had very high fatality rates (90%+), so this feature partly reflects vehicle type as well as belt use.

2. **Alcohol matters, but less than originally reported.** With correct coding, 13.3% of drivers had a measured BAC ≥ 0.08, and 79.2% of them were killed, compared with 41.2% of other drivers. Its SHAP importance (0.0461) is about half the original figure. Survivors are tested much less often than killed drivers, so this gap is probably overstated.

3. **Rural crashes are more deadly.** 53.1% of drivers in rural crashes were killed vs. 39.8% in urban crashes, and rural/urban is now the 4th most important feature (0.0433).

4. **Geographic concentration.** The top five states (TX, CA, FL, GA, NC) account for 34.1% of fatally injured drivers; Texas and California alone account for 18.2%.

5. **Afternoon peak.** Driver deaths peak at 4 PM (2,659), 3 PM (2,593) and 5 PM (2,581); the 12–6 PM block accounts for 31.4% of deaths with a known hour. Midnight–2 AM is somewhat higher than 3–5 AM but below the late-afternoon and evening hours, so the "secondary overnight peak" in the original report is weak.

---

## Policy Recommendations (original submission — review against corrected findings)

The recommendations below are from the original report. Notes in *italics* show how the corrected results bear on each.

- Prioritize federal advocacy for primary enforcement seatbelt laws in the 17 states currently limited to secondary enforcement — *still well supported (finding 1).*
- Deploy DUI checkpoints concentrated in the midnight–3 AM window rather than distributed uniformly across overnight hours — *weaker: the overnight peak is modest (finding 5); alcohol is still important (finding 2).*
- Concentrate federal highway safety investment in the five highest-burden states — *supported by raw counts (34.1%), but these largely track population and road miles, not per-capita risk.*
- Expand ignition interlock programs targeting repeat alcohol-impaired offenders — *still reasonable; alcohol remains a top-3 feature.*
- Evaluate rideshare subsidy programs timed to overnight high-risk windows — *weaker, for the same reason as the DUI timing recommendation.*
- *Possible addition:* rural-road safety measures (finding 3) now have stronger support than in the original analysis.

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
