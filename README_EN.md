<div align="center">

# Chronic Disease Risk Models

**7 chronic-disease risk models · Built entirely on public population-survey data · Reproducible byte-for-byte**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Data](https://img.shields.io/badge/data-Public%20Domain-green.svg)](#-data-sources)

**English** | [简体中文](./README.md)

</div>

---

## What is this

A set of **ready-to-run** risk-prediction models covering 7 common chronic diseases. Each model ships as a self-contained skill package: **unzip and predict — no retraining required** — and includes the full training pipeline so results can be **reproduced byte-for-byte**.

> **This is not a paper reproduction and not a demo.** Every model comes with a data download + SHA256 verification script, a training script, an inference script, complete validation metrics, and a provenance document that spells out every limitation.

### Key results

| Disease | Data source | N | Cases | Features | CV AUC | Balanced acc. |
|---------|-------------|--:|------:|---------:|-------:|--------------:|
| **Coronary heart disease** | BRFSS 2017 | 445,872 | 39,357 | 15 | **0.8460** | 0.7684 |
| **COPD** | BRFSS 2017 | 447,718 | 37,577 | 14 | **0.8415** | 0.7641 |
| **Stroke** | BRFSS 2017 | 448,666 | 18,956 | 15 | **0.8149** | 0.7436 |
| **Diabetes** (v3) | NHIS 2016–2025 | 291,945 | 31,760 | 7 | **0.8124** | 0.7419 |
| **Hypertension** | NHANES 2017–2018 | 5,250 | 2,857 | 14 | 0.8234 | — |
| **Hyperlipidemia** | NHIS 2016–2025 | 291,295 | 91,932 | 7 | **0.7956** | 0.7251 |
| **Diabetes** (v2, Pima) | Pima Indians | 768 | 268 | 8 | — | — |

> AUC values are 3×5-fold repeated stratified cross-validation (hypertension and diabetes v2 are earlier releases — see [Version history](#-version-history-what-changed-and-why)). Full metrics in [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

---

## Why this project exists

Three traps show up again and again in chronic-disease risk modelling:

**Trap 1 — Using a diagnostic criterion to predict the diagnosis.**
Predicting diabetes from HbA1c, hyperlipidemia from lipid panels, or hypertension from blood pressure. These measurements **are the diagnostic criteria**. The resulting AUC looks great, but the model has learned to predict a diagnosis from the diagnosis — circular reasoning with no predictive value.

**Trap 2 — A dataset that does not represent a population.**
The classic Pima Indians Diabetes Database has **768 records**, and its own source file states:

> *"all patients here are females at least 21 years old of Pima Indian heritage"*

— **female only, ≥21, Pima only**. The dataset does not even contain a sex column. A model trained on it cannot be transported to men or to other ancestries.

**Trap 3 — Judging a rare outcome by accuracy.**
With 10.88% prevalence, predicting "no diabetes" for everyone already yields **89.12% accuracy**. Reporting "89% accuracy" is meaningless here.

**This project addresses all three systematically** — see [Methodology](#-methodology).

---

## Quick start

```bash
git clone https://github.com/bellawang2024/chronic-disease-risk-models.git
cd chronic-disease-risk-models
pip install -r requirements.txt

# Example: diabetes v3, using the bundled model
cd skills/diabetes_prediction_v3
python scripts/predict.py --age 55 --bmi 28.5 --smoker 4 --hypertension 1 --high-chol 1
```

```
Risk probability: 26.34%   (decision threshold 0.105)
Risk band       : Elevated
Reference       : observed ~29.5% in this band (~2.7x the population baseline)
```

Batch inference:

```bash
python scripts/batch_predict.py --input examples/people.csv --output results.csv
```

---

## Data sources

**All data is U.S. government public-domain data — no registration, no data use agreement, freely downloadable.**

| Dataset | Description | Population | Homepage |
|---------|-------------|------------|----------|
| **NHIS** | National Health Interview Survey — in-person household sampling | U.S. civilian non-institutionalized, 18+ | [cdc.gov/nchs/nhis](https://www.cdc.gov/nchs/nhis/) |
| **BRFSS** | Behavioral Risk Factor Surveillance System — nationwide telephone survey | U.S. adults, 400k+ per year | [cdc.gov/brfss](https://www.cdc.gov/brfss/) |
| **NHANES** | National Health and Nutrition Examination Survey — includes physical exam and lab tests | U.S. civilians, ~5,000 adults per cycle | [cdc.gov/nchs/nhanes](https://www.cdc.gov/nchs/nhanes/) |
| **Pima Indians** | Pima Indians Diabetes Database (retained for version history only) | ⚠️ Pima women only, ≥21, 768 records | [openml.org/d/37](https://www.openml.org/d/37) |

### Data selection survey: what we evaluated and rejected

We systematically evaluated many candidate sources and **recorded why each was rejected**. This part rarely makes it into papers but is the most useful thing for anyone reproducing the work:

| Source | Verdict | Reason |
|--------|---------|--------|
| **SEER** cancer registry | ❌ Not usable for risk prediction | Full variable-dictionary search confirms **zero** smoking / alcohol / BMI / family-history variables. It is a case registry with **no healthy controls**. Suitable only for incidence and survival calibration |
| **NHANES** for cancer / rare outcomes | ❌ Too few cases | Stomach cancer across **all 11 cycles (1999–2023) totals 59 cases** (3–10 per cycle). *H. pylori* exists only in 1999–2000 |
| **NHIS education/income** | ❌ Not alignable across years | The **2016–2018 adult files contain no `EDUC`/`RATCAT` variables at all**; they were introduced in 2019, making a consistent 10-year model impossible |
| **NHIS physical activity / alcohol** | ❌ Alternating-year modules | After the 2019 redesign, physical activity is collected **only in even years (2020/2022/2024)**; alcohol is absent from post-2019 adult files |
| **NHIS 2011–2015** | ❌ Format unusable | Fixed-width ASCII only, with per-year record lengths of 827/1086/1021/1087/904; layout PDFs contain no byte positions |
| **Large Kaggle chronic-disease datasets** | ⚠️ Screened case by case | Some are synthetic (detection methodology in [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md)) |

> **A counter-intuitive finding:** the coronary heart disease / stroke / COPD models are built on 440k+ BRFSS records, while diabetes / hyperlipidemia are built on ~290k NHIS records — not because NHIS is better, but because **BRFSS does not record cancer site**, and NHIS hypertension/diabetes questions became cross-year alignable after the 2019 redesign. **Variable availability, not sample size, drove the data selection.**

Per-year variable alignment tables, SHA256 checksums, and verbatim codebook verifications are in each skill's `data/DATA_PROVENANCE.md`.

---

## Methodology

### One pipeline for all models

All 7 models share an **identical** pipeline so they can be compared directly:

```
Official microdata (CDC FTP)
   │  ① fetch_data.py — download + per-file SHA256 verification + cross-era variable harmonisation
   ▼
Standardised cohort CSV (data/*.csv, with SHA256)
   │  ② Feature ladder: add candidate features in layers, evaluate on 8 random seeds
   ▼
Minimal feature set
   │  ③ train_model.py
   ▼
XGBoost model + threshold + risk bands (models/, results/)
   │  ④ predict.py / batch_predict.py
   ▼
Individual probability + risk band + observed band rate
```

### Design principles

**Principle 1 — Risk factors only; never diagnostic criteria.**

The test is explicit: **if the measurement is itself a diagnostic criterion for the disease, it cannot be a feature.**

| Disease | Excluded | Reason |
|---------|----------|--------|
| Diabetes | HbA1c, fasting glucose, OGTT | HbA1c ≥6.5% and FPG ≥126 mg/dL *are* the diagnostic criteria |
| Hyperlipidemia | Total cholesterol, LDL-C, triglycerides | Same |
| Hyperlipidemia | Lipid-lowering medication, recent lipid panel | Taking medication or being tested already implies diagnosis |
| Hypertension | Blood pressure itself | Same |

**Principle 2 — No demographic partitioning attributes.**

Sex and race variables are **never extracted at the data-construction stage** (see the `OUTPUT_COLUMNS` allow-list in each `fetch_data.py`). This is a **structural guarantee**, not a training-time promise — the fields simply do not exist in the model artefacts.

> **This choice has a cost, and we quantified it.** For thyroid cancer, adding sex raises AUC from 0.6394 to **0.7004** (+0.061). We publish both numbers and let users decide.

**Principle 3 — No proxies for healthcare access.**

Features like "skips breakfast" or "poor sleep" are common in cardiovascular datasets. They are not causes — they are **proxies for socioeconomic status and healthcare-seeking behaviour**. Including them raises AUC while teaching the model *who gets diagnosed* rather than *who gets sick*. The same applies to education and income.

**Principle 4 — Rare outcomes are judged by AUC / PR-AUC / balanced accuracy.**

Prevalence varies enormously across diseases (CHD 8.83%, diabetes 10.88%), and majority-class baseline accuracy ranges from 68% to 91%. **This project never uses plain accuracy as an acceptance criterion.**

### Feature selection

**Method: the feature ladder.**

Start from the smallest feature set, add candidates in layers, and evaluate each layer on the **same split** across **8 random seeds**, taking the mean AUC. **Only features with a material gain are kept.**

Diabetes v3:

| Feature set | n | CV AUC | Δ |
|-------------|---|-------:|--:|
| Core risk factors (age/BMI/height/weight/smoking) | 5 | 0.7677 | — |
| **+ hypertension + high cholesterol ← adopted** | **7** | **0.8124** | **+0.045** |
| + CHD + stroke + COPD | 10 | 0.8162 | +0.004 |

**Hypertension and high cholesterol contribute a material +0.045** (they are metabolic-syndrome components, and standard diabetes risk scores such as FINDRISC include them). **CHD/stroke/COPD add only +0.004** and are *complications* of diabetes (clear reverse causation), so they were excluded.

Lung cancer, using the same method, produced the opposite conclusion — **only 3 features are needed**:

| Feature set | n | Mean AUC |
|-------------|---|---------:|
| Age | 1 | 0.7927 |
| + smoking status | 2 | 0.8425 |
| **+ COPD ← adopted** | **3** | **0.8706** |
| + cigarettes/day + BMI + height + weight | 7 | 0.8681 |

Cigarettes/day, BMI, height and weight each gain **≤0.001**, so all were dropped — **three questions** (age, smoking status, ever told COPD) reach AUC 0.87.

> **The criterion for keeping a feature is its measured gain — not whether a textbook lists it as a risk factor.**

### How performance is judged

This is the point we most want to make clear: **the same metric means completely different things at different prevalence levels.**

**Layer 1 — Look at the baseline before the model.**

| Disease | Prevalence | Majority-class accuracy | Model plain accuracy | Verdict |
|---------|-----------:|------------------------:|---------------------:|---------|
| Diabetes v3 | 10.88% | 89.12% | 0.6980 | **Below baseline** — the price of detecting 80% of cases |
| Lung cancer | 0.412% | 99.59% | 0.8272 | **Far below baseline**; plain accuracy is useless |
| CHD | 8.83% | 91.17% | — | Same |

> Predicting "healthy" for everyone yields 99.59% accuracy. **Any model report that quotes only accuracy can be ignored.**

**Layer 2 — Four metrics must be read together.**

| Metric | Question it answers | When it matters |
|--------|--------------------|-----------------|
| **AUC** | Ranking ability — probability a random positive outranks a random negative | General discrimination |
| **PR-AUC** | At extreme rarity, AUC is inflated by the huge number of true negatives; PR-AUC is more sensitive | Essential when prevalence <1% |
| **Balanced accuracy** | Mean of sensitivity and specificity — the imbalance-aware "accuracy" | Replaces plain accuracy |
| **Sensitivity / specificity** | The trade-off at a specific operating point | Determines screening use |

**Lung cancer as an example:** prevalence is 0.412%, so the PR-AUC baseline is only 0.0041 and the model reaches 0.0377 — a **9.17× lift**. That information is invisible in the AUC.

**Layer 3 — The threshold must not default to 0.5.**

At 10.88% prevalence, a 0.5 threshold classifies nearly every case as negative. This project **always selects the threshold by maximising balanced accuracy on the training set via cross-validation** (diabetes v3: 0.105; lung cancer: 0.0055) — **never touching the test set.**

**Layer 4 — Publish an operating-point table and let users choose.**

Lung cancer:

| Threshold | Sensitivity | Specificity | Precision | % flagged | Use |
|----------:|------------:|------------:|----------:|----------:|-----|
| 0.002 | 0.896 | 0.641 | 1.02% | 36.1% | Triage (favour sensitivity) |
| **0.0055 (default)** | **0.746** | **0.828** | **1.75%** | **17.5%** | Balanced-accuracy optimum |
| 0.015 | 0.479 | 0.954 | 4.10% | 4.8% | High-risk flagging (favour precision) |

**Layer 5 — Acceptance thresholds must match disease difficulty, with a stated rationale.**

This project **does not use a single AUC cut-off**. Every disease's threshold is published with its justification:

| Disease | Threshold AUC | Rationale |
|---------|-------------:|-----------|
| Lung cancer | ≥ 0.80 | Meets the series chronic-disease bar |
| Diabetes v3 | ≥ 0.80 | Same |
| Colorectal cancer | ≥ 0.75 | Prevalence only 0.72%; performance is inherently limited at this rarity |
| Breast cancer | ≥ 0.72 | Female-only cohort with only 4 available features |
| Thyroid cancer | ≥ 0.60 | The three strongest risk factors (childhood radiation, family history, thyroid nodules) are **never collected by NHIS**; using all 9 alignable candidates still only reaches 0.6440 — **this is the measured data ceiling** |
| Hyperlipidemia | ≥ 0.78 | The label is *detection-dependent* (an asymptomatic condition requiring a prior lipid panel); a 48-configuration hyperparameter sweep spans only 0.7917–0.7960, i.e. the model has saturated |

> ⚠️ **The hyperlipidemia threshold was set post hoc** (after observing saturation), and we **state this explicitly in its documentation** rather than presenting it as a pre-registered cut-off. This is our methodological honesty floor.

---

## Version history: what changed and why

The models in this repo **were not built in one pass**. Diabetes exists in two versions, and the difference between them is exactly how the methodology evolved:

| | Diabetes v2 | Diabetes v3 |
|---|---|---|
| Dataset | Pima Indians | **NHIS 2016–2025** |
| N | 768 | **291,945** (380×) |
| Cases | 268 | **31,760** (118×) |
| Sex | **Female only** | Both sexes |
| Ancestry | **Pima only** | All ancestries |
| Features | 8 (incl. glucose, insulin, skinfold) | **7 (risk factors only; no biomarkers)** |
| Threshold | Fixed 0.5 | **Train-CV selected (0.105)** |
| Acceptance | Plain accuracy ≥ 0.75 | **AUC ≥ 0.80 and balanced accuracy ≥ 0.70** |
| Evaluation | Single holdout | **3×5-fold repeated CV** |

v2 is retained **as a historical record of the methodology** and as a contrast case for "small, population-restricted data". **We do not recommend using v2 in practice.**

> Likewise, the **hypertension model (NHANES) is an early release**: it uses HbA1c, cholesterol, HDL, creatinine and uric acid — violating the later "no diagnostic criteria" principle. It is kept for comparison, and its AUC of 0.8234 should not be compared directly with the v3 series.

---

## Further reading

| Document | Contents |
|----------|----------|
| [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) | **Data selection survey**: per-year variable alignment, SHA256 checksums, and the measured reasons 8 sources were rejected |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | **Methodology**: the four design principles, full feature-ladder measurements, threshold selection, threshold-tiering rationale |
| [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) | **All metrics**: auto-aggregated from `model_metadata.json`, with holdout/CV metrics and confusion matrices per model |

---

## Repository layout

```
chronic-disease-risk-models/
├── README.md                    # Chinese (primary)
├── README_EN.md                 # This file
├── LICENSE
├── requirements.txt
├── docs/
│   ├── DATA_SOURCES.md          # Full data-selection survey incl. rejections
│   ├── METHODOLOGY.md           # Detailed methodology
│   └── PERFORMANCE.md           # All metrics and operating points
└── skills/
    ├── chd_prediction/          # Coronary heart disease
    ├── copd_prediction/         # COPD
    ├── stroke_prediction/       # Stroke
    ├── hypertension_prediction/ # Hypertension
    ├── diabetes_prediction/     # Diabetes v2 (Pima, historical)
    ├── diabetes_prediction_v3/  # Diabetes v3 (recommended)
    └── hyperlipidemia_prediction/  # Hyperlipidemia
```

Every skill package has the same layout:

```
<disease>_prediction/
├── SKILL.md                 # Skill docs (full evaluation + interpretation warnings)
├── README.md
├── requirements.txt
├── data/
│   ├── <disease>_*.csv      # Modelling cohort (bundled, works offline)
│   └── DATA_PROVENANCE.md   # SHA256, codebook evidence, label rules, feature rationale
├── models/                  # Pre-trained inference artefacts
├── results/                 # All evaluation metrics
├── examples/people.csv
└── scripts/
    ├── fetch_data.py        # Download + SHA256 verification + harmonisation
    ├── train_model.py       # Training + validation
    ├── predict.py           # Single-record inference
    ├── batch_predict.py     # Batch inference
    └── feature_importance.py
```

---

## Reproducibility

**Every model can be rebuilt byte-for-byte from the official raw files.**

```bash
cd skills/diabetes_prediction_v3
python scripts/fetch_data.py    # Download 10 NHIS years (~45MB) → verify SHA256 → rebuild cohort
python scripts/train_model.py   # Retrain, ~45 seconds
```

Four properties were verified for every model:

| Property | Method | Result |
|----------|--------|--------|
| **Data rebuilds** | Rebuild from the 10 official ZIPs end-to-end, compare SHA256 | ✅ Byte-identical |
| **Model reproduces** | Retrain and compare model artefact SHA256 | ✅ Byte-identical |
| **Works offline** | Delete `data/` and `results/`, then run inference | ✅ Works (imputer falls back to metadata) |
| **No leakage** | Threshold selected on training folds only | ✅ Selected inside each fold |

> Measured: `hyperlipidemia_nhis.csv` rebuilt to SHA256 `6c4055d8...98de1`, **byte-identical** to the shipped file.

---

## Known limitations

**This section is prominent because it matters more than the results.**

**1. Every label is self-reported.**
"Ever told by a doctor that you had X" conflates three things: true disease status, **whether a diagnostic opportunity occurred**, and whether the patient was told. Questionnaires cannot separate them. **"Low risk" does not mean disease-free — it may simply mean never tested.**

**2. Cross-sectional design: case identification, not incidence prediction.**
NHIS/BRFSS ask "have you ever been told" at a single time point. The models predict **the likelihood of existing disease**, not "risk of developing it in the next N years".

**3. Reverse causation cannot be excluded.**
For lung cancer: **former smokers have a higher lung-cancer rate (0.997%) than current smokers (0.681%)**. This is not "quitting is harmful" — it is the superposition of (a) quitting for health reasons, (b) quitting after diagnosis, and (c) former smokers being older and more frequently examined. Likewise BMI is inversely associated with lung cancer (more likely tumour-induced weight loss). A cross-sectional design cannot separate these.

**4. Detection bias.**
People already diagnosed interact more with the healthcare system and are more likely to be found to have other conditions. In diabetes v3, hypertension's importance of 0.4684 reflects both a genuine metabolic-syndrome association *and* detection bias.

**5. Tree-model feature importance is absorbed by binary features.**
In diabetes v3, hypertension shows importance 0.4684 — but after removing hypertension and high cholesterol, **age jumps from 0.068 to 0.630 and BMI from 0.034 to 0.221**. Do not read gain importance as a ranking of risk-factor strength.

**6. Population applicability.**
All models are based on the **U.S.** population. Applying them elsewhere — especially where diet and healthcare access differ — requires re-validation.

**7. The thyroid-cancer and hyperlipidemia models have limited discrimination.**
AUC 0.6394 and 0.7956 respectively. The former is limited by the data (its strongest risk factors are not collected); the latter by the detection-dependence of its label. **Do not use either for clinical decisions.**

---

## ⚠️ Disclaimer

**This project is for research and educational use only.**

- These models **cannot replace** any medical examination, diagnosis, or treatment.
- A "high risk" output means **relatively elevated risk**; absolute risk is typically in the single or low double digits.
- Diagnosing any condition requires **clinical testing and physician judgement**.
- Follow local clinical guidelines for screening (diabetes screening, lipid panels, blood-pressure monitoring).
- **Do not use this project for any clinical decision-making.**

---

## Citation

```bibtex
@misc{chronic_disease_risk_models,
  title  = {Chronic Disease Risk Models: Reproducible XGBoost Models on US National Survey Data},
  year   = {2025},
  note   = {NHIS 2016--2025, BRFSS 2017, NHANES 2017--2018},
  url    = {https://github.com/bellawang2024/chronic-disease-risk-models}
}
```

**Data citations** (please cite the original surveys as well):

- National Center for Health Statistics. *National Health Interview Survey, 2016–2025.*
- Centers for Disease Control and Prevention. *Behavioral Risk Factor Surveillance System, 2017.*
- National Center for Health Statistics. *National Health and Nutrition Examination Survey, 2017–2018.*

---

## License

- **Code**: MIT License — see [`LICENSE`](./LICENSE)
- **Data**: U.S. government public domain. This repo does **not** redistribute raw data; `scripts/fetch_data.py` downloads it from official CDC endpoints.
- **Models**: Research and educational use only.

---

## Roadmap

- [ ] Add SHAP interpretability analysis
- [ ] Add calibration plots
- [ ] Provide a Docker image
- [ ] Add external validation cohorts
- [ ] Cross-population transportability checks (UK Biobank / CKB)

**Contributing**: see [`CONTRIBUTING.md`](./CONTRIBUTING.md). New disease models are welcome, but **must follow the four design principles above and ship a complete provenance document**.

---

<div align="center">

**If this project helps you, please consider giving it a Star ⭐**

Related: [Cancer Risk Models](https://github.com/bellawang2024/cancer-risk-models)

</div>
