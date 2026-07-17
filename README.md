# Predictive Modeling of FIFA World Cup Match Outcomes

BDS23114 Data Analytics — May 2026 Semester
Author: Lee Wen Xin (Student ID: BIT-B2201F-2505004)
Lecturer: Dr. Ng Choon Ching

## Project description

This project applies the full data-analytics lifecycle — cleaning, EDA,
feature engineering, and predictive modelling — to 150+ years of
international football results, with the goal of both predicting match
outcomes (Win / Draw / Loss) and explaining *why* a team wins. A bonus
Streamlit dashboard turns the same pipeline into an interactive tool where
a user can pick any two teams and see the prediction plus the reasoning
behind it.

## Repository structure

```
world-cup-outcome-prediction/
├── README.md
├── requirements.txt
├── Dockerfile
├── .gitignore
│
├── data/                         Datasets (see "Dataset source" below)
│   ├── international_matches1.csv
│   ├── world_cup_matches1.csv
│   ├── world_cups1.csv
│   └── 2022_world_cup_matches1.csv
│
├── notebook/
│   └── World_Cup_Prediction.ipynb   Main deliverable — cleaning, EDA,
│                                     feature engineering, three compared
│                                     models, and a live 2026 prediction
│
├── dashboard/                    Bonus feature — interactive Streamlit app
│   ├── app.py                    UI — dropdowns, prediction, explanation
│   ├── model.py                  Same Elo/feature/RF pipeline as the notebook
│   ├── precompute.py             Pre-trains the model at Docker build time
│   ├── international_matches1.csv  Copy of the dataset for standalone runs
│   ├── DASHBOARD_GUIDE.md        Demo script and Q&A prep
│   └── .streamlit/config.toml    Theme config
│
├── report/
│   ├── World_Cup_Report.docx
│   └── World_Cup_Report.pdf
│
└── presentation/
    └── World_Cup_Presentation.pptx
```

## Dataset source

International Football Results (1872–2026), Kaggle (user: martj42),
continuously-updated snapshot:
https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017

Note: the dataset's own page title now reads "International football results
from 1872 to 2026" — Kaggle keeps a dataset's original URL slug even after
the maintainer updates its contents, which is why the link above still says
"2017" even though the data (and page title) go through July 2026.

## Setup instructions

### 1. Notebook (main deliverable)

```bash
pip install -r requirements.txt
jupyter notebook notebook/World_Cup_Prediction.ipynb
```
Run all cells top to bottom (Kernel → Restart & Run All). The notebook
reads its CSVs from `../data/` — open it from inside `notebook/` (the
normal Jupyter behaviour) and the relative paths resolve automatically. No
internet access is required.

### 2. Dashboard, without Docker

```bash
pip install -r requirements.txt
cd dashboard
cp ../data/international_matches1.csv .   # model.py reads the CSV from the
                                           # current directory - no ../data/
                                           # fallback exists, so it must be
                                           # copied in first when running this
                                           # way (Docker does this for you
                                           # automatically at build time)
streamlit run app.py
```
Opens at `http://localhost:8501`.

### 3. Dashboard, with Docker

From the **repo root** (the Dockerfile's COPY paths expect this):
```bash
docker build -t worldcup-dashboard .
docker run -p 8501:8501 worldcup-dashboard
```
The build step also runs `precompute.py`, which trains the model and
pre-computes all Elo/form/head-to-head features once and bakes the result
into the image as `model_cache.pkl`. This is why the very first `docker
build` takes a little longer (a one-time cost) — but it means every
`docker run` afterwards (including every restart while testing or
recording a demo) loads that pre-built result instantly instead of
re-running the full pipeline from scratch each time.

See `dashboard/DASHBOARD_GUIDE.md` for the full walkthrough and demo script.

## Key results

- Random Forest is the selected model: 59.7% accuracy vs a 47.6% baseline
  (always predicting the majority class, Win). All three models (Logistic
  Regression, Random Forest, Gradient Boosting) land within 0.006 macro-F1
  of each other; Random Forest is chosen for its native feature importances
  and because it is the same model used in the dashboard, not because it
  scored highest.
- Elo rating difference is the dominant predictor of match outcome,
  confirmed independently by Random Forest feature importance, Logistic
  Regression coefficient direction, and permutation importance.
- Known limitation: per-class recall for Draw is low (0.01) due to class
  imbalance — see Section 4.1 of the report for the full breakdown and
  proposed remedies.
- Live validation: Section 4.4 of the report (and Section 8b of the
  notebook) predicts the France v Spain 2026 World Cup semifinal, made
  before kickoff on 14 July — a genuine forecast that fixes the timing
  issue disclosed in Section 4.3's 2022 example. Update: the match has
  since been played — Spain won 2–0 — and the model's prediction favoured
  Spain, so the directional call was correct.

## AI usage disclosure

Parts of this project's code (the Elo rating system, feature-engineering
pipeline, and model training/evaluation code) were developed with the
assistance of AI tools. All code has been reviewed, tested, and is
understood by the author, who can explain the logic and purpose of every
line.