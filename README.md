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

## Repository contents

| File / folder | What it is |
|---|---|
| `World_Cup_Prediction.ipynb` | Main deliverable — data cleaning, EDA, feature engineering, and three compared models (Logistic Regression, Random Forest, Gradient Boosting), executed top to bottom. |
| `World_Cup_Report.docx` / `.pdf` | Written report (methodology, results, insights, limitations). |
| `international_matches1.csv` | Primary modelling dataset (49,490 international matches, 1872–2026). |
| `world_cup_matches1.csv`, `world_cups1.csv`, `2022_world_cup_matches1.csv` | Supporting World Cup context data and the 2022 prediction target. |
| `model.py` | The notebook's pipeline (Elo ratings, form, head-to-head, Random Forest) refactored into reusable functions for the dashboard. |
| `app.py` | Streamlit dashboard — bonus feature. |
| `Dockerfile`, `.streamlit/`, `requirements.txt` | Containerisation for the dashboard (bonus). |
| `DASHBOARD_GUIDE.md` | Script and talking points for demoing the dashboard. |

## Dataset source

International Football Results (1872–2026), Kaggle (user: martj42),
continuously-updated snapshot:
https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017

## Setup instructions

### 1. Notebook (main deliverable)

```bash
pip install -r requirements.txt
jupyter notebook World_Cup_Prediction.ipynb
```
Run all cells top to bottom (Kernel → Restart & Run All) — no internet
access is required, all four CSVs are read from this folder.

### 2. Dashboard, without Docker

```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at `http://localhost:8501`.

### 3. Dashboard, with Docker

```bash
docker build -t worldcup-dashboard .
docker run -p 8501:8501 worldcup-dashboard
```
See `DASHBOARD_GUIDE.md` for the full walkthrough and demo script.

## Key results

- Random Forest is the selected model: 59.6% accuracy vs a 47.6% baseline
  (always predicting the majority class, Win).
- Elo rating difference is the dominant predictor of match outcome,
  confirmed independently by Random Forest feature importance, Logistic
  Regression coefficient direction, and permutation importance.
- Known limitation: per-class recall for Draw is low (0.02) due to class
  imbalance — see Section 4.1 of the report for the full breakdown and
  proposed remedies.
- Live validation: Section 4.4 of the report (and Section 8b of the notebook)
  predicts the France v Spain 2026 World Cup semifinal, made before kickoff
  on 14 July — a genuine forecast that fixes the timing issue disclosed in
  Section 4.3's 2022 example.

## AI usage disclosure

Parts of this project's code (the Elo rating system, feature-engineering
pipeline, and model training/evaluation code) were developed with the
assistance of AI tools. All code has been reviewed, tested, and is
understood by the author, who can explain the logic and purpose of every
line.