# World Cup Match Outcome Prediction

**Course:** BDS23114 Data Analytics | May 2026 Semester
**Project:** Predictive Modeling of FIFA World Cup Match Outcomes — A Machine Learning Approach Using Historical International Match Data
**Author:** Lee Wen Xin | Student ID: BIT-B2201F-2505004
**Lecturer:** Dr. Ng Choon Ching

---

## 1. Project Overview

This project applies the full data-analytics lifecycle to historical international football data. It has two goals:

1. **Predict** the outcome of a match (Win / Draw / Loss, from the home team's perspective).
2. **Explain** *why* one team beats another — which pre-match, measurable factors actually drive the result.

Rather than using the final score as an input (which would be data leakage), the project engineers pre-match indicators of team strength — Elo rating, recent form, head-to-head record, home advantage — and uses interpretable models to quantify how much each factor contributes to the outcome.

**Research questions:**
- Can match outcomes be predicted more accurately than a naive baseline?
- Which factors explain a win, and how much does each contribute?
- Do these factors generalise to World Cup matches, including the 2022 tournament?

## 2. Repository Structure

```
├── World_Cup_Prediction.ipynb   # Main analysis notebook: EDA, feature engineering, modelling
├── World_Cup_Report.pdf         # Written report (max 10 pages)
├── model.py                     # Reusable pipeline (Elo, features, training, prediction) used by the dashboard
├── app.py                       # Streamlit dashboard — interactive version of the model
├── Dockerfile                   # Container definition for the dashboard
├── requirements.txt             # Python dependencies
├── DASHBOARD_GUIDE.md           # Notes on running and presenting the dashboard
├── international_matches1.csv   # Main dataset (17,769 international matches, 1872–2022)
├── world_cup_matches1.csv       # World Cup match-level data (supporting context)
├── world_cups1.csv              # World Cup tournament-level summaries (supporting context)
└── 2022_world_cup_matches1.csv  # 2022 World Cup fixtures (used for live prediction demo)
```

## 3. Dataset & Source Citation

| File | Description | Source |
|---|---|---|
| `international_matches1.csv` | 17,769 international football results, 1872–2022 | Kaggle — International Football Results (martj42), continuously updated dataset. Snapshot used covers matches through September 2022. |
| `world_cup_matches1.csv`, `world_cups1.csv` | World Cup match and tournament summaries | Kaggle — FIFA World Cup dataset. |
| `2022_world_cup_matches1.csv` | 2022 World Cup group-stage fixtures (no scores, used for live prediction) ||

https://www.kaggle.com/datasets/abhijitdahatonde/fifa-world-cup-all-dataset

The dataset meets the project's minimum requirement of 500+ rows and 8+ columns, and contains no personally identifiable information.

## 4. Setup Instructions

### Run the notebook
```bash
pip install -r requirements.txt
jupyter notebook World_Cup_Prediction.ipynb
```
Run all cells top to bottom. All four CSV files must be in the same folder as the notebook.

### Run the dashboard (without Docker)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

### Run the dashboard (with Docker)
```bash
docker build -t worldcup-dashboard .
docker run -p 8501:8501 worldcup-dashboard
```
Then open `http://localhost:8501` in your browser.

## 5. Methodology Summary

- **Data cleaning:** date parsing, duplicate removal, sanity-checking extreme scorelines, target encoding (Win/Draw/Loss).
- **Feature engineering:** Elo rating (leakage-safe, computed chronologically), rolling 5-match form, head-to-head record, home advantage, experience.
- **Modelling:** chronological 80/20 train/test split (not random, to avoid leaking future matches into training). Three models compared — Logistic Regression, Random Forest (GridSearchCV-tuned), Gradient Boosting.
- **Evaluation:** accuracy, precision, recall, macro F1. Best model selected by macro F1 to fairly balance the hard-to-predict Draw class.
- **Explainability:** Random Forest feature importance, Logistic Regression coefficients (direction), and permutation importance (model-agnostic check) are used together to answer *why* teams win.

## 6. Key Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Baseline (always predict Win) | 0.482 | – |
| Logistic Regression | 0.604 | 0.453 |
| Random Forest | 0.600 | 0.451 |
| **Gradient Boosting (best)** | 0.592 | **0.470** |

All three models clearly beat the baseline. Team strength (Elo difference) is the dominant factor in explaining match outcomes.

## 7. AI Usage Disclosure

Parts of this project's code and documentation were developed with the assistance of AI tools. The student has reviewed, tested, and can explain the logic and purpose of every part of the code, including the Elo rating implementation, feature engineering, model training, and evaluation.

## 8. Bonus Features

- Interactive Streamlit dashboard with a "why this prediction" explanation panel
- Dockerised deployment
- Hyperparameter tuning via GridSearchCV with time-series cross-validation
- Ensemble methods (Random Forest, Gradient Boosting)

## 9. Limitations & Future Work

- Draws are inherently hard to predict; an ordered or Poisson-based model may help.
- No squad-level data (injuries, line-ups, red cards).
- Elo cold-start issue for newer national teams.
- Future work: player-level features, tournament-importance weighting, per-match feature contribution display.
