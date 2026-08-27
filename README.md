# EarlyGrade — Student Performance & Risk Predictor

Predicts a student's final grade and pass/fail risk **before** period grades exist, so schools can step in early instead of reacting after the fact.

## Overview

This notebook uses the [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance) (secondary school students from two Portuguese schools) to tackle one question two ways:

1. **Regression** — predict the final grade `G3` (0–20) → evaluated with **RMSE**
2. **Classification** — predict pass/fail (`G3 >= 10`) → evaluated with **F1**

Both models are trained **with and without** the period grades `G1`/`G2`. Including them makes prediction almost trivial (they're highly correlated with `G3`), but that defeats the purpose — the whole point is flagging at-risk students *before* any grades exist. The "early_warning" (no G1/G2) version is the one meant for real deployment.

A **fairness check** is also included: how much predictive weight falls on family-background features (parental education, job, family structure) versus features a student or school can actually act on (study time, absences, past failures) — and what that means for responsible deployment.

## What's Inside

| Section | What it does |
|---|---|
| 1–2. Imports & Data Loading | Loads `student-mat.csv` (Math, 395 students) and `student-por.csv` (Portuguese, 649 students), with automatic fallback download and local-file support |
| 3. Exploratory Data Analysis | Grade distribution, pass/fail balance, correlation with `G3` |
| 4. Preprocessing | One-hot encoding; builds two feature sets — `with_grades` and `early_warning` (no `G1`/`G2`) — on a shared, stratified train/test split |
| 5. Regression Models | Linear Regression, Random Forest, Gradient Boosting → compared by RMSE/MAE/R² |
| 6. Classification Models | Logistic Regression, Random Forest, Gradient Boosting → compared by F1, recall, and precision on the *fail* class (missing an at-risk student matters more than a false alarm) |
| 7. Hybrid Approach | Regress to predict a grade, then bucket into risk tiers (High risk / Borderline / On track) — combines regression's granularity with classification's actionability |
| 8. Feature Importance | Top drivers of the early-warning classifier |
| 9. Fairness Check | Measures how much weight family-background features carry, and whether removing them costs meaningful accuracy |
| 10–11. Results & Findings | Summary tables and a template for writing up conclusions |
| 12. Interactive Demo | `predict_for_student()` — pass in a student's features (no grades needed) and get back a predicted grade, risk tier, and fail probability |

## Dataset

- **Source:** [UCI Machine Learning Repository — Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Files:** `student-mat.csv` (395 students), `student-por.csv` (649 students, used by default)
- The loader checks for local copies first, then falls back to a public mirror — no manual setup required unless every source is unreachable, in which case it prints instructions.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run

1. Open `student_performance_predictor.ipynb` in Jupyter or Google Colab.
2. Run all cells top to bottom.
3. (Optional) Change `SUBJECT = "por"` to `SUBJECT = "mat"` in Section 2 to re-run the entire analysis on the Math cohort instead.
4. Use `predict_for_student(student_row)` in Section 12 to get a prediction for a new/held-out student.

## Key Design Choices

- **Two feature sets, one split** — `with_grades` and `early_warning` share the exact same stratified train/test split, so results are directly comparable.
- **Fail-class metrics prioritized** — for classification, recall/F1 on the *fail* class matter more than overall accuracy, since missing an at-risk student is costlier than a false alarm.
- **Fairness-aware by default** — the recommended deployment model excludes family-background features (parental education/job, family structure) unless they're shown to meaningfully improve performance.

## Recommended Deployment Model

The **early-warning classifier, trained without G1/G2 and without family-background features**, evaluated on fail-class recall rather than raw accuracy. Predictions are meant to prioritize which students a counselor reviews — not to make automated decisions.

## Next Steps

- Re-run with `SUBJECT = "mat"` to check whether findings hold on the smaller Math cohort.
- Tune hyperparameters (`GridSearchCV` / `RandomizedSearchCV`) on the early-warning models specifically, since that's the deployable version.
- Wrap `predict_for_student()` in a simple form (studytime, absences, failures, etc.) for a live demo.

## License

Dataset usage subject to the UCI Machine Learning Repository's terms. Code in this notebook is free to reuse/adapt.