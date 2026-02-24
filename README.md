# Kaggle Titanics

This repository contains machine learning pipelines for two of Kaggle's most prominent classification challenges: the historical **Titanic** dataset and the sci-fi **Spaceship Titanic** dataset.

## 📂 Repository Structure

```text
├── Titanic/
│   ├── titanic_Rambo.py          # Stacking ensemble and Bayesian search
│   └── titanic_utils.py          # Title-based imputation & group masking
└── Spaceship_Titanic/
    ├── spaceship_titanic_v2.py   # CatBoost pipeline & Optuna optimization
    └── spaceship_titanic_utils_v2.py # Spatial audits & social consensus logic

```

---

## 🚢 Project 1: Titanic (Historical)

This project focuses on predicting passenger survival using historical manifest data, with an emphasis on social structures and family groupings.

### Code Details

* **titanic_Rambo.py**:
* **Model Architecture**: Implements a `StackingClassifier` with a Logistic Regression meta-learner. Base estimators include Support Vector Machine, Random Forest, XGBoost, and CatBoost.
* **Optimization**: Uses `BayesSearchCV` for hyperparameter tuning across the ensemble members.


* **titanic_utils.py**:
* **AgeImputer**: A custom imputer that fills missing ages based on passenger **Titles** (Master, Miss, Mr, etc.) to preserve age-specific survival signals.
* **Geometric Features**: Includes `get_distance_from_stairs`, which calculates cabin proximity to the Forward and Aft Grand Staircases based on historical deck plans.
* **Group Masking**: Identifies family survival patterns in the training set to adjust test-set predictions for related members.



---

## 🛸 Project 2: Spaceship Titanic (Sci-Fi)

This project addresses an interdimensional transport event, focusing on spatial coordinates, high-dimensional spending data, and social connectivity.

### Code Details

* **spaceship_titanic_v2.py**:
* **Model Architecture**: A standalone `CatBoostClassifier` utilizing its native handling of categorical features.
* **Optimization**: Implements an `OptunaSearchCV` pipeline for automated hyperparameter tuning (learning rate, depth, l2_leaf_reg).
* **Feature Engineering**: Incorporates binary features for **BowAnomaly** and **StarboardScrape**, along with interaction terms for high-spenders (based on a **median** threshold) in risk zones.


* **spaceship_titanic_utils_v2.py**:
* **Audit Suite**: Contains `plot_confidence_audit` for probability density analysis and `plot_feature_bias_audit` to detect systematic errors across feature bins.
* **Social Consensus**: Implements `apply_spaceship_group_mask`, which nudges test-set probabilities based on the transport outcomes of group-mates found in the training data.



---

## 🛠️ Requirements & Installation

The following libraries are required to run the scripts:

```bash
pip install pandas numpy scikit-learn catboost xgboost optuna category_encoders matplotlib seaborn

```

## 📊 Implementation Summary

| Component | Titanic Strategy | Spaceship Titanic Strategy |
| --- | --- | --- |
| **Model** | Meta-Stacking Ensemble | Hyper-Optimized CatBoost |
| **Imputation** | Title-Specific Heuristics | Hybrid Heuristic/ML |
| **Spatial Logic** | Stairwell Proximity | Hull Anomaly Zones |
| **Group Logic** | Family Title Heuristics | Probability Nudging (NMI) |

---
