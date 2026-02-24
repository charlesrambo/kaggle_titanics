# Titanic: Machine Learning from Disaster

This sub-repository contains a high-performance pipeline for the classic Titanic survival prediction challenge. The implementation focuses on preserving historical demographic signals through custom imputation and leveraging social connectivity via family-group survival heuristics.

## 📂 Files

* **`titanic_Rambo.py`**: The main execution script containing the model stacking architecture and Bayesian optimization loop.
* **`titanic_utils.py`**: A utility library for feature engineering, title-based imputation, and post-processing masks.

---

## 🛠️ Technical Implementation

### 1. Title-Based Age Imputation

Standard median imputation often ignores the relationship between age and social status. This project utilizes the **`AgeImputer`** class:

* It extracts **Titles** (e.g., *Master*, *Miss*, *Mrs*, *Mr*) from passenger names.
* It calculates the median age for each title within the training set.
* It fills missing values in the test set using these specific medians, ensuring that "Master" (young boys) and "Miss" (young girls/unmarried women) maintain their distinct age-based survival profiles.

### 2. Geometric Accessibility Features

To model the physical difficulty of reaching the lifeboats, the script includes **`get_distance_from_stairs`**:

* It maps cabin numbers to historical deck plans.
* It calculates the absolute distance to the **Forward Grand Staircase** (near cabin #55) and the **Aft Grand Staircase** (near cabin #105).
* Passengers closer to these hubs are modeled as having higher accessibility to the boat deck.

### 3. Model Stacking & Bayesian Optimization

The pipeline uses a **`StackingClassifier`** to reduce variance and capture diverse signals:

* **Base Learners**: Support Vector Machine (diversity), Random Forest (robustness), XGBoost (gradient boosting), and CatBoost (categorical handling).
* **Meta-Learner**: Logistic Regression is used to find the optimal weighted blend of the base learners' probabilities.
* **Tuning**: Uses **`BayesSearchCV`** (Scikit-Optimize) to perform a 50-iteration search through the hyperparameter space for each model.

### 4. Group Survival Heuristic

The **`apply_group_mask`** function provides a post-processing boost by leveraging social data:

* It groups passengers by **Surname** and **Ticket ID**.
* If a family/group in the training set survived or perished as a unit, that "consensus" is applied to the related members in the test set.
* This logic specifically targets the "Safe" survivor titles (*Mrs, Miss, Master, Ms*) to refine predictions where the model's probability is near the decision threshold.

---

## 📊 Requirements

To run the Titanic pipeline, install the following:

```text
numpy
pandas
scikit-learn
scikit-optimize
xgboost
catboost
category_encoders

```

---

### **Titanic About & Tags**

**Description:**
A specialized Titanic survival model featuring Title-based imputer logic, geometric staircase-proximity engineering, and a Bayesian-optimized stacking ensemble.

**Tags:**
`titanic` `machine-learning` `stacking-ensemble` `bayesian-optimization` `feature-engineering` `python` `scikit-learn`
