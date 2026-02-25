# Spaceship Titanic: Interdimensional Transport Prediction

This sub-repository contains a high-precision machine learning pipeline designed to predict passenger transport status during a spacetime anomaly. The project emphasizes **Geometric Feature Engineering** and **Model Auditing** to correct systematic biases in standard classification models.

## 📂 Files

* **`spaceship_titanic_v2.py`**: The primary pipeline featuring CatBoost architecture and Optuna hyperparameter optimization.
* **`spaceship_titanic_utils_v2.py`**: A comprehensive utility suite for spatial data discretization, NMI-based feature selection, and prediction auditing.

---

## 🛠️ Technical Implementation

### 1. Geometric Anomaly Engineering

Standard models often struggle with local spatial patterns. By auditing error clusters, this project identifies and flags specific hull locations:

* **`BowAnomaly` & `StarboardScrape`**: Binary flags representing specific hull coordinates where spatial damage overrides the predictive power of spending patterns.

### 2. Model Auditing & Confidence Analysis

To move beyond a simple accuracy score, the **`spaceship_titanic_utils_v2.py`** provides diagnostic visualizations:

* `plot_confidence_audit`: Generates a normalized probability density plot to visualize class separation. This allows for the identification of "High Confidence Failures"—instances where the model is certain but incorrect.
* `plot_feature_bias_audit`: Bins continuous variables (like `Amenities` or `Age`) and calculates accuracy per bin to detect if the model is systematically biased against specific demographics.

### 3. Social Consensus (Group Nudging)

The dataset shows a high **Normalized Mutual Information (NMI)** between the `Group` ID and the target variable. The **`apply_spaceship_group_mask`** leverages this:

* It calculates the transport rate of groups using the training set's transported feature and the test set's (bias corrected ) ML probability of being transported.
* It identifies "Test" passengers belonging to groups that have high or low transportation rates.
* It applies a **probability nudge** to align test predictions with the proven outcome of their social group, effectively utilizing local consensus to refine individual probabilities.

### 4. Optuna Optimization

The CatBoost model is tuned using an **`OptunaSearchCV`** pipeline, focusing on:

* **Depth & Learning Rate**: Balancing model complexity with convergence speed.
* **L2 Leaf Reg**: Optimizing regularization to prevent overfitting on the cabin-specific noise.
* **Categorical Handling**: Leveraging CatBoost’s native support for high-cardinality features like `Cabin` and `PassengerId`.

---

## 📊 Requirements

To run the Spaceship Titanic pipeline, install the following:

```text
numpy
pandas
scikit-learn
catboost
optuna
matplotlib
seaborn
scipy
category_encoders

```

---

### **Spaceship Titanic About & Tags**

**Description:**
A spatial-logic-driven solution for the Spaceship Titanic challenge. Features include geometric anomaly flagging, Optuna-tuned CatBoost, and an NMI-based social group probability nudging system.

**Tags:**
`spaceship-titanic` `catboost` `optuna` `feature-engineering` `data-auditing` `spatial-analysis` `python`

---
