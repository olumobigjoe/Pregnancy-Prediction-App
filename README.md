# 🤖 Predicting IVF Pregnancy Outcomes with Random Forest Classifier

> **Machine Learning for Assisted Reproductive Technology (ART)**  
> Binary classification model to predict IVF pregnancy outcomes using clinical and embryological features

[![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?style=flat-square)](https://scikit-learn.org)
[![Dataset](https://img.shields.io/badge/Dataset-604_patients-green?style=flat-square)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-70.25%25-informational?style=flat-square)]()
[![AUC](https://img.shields.io/badge/ROC--AUC-0.5856-yellow?style=flat-square)]()

---

## 📌 Project Overview

Assisted Reproductive Technology (ART), particularly In Vitro Fertilisation (IVF), involves complex clinical decision-making where patient outcomes are influenced by a combination of patient characteristics and embryological factors. This project builds a **Random Forest Classifier** to predict whether an IVF procedure will result in a successful pregnancy, using **six clinically relevant features** drawn from 604 real patient records spanning over a decade of practice (2010–2021).

The model addresses a real-world binary classification problem with direct implications for patient counselling, clinical planning, and the development of decision-support tools in reproductive medicine.

---

## 🎯 Problem Statement

| Item | Detail |
|---|---|
| **Task type** | Binary classification |
| **Target variable (y)** | `Outcome` — `1` = Pregnancy achieved · `0` = No pregnancy |
| **Input features (X)** | Age, BMI, No. of Embryos Transferred, Type of Embryo Transferred, Embryo Quality, Sperm Quality |
| **Algorithm** | Random Forest Classifier |
| **Language / Library** | Python 3 · scikit-learn |
| **Data split** | 80% training · 20% testing (stratified) |

---

## 📂 Dataset

### Summary

| Property | Value |
|---|---|
| Source | ART / IVF clinic patient records |
| Date range | July 2010 — November 2021 |
| Total records | **604 patients** |
| Total features | 23 columns |
| Features used in model | **6** |
| Target classes | Binary (0 = No Pregnancy, 1 = Pregnancy) |
| Missing values | Minimal — 4 in date field, 2 in ovarian reserve markers |

### Class Distribution

| Class | Label | Count | Percentage |
|---|---|---|---|
| 0 | No Pregnancy | 461 | 76.3% |
| 1 | Pregnancy | 143 | 23.7% |

> ⚠️ **Class imbalance note:** The positive class (pregnancy) represents only 23.7% of records — a realistic clinical ratio, but one that biases classifiers toward predicting the majority class. This is a key factor in interpreting performance metrics.

---

## 🧪 Features Used in the Model

Six features were selected as predictors based on clinical relevance:

| # | Feature (X) | Original Column | Type | Description |
|---|---|---|---|---|
| 1 | `Age` | `Age` | Numeric | Patient age in years |
| 2 | `BMI` | `(BMI)` | Numeric | Body Mass Index (kg/m²) |
| 3 | `No_Embryos_Transferred` | `Number of embryo(s) transfered?` | Numeric | Count of embryos transferred in the cycle |
| 4 | `Type_Embryo` | `Type of embryo transferred` | Categorical | D3 (Day 3) or D5 (Day 5 blastocyst) |
| 5 | `Embryo_Quality` | `Embryo Quality` | Categorical | Grade 1, 2, 2.5, 3, or 4 |
| 6 | `Sperm_Quality` | `Sperm Quality` | Categorical | Grades A, B, C, D, E |

### Feature Distributions

**Age**

| Statistic | Value |
|---|---|
| Mean | 37.7 years |
| Std Dev | 6.2 years |
| Min – Max | 18 – 55 years |
| Median (P50) | 37 years |
| IQR (P25 – P75) | 33 – 42 years |

**BMI**

| Statistic | Value |
|---|---|
| Mean | 27.3 kg/m² |
| Std Dev | 3.7 kg/m² |
| Min – Max | 16.5 – 56.1 kg/m² |
| Median | 27.2 kg/m² |

**Number of Embryos Transferred**

| Value | Count |
|---|---|
| 0 | 2 |
| 1 | 19 |
| 2 | 576 ← dominant (95.4%) |
| 3 | 1 |
| 5 | 6 |

**Type of Embryo Transferred**

| Value | Count | Encoded |
|---|---|---|
| D5 (Day 5 / Blastocyst) | 555 (91.9%) | 1 |
| D3 (Day 3 / Cleavage) | 49 (8.1%) | 0 |

**Embryo Quality**

| Grade | Count | Encoded |
|---|---|---|
| Grade 1 (Best) | 42 | 0 |
| Grade 2 | 533 (88.2%) | 1 |
| Grade 2.5 | 21 | 2 |
| Grade 3 | 5 | 3 |
| Grade 4 | 3 | 4 |

**Sperm Quality**

| Grade | Count | Encoded |
|---|---|---|
| A (Best) | 21 | 0 |
| B | 459 (76.0%) | 1 |
| C | 112 (18.5%) | 2 |
| D | 8 | 3 |
| E | 4 | 4 |

---

## 🧠 Methodology

### Preprocessing Pipeline

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_excel("DATASET__ART_csv.xlsx")
df.columns = df.columns.str.strip()

# Select features and target
X = df[['Age', '(BMI)', 'Number of embryo(s) transfered?',
        'Type of embryo transferred', 'Embryo Quality', 'Sperm Quality']].copy()
X.columns = ['Age', 'BMI', 'No_Embryos_Transferred',
             'Type_Embryo', 'Embryo_Quality', 'Sperm_Quality']
y = df['Outcome']

# Encode categorical features
le = LabelEncoder()
for col in ['Type_Embryo', 'Embryo_Quality', 'Sperm_Quality']:
    X[col] = le.fit_transform(X[col].astype(str).str.strip())
```

**Encoding map applied:**

| Feature | Encoding |
|---|---|
| `Type_Embryo` | D3 → 0, D5 → 1 |
| `Embryo_Quality` | Grade 1 → 0, Grade 2 → 1, Grade 2.5 → 2, Grade 3 → 3, Grade 4 → 4 |
| `Sperm_Quality` | A → 0, B → 1, C → 2, D → 3, E → 4 |

### Model Training

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Stratified 80/20 split — preserves class ratio in both subsets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Random Forest with 100 trees
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

### Train / Test Split

| Subset | Records | Positive (Pregnancy) | Negative |
|---|---|---|---|
| Training set | 483 (80%) | ~114 | ~369 |
| Testing set | 121 (20%) | 29 | 92 |

---

## 📊 Results

### Core Performance Metrics

| Metric | Value |
|---|---|
| **Accuracy** | **70.25%** |
| **ROC-AUC** | **0.5856** |
| **Precision (Pregnancy class)** | 31.6% |
| **Recall / Sensitivity (Pregnancy class)** | 20.7% |
| **F1-Score (Pregnancy class)** | 0.25 |
| **Specificity (No Pregnancy class)** | 85.9% |
| **5-Fold CV Accuracy** | 72.7% ± 2.6% |
| **5-Fold CV ROC-AUC** | 0.583 ± 0.041 |

### Classification Report

```
                  Precision    Recall    F1-Score    Support
No Pregnancy        0.77        0.86       0.81         92
   Pregnancy        0.32        0.21       0.25         29

    Accuracy                               0.70        121
   Macro Avg        0.55        0.53       0.53        121
Weighted Avg        0.66        0.70       0.68        121
```

### Confusion Matrix

```
                     Predicted: No Pregnancy    Predicted: Pregnancy
Actual: No Pregnancy       79  (TN)                  13  (FP)
Actual: Pregnancy          23  (FN)                   6  (TP)
```

| Metric | Value | Meaning |
|---|---|---|
| True Negatives (TN) | **79** | Correctly predicted no pregnancy |
| False Positives (FP) | **13** | Incorrectly predicted pregnancy when there was none |
| False Negatives (FN) | **23** | Missed actual pregnancies (critical in clinical context) |
| True Positives (TP) | **6** | Correctly predicted pregnancy |

### Improvement Over 3-Feature Baseline

Expanding from 3 features (Age, BMI, Embryos) to 6 features (adding Type, Quality, Sperm) produced measurable gains across all metrics:

| Metric | 3-Feature Model | 6-Feature Model | Change |
|---|---|---|---|
| Accuracy | 66.1% | **70.25%** | +4.2 pts |
| ROC-AUC | 0.5425 | **0.5856** | +0.043 |
| CV Accuracy | 70.5% | **72.7%** | +2.2 pts |
| CV AUC | — | **0.583** | — |
| Precision (pos) | 25.0% | **31.6%** | +6.6 pts |
| F1-Score (pos) | 0.23 | **0.25** | +0.02 |

---

## 🌟 Feature Importance

The Random Forest computes importance as the mean decrease in Gini impurity across all 100 trees:

| Rank | Feature | Importance Score | Contribution |
|---|---|---|---|
| 🥇 1 | **BMI** | 0.4849 | **48.5%** |
| 🥈 2 | **Age** | 0.3451 | **34.5%** |
| 🥉 3 | **Sperm Quality** | 0.0735 | **7.4%** |
| 4 | **Embryo Quality** | 0.0544 | **5.4%** |
| 5 | **Type of Embryo** | 0.0212 | **2.1%** |
| 6 | **No. Embryos Transferred** | 0.0208 | **2.1%** |

**Key observations:**

- **BMI and Age together account for 83% of model decisions** — consistent with clinical evidence that patient demographics are strong predictors in IVF.
- **Sperm Quality (7.4%) and Embryo Quality (5.4%)** are the most informative laboratory features — they improve model performance compared to the 3-feature baseline.
- **Type of embryo transferred and number transferred contributed only ~2% each** — largely because 91.9% of transfers were Day 5 blastocysts and 95.4% involved exactly 2 embryos, leaving little variance for the model to learn from.

---

## 🔍 Discussion

### What the model does well

- Achieves **86% specificity** — correctly identifies 79 of 92 patients who will not achieve pregnancy, which is useful for clinical resource planning.
- **Cross-validation accuracy of 72.7%** with low variance (±2.6%) indicates stable generalisation across folds.
- Adding embryological features over the 3-feature baseline improved all performance metrics.

### Limitations

**1. Persistent class imbalance effect**
Despite using stratified splitting, the model's recall on the positive (pregnancy) class remains low at 20.7%. It correctly identifies only 6 of 29 actual pregnancies in the test set. In clinical practice, false negatives (missed pregnancies) carry significant counselling implications.

**2. Near-constant categorical features**
Both `Type of Embryo` (91.9% Day 5) and `No. of Embryos Transferred` (95.4% = 2) have very low variance. They provide minimal discrimination power and their 2.1% importance reflects this.

**3. Ordinal encoding of categorical data**
`Embryo Quality` and `Sperm Quality` were label-encoded as ordinal integers (e.g. Grade 1→0, Grade 2→1). While clinically reasonable, Random Forests may benefit from one-hot encoding for nominal categories.

**4. ROC-AUC of 0.59**
While improved from the 3-feature baseline, an AUC of 0.59 still reflects limited discriminative ability — the model is better than random, but not yet clinically reliable for individual prediction.

**5. No hyperparameter optimisation**
The model used default hyperparameters (n_estimators=100). Grid search over `max_depth`, `min_samples_leaf`, and `max_features` would likely improve performance.

---

## 🚀 Reproducing the Results

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn openpyxl
```

### 2. Full reproducible script

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, roc_auc_score)
from sklearn.preprocessing import LabelEncoder

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_excel("DATASET__ART_csv.xlsx")
df.columns = df.columns.str.strip()

# ── Features & Target ─────────────────────────────────────────────────────────
X = df[['Age', '(BMI)', 'Number of embryo(s) transfered?',
        'Type of embryo transferred', 'Embryo Quality', 'Sperm Quality']].copy()
X.columns = ['Age', 'BMI', 'No_Embryos_Transferred',
             'Type_Embryo', 'Embryo_Quality', 'Sperm_Quality']
y = df['Outcome']

# ── Encode Categoricals ───────────────────────────────────────────────────────
le = LabelEncoder()
for col in ['Type_Embryo', 'Embryo_Quality', 'Sperm_Quality']:
    X[col] = le.fit_transform(X[col].astype(str).str.strip())

# ── Split ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── Train ─────────────────────────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred,
      target_names=['No Pregnancy', 'Pregnancy']))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ── Cross-Validation ──────────────────────────────────────────────────────────
cv = cross_val_score(rf, X, y,
                     cv=StratifiedKFold(5, shuffle=True, random_state=42),
                     scoring='accuracy')
print(f"\n5-Fold CV Accuracy: {cv.mean():.4f} ± {cv.std():.4f}")

# ── Feature Importance ────────────────────────────────────────────────────────
fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances:\n", fi)
```

---

## 📁 Repository Structure

```
art-ivf-outcome-prediction/
├── art_rf_classifier.py              ← main model script
├── DATASET__ART_csv.xlsx             ← raw dataset (604 records, 23 features)
├── notebooks/
│   └── ART_RF_Analysis.ipynb         ← full EDA + modelling notebook
├── outputs/
│   ├── confusion_matrix.png          ← confusion matrix heatmap
│   ├── roc_curve.png                 ← ROC curve plot
│   └── feature_importance.png        ← feature importance bar chart
└── README.md                         ← this file
```

---

## 🔭 Future Work

| Priority | Enhancement | Expected Benefit |
|---|---|---|
| 🔴 High | **Handle class imbalance** with SMOTE or `class_weight='balanced'` | Drastically improve recall on the pregnancy class |
| 🔴 High | **Hyperparameter tuning** — GridSearchCV over `max_depth`, `n_estimators`, `min_samples_leaf` | Optimise tree complexity and reduce overfitting |
| 🟡 Medium | **Include all 23 features** — ovarian reserve markers, stimulation protocol, endometrial preparation | Richer signal for the classifier |
| 🟡 Medium | **One-hot encode** `Embryo_Quality` and `Sperm_Quality` instead of ordinal encoding | Avoid false ordinal assumptions |
| 🟡 Medium | **Compare classifiers** — XGBoost, Logistic Regression, SVM, GradientBoosting | Identify best-performing algorithm |
| 🟢 Low | **SHAP values** for patient-level explainability | Understand which features drive each prediction |
| 🟢 Low | **Time-series analysis** across 2010–2021 | Detect outcome improvement trends over the decade |

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.x | Core language |
| scikit-learn | ≥ 1.0 | Random Forest, metrics, preprocessing |
| Pandas | ≥ 1.3 | Data loading and wrangling |
| NumPy | ≥ 1.21 | Numerical operations |
| openpyxl | ≥ 3.0 | Reading `.xlsx` files |
| Matplotlib / Seaborn | ≥ 3.4 | Visualisations |

---

## 📋 Dataset Ethics Note

This dataset contains de-identified patient records from an ART clinic. All analyses are conducted for academic and research purposes only. No personally identifiable information is used or disclosed in this repository.

---

## 📄 License

MIT — free to use and adapt for non-commercial research purposes.

---

*Random Forest Classifier · IVF / ART Outcome Prediction · Clinical Dataset 2010–2021 · 604 patients · 6 features*
