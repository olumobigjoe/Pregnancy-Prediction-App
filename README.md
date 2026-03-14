# 🤖 Predicting IVF Pregnancy Outcomes with Random Forest

> **Machine Learning for Assisted Reproductive Technology (ART) — Binary Classification using Random Forest Classifier**

---

## 📌 Project Overview

This project applies a **Random Forest Classifier** to predict the outcome of In Vitro Fertilisation (IVF) procedures — specifically whether a patient will achieve a successful pregnancy — using three key clinical features: **patient age**, **BMI**, and **number of embryos transferred**.

The dataset contains **604 real patient records** from an ART clinic, collected over an 11-year period (2010–2021), making this a genuine clinical prediction problem with real-world implications for reproductive medicine.

---

## 🎯 Objective

| Item | Detail |
|---|---|
| **Task** | Binary classification |
| **Target variable (y)** | `Outcome` — `1` = Pregnancy achieved, `0` = No pregnancy |
| **Features (X)** | `Age`, `BMI`, `Number of Embryos Transferred` |
| **Algorithm** | Random Forest Classifier |
| **Framework** | Python · scikit-learn |

---

## 📂 Dataset Description

| Property | Value |
|---|---|
| Source | ART / IVF Clinic Records |
| Date range | July 2010 — November 2021 |
| Total records | 604 patients |
| Total features | 23 columns |
| Missing values | Minimal (4 in date column, 2 in ovarian reserve) |
| Target class balance | 76.3% negative (no pregnancy) · 23.7% positive (pregnancy) |

### Full Feature List (23 columns)

| Column | Type | Description |
|---|---|---|
| `Date of IVF Presentation` | Date | Date patient presented for IVF |
| `Patient_ID` | Integer | Unique patient identifier |
| `Age` | Integer | Patient age in years |
| `Religion` | Categorical | Christianity (385), Islam (117), Others (102) |
| `Tribe` | Categorical | Patient's ethnic group |
| `Parity` | Integer | Number of previous pregnancies |
| `Nos children alive` | Integer | Number of living children |
| `(BMI)` | Float | Body Mass Index (kg/m²) |
| `Stimulation protocol` | Categorical | Ovarian stimulation protocol used |
| `Fertilization method` | Categorical | ICSI (593) or IVF (11) |
| `Embryo Quality` | Categorical | Grade 1–4 (Grade 2 = 533 cases) |
| `Endometrial preparation` | Categorical | Uterine lining preparation method |
| `Pre-implantation genetic screening` | Binary | Whether PGS was performed |
| `Number of embryo(s) transferred` | Integer | **Feature used in model** |
| `Type of embryo transferred` | Categorical | Fresh or frozen embryo |
| `Sperm Quality` | Categorical | Sperm assessment grade |
| `Outcome` | Binary | **Target — 1 = pregnancy, 0 = no pregnancy** |
| `IVF experience` | Categorical | Patient satisfaction rating |
| `Ovarian reserve markers` | Categorical | AMH / AFC classification |
| `Type of cycle` | Categorical | PC (418) or DC (186) |
| `Was donor gamete used?` | Binary | 0 = own gamete, 1 = donor |
| `Type of donor gamete used` | Categorical | Donor type if applicable |
| `Complications developed` | Categorical | Post-procedure complications |

### Class Distribution

```
Outcome 0 (No Pregnancy) : 461 patients  (76.3%)
Outcome 1 (Pregnancy)    : 143 patients  (23.7%)
```

> ⚠️ **Imbalanced dataset** — the positive class (pregnancy) accounts for only 23.7% of records. This is clinically realistic but requires careful interpretation of model performance metrics.

---

## 🔬 Feature Statistics

### Age (`X₁`)
| Statistic | Value |
|---|---|
| Mean | 37.7 years |
| Std Dev | 6.2 years |
| Min | 18 years |
| Max | 55 years |
| Median (P50) | 37 years |
| Q1 (P25) | 33 years |
| Q3 (P75) | 42 years |

### BMI (`X₂`)
| Statistic | Value |
|---|---|
| Mean | 27.3 kg/m² |
| Std Dev | 3.7 kg/m² |
| Min | 16.5 kg/m² |
| Max | 56.1 kg/m² |
| Median | 27.2 kg/m² |

### Number of Embryos Transferred (`X₃`)
| Value | Count |
|---|---|
| 0 | 2 |
| 1 | 19 |
| 2 | 576 (most common) |
| 3 | 1 |
| 5 | 6 |

---

## 🧠 Model Architecture

### Algorithm: Random Forest Classifier

A Random Forest is an ensemble learning method that builds multiple decision trees during training and outputs the class that is the mode of the classes predicted by individual trees. It is well-suited to clinical datasets because it handles non-linear relationships, is robust to outliers, and provides feature importance scores.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Features and target
X = df[['Age', '(BMI)', 'Number of embryo(s) transfered?']]
y = df['Outcome']

# Train/test split — 80/20 stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # preserves class balance in both splits
)

# Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

### Train / Test Split

| Set | Records | Positive (Pregnancy) | Negative |
|---|---|---|---|
| Training | 483 (80%) | ~115 | ~368 |
| Testing | 121 (20%) | 29 | 92 |

---

## 📊 Model Performance

### Key Metrics

| Metric | Score |
|---|---|
| **Accuracy** | 66.1% |
| **ROC-AUC** | 0.5425 |
| **5-Fold CV Accuracy** | 70.5% ± 2.2% |

### Classification Report

```
                 Precision    Recall    F1-Score    Support
No Pregnancy       0.76        0.80       0.78        92
   Pregnancy       0.25        0.21       0.23        29

    Accuracy                              0.66       121
   Macro Avg       0.51        0.51       0.50       121
Weighted Avg       0.64        0.66       0.65       121
```

### Confusion Matrix

```
                    Predicted: No      Predicted: Yes
Actual: No              74  (TN)            18  (FP)
Actual: Yes             23  (FN)             6  (TP)
```

| Term | Value | Meaning |
|---|---|---|
| True Negatives (TN) | 74 | Correctly predicted no pregnancy |
| False Positives (FP) | 18 | Incorrectly predicted pregnancy |
| False Negatives (FN) | 23 | Missed actual pregnancies |
| True Positives (TP) | 6 | Correctly predicted pregnancy |

---

## 🌟 Feature Importance

The Random Forest assigns importance scores to each feature based on the mean decrease in impurity across all trees:

| Rank | Feature | Importance Score | Contribution |
|---|---|---|---|
| 1 | **BMI** | 0.6243 | 62.4% |
| 2 | **Age** | 0.3550 | 35.5% |
| 3 | **No. Embryos Transferred** | 0.0207 | 2.1% |

> **BMI dominates** with 62.4% of the model's decision-making weight, followed by Age at 35.5%. Number of embryos transferred contributed only 2.1%, likely because 95.4% of patients had exactly 2 embryos transferred — giving the model almost no variance to learn from on that feature.

---

## 🔍 Interpretation & Discussion

### What the model captures well
- The model achieves **76.3% accuracy on the majority class** (no pregnancy), correctly identifying 74 of 92 non-pregnant outcomes on the test set.
- **Cross-validation accuracy of 70.5%** suggests the model generalises reasonably across folds.

### Limitations

**1. Class imbalance**
With only 23.7% positive cases, the model is biased toward predicting the majority class. The **recall for the pregnancy class is only 21%** — meaning the model misses 79% of actual pregnancies. In a clinical context, this is a critical failure mode.

**2. Limited feature set**
Only 3 of 23 available features were used. Key clinical predictors — embryo quality, ovarian reserve markers, stimulation protocol, sperm quality, and type of cycle — were excluded. Including these would likely improve predictive performance substantially.

**3. Low embryo transfer variance**
95.4% of patients received exactly 2 embryos, making `Number of Embryos Transferred` near-constant and therefore nearly useless as a predictive feature.

**4. ROC-AUC of 0.54**
A ROC-AUC near 0.5 indicates the model is barely better than random chance at distinguishing between pregnant and non-pregnant outcomes — reinforcing the need for richer features.

### Clinical Implications
These results suggest that **age and BMI alone are insufficient** to accurately predict IVF outcomes. A clinically useful model would require incorporation of laboratory parameters (embryo grading, AMH levels, antral follicle count) and procedural variables (stimulation protocol, number of oocytes retrieved).

---

## 🚀 Reproducing the Results

### Requirements

```bash
pip install pandas numpy scikit-learn openpyxl matplotlib seaborn
```

### Steps

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. Load data
df = pd.read_excel("DATASET__ART_csv.xlsx")
df.columns = df.columns.str.strip()

# 2. Define features and target
X = df[['Age', '(BMI)', 'Number of embryo(s) transfered?']]
y = df['Outcome']

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Train
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 5. Evaluate
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print(f"Accuracy : {rf.score(X_test, y_test):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

---

## 📁 Repository Structure

```
art-ivf-outcome-prediction/
├── art_rf_classifier.py          ← main model script
├── DATASET__ART_csv.xlsx         ← raw dataset (604 records, 23 features)
├── notebooks/
│   └── ART_Analysis.ipynb        ← exploratory data analysis notebook
├── outputs/
│   ├── confusion_matrix.png      ← confusion matrix heatmap
│   ├── roc_curve.png             ← ROC curve plot
│   └── feature_importance.png    ← feature importance bar chart
└── README.md                     ← this file
```

---

## 🔭 Future Work

| Enhancement | Expected Benefit |
|---|---|
| Include all 23 features | Richer signal for the classifier |
| Handle class imbalance (SMOTE / class_weight) | Improve recall on the positive class |
| Hyperparameter tuning (GridSearchCV) | Optimise n_estimators, max_depth, min_samples_split |
| Compare with other classifiers (XGBoost, SVM, Logistic Regression) | Identify best-performing algorithm |
| Incorporate embryo quality and ovarian reserve markers | These are clinically proven predictors |
| Time-series analysis across the 2010–2021 span | Detect trend improvements in clinic outcomes |
| SHAP values for explainability | Understand individual patient predictions |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| scikit-learn | Random Forest, train/test split, metrics |
| Pandas | Data loading and preprocessing |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Visualisations |
| openpyxl | Reading `.xlsx` files |

---

## 📄 Dataset Ethics Note

This dataset contains de-identified patient records from an ART clinic. All analyses are conducted strictly for academic and research purposes. No personally identifiable information is used or disclosed.

---

## 📄 License

MIT — free to use and adapt for research purposes.

---

*Random Forest Classifier · IVF Outcome Prediction · ART Clinical Dataset (2010–2021)*
