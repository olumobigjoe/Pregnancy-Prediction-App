import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, roc_auc_score)
from sklearn.preprocessing import LabelEncoder
import joblib

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_excel("C:/Users/ADMIN/Desktop/Streamlit/ML/DATASET__ART.csv.xlsx")
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
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred,
      target_names=['No Pregnancy', 'Pregnancy']))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ── Cross-Validation ──────────────────────────────────────────────────────────
cv = cross_val_score(model, X, y,
                     cv=StratifiedKFold(5, shuffle=True, random_state=42),
                     scoring='accuracy')
print(f"\n5-Fold CV Accuracy: {cv.mean():.4f} ± {cv.std():.4f}")

# ── Feature Importance ────────────────────────────────────────────────────────
fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances:\n", fi)
# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


#Save the trained model
joblib.dump(model, filename="pregnancy_outcome.pkl")

# Load the trained model (for testing purpose)
model = joblib.load('pregnancy_outcome.pkl')















