import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import pickle

# Read CSV file
data = pd.read_excel("C:/Users/ADMIN/Desktop/Streamlit/ML/DATASET__ART.csv.xlsx")
print(data.head(3))

# Check the distribution of outcome
print(data['Outcome'].value_counts())

# Map Outcome to numerical values
data['Outcome'] = data['Outcome'].map({'Yes': 1, 'No': 0})

# Split the data into features and target variable
X = data[["Age ", "(BMI)", "Number of embryo(s) transfered? "]]
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)
print(y_pred)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


#Save the trained model
joblib.dump(model, filename="pregnancy_outcome.pkl")

# Load the trained model (for testing purpose)
model = joblib.load('pregnancy_outcome.pkl')








