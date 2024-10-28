# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, \
    confusion_matrix, classification_report
import joblib  # For model saving

# Load a hypothetical dataset (you can replace it with real insurance data)
# Dataset might contain columns like: ['age', 'income', 'claims_history', 'policy_type', 'risk_score']
df = pd.read_csv('insurance_claims_data.csv')

# Data Preprocessing and Cleaning
df = df.dropna()  # Drop missing values for simplicity
df['claims_history'] = df['claims_history'].apply(
    lambda x: 1 if x == 'Yes' else 0)  # Convert categorical to binary
df = pd.get_dummies(df, drop_first = True)  # Convert other categorical features to dummy variables

# Exploratory Data Analysis (EDA)
# Correlation heatmap
plt.figure(figsize = (10, 6))
sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Pairplot to visualize relationships between key features
sns.pairplot(df, hue = 'claims_history')
plt.show()

# Splitting data into features and target variable
X = df.drop('claims_history', axis = 1)
y = df['claims_history']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training

# Logistic Regression Model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Random Forest Model
rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf.fit(X_train, y_train)

# XGBoost Model
xgb = XGBClassifier(use_label_encoder = False, eval_metric = 'mlogloss')
xgb.fit(X_train, y_train)

# Model Evaluation

models = [logreg, rf, xgb]
model_names = ['Logistic Regression', 'Random Forest', 'XGBoost']

for model, name in zip(models, model_names):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, AUC-ROC: {roc_auc:.2f}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("-" * 50)

# Saving the best model (Assuming Random Forest performs best in this example)
joblib.dump(rf, 'insurance_risk_model.pkl')

# Mock Deployment (Creating an API for the model - Flask Example)
# Create a simple Flask app to deploy the model (Install Flask: pip install Flask)

from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('insurance_risk_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.get_json(force=True)

    # Convert data into a DataFrame
    df = pd.DataFrame([data])

    # Scale the data using the same scaler as the training set
    scaled_data = scaler.transform(df)

    # Make a prediction
    prediction = model.predict(scaled_data)

    return jsonify({'risk_prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)

"""# To run this app:
# 1. Save it as 'app.py'.
# 2. Use `python app.py` to run the server.
# 3. Send a POST request with customer data to `/predict` to get risk assessment."""










