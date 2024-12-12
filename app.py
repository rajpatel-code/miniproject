# app.py
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sb
from flask import Flask, request, jsonify, render_template

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Data loading and preprocessing
df = pd.read_csv('newdata.csv')

# Ensure 'Date' column is correctly parsed
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S%z', errors='coerce')

# Check for missing values in 'Date' column
if df['Date'].isnull().sum() > 0:
    print(f"There are {df['Date'].isnull().sum()} invalid dates. These will be removed.")
    df = df.dropna(subset=['Date'])  # Drop rows with invalid dates

# Extract year, month, and day
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day

# Create the 'is_quarter_end' column
df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)

# Feature engineering and target creation
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Define features and target
features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

# Scale the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Train-test split
X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=2022)

# Model training
models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]
for model in models:
    model.fit(X_train, Y_train)

# Home route to render the frontend
@app.route('/')
def home():
    return render_template('index.html')

# Prediction API route
@app.route('/predict', methods=['POST'])
def predict():
    # Input from frontend
    data = request.json
    open_close = data['open_close']
    low_high = data['low_high']
    is_quarter_end = data['is_quarter_end']

    # Convert to array and scale
    features = np.array([[open_close, low_high, is_quarter_end]])
    features_scaled = scaler.transform(features)

    # Prediction using Logistic Regression (for example)
    model = models[0]  # You can allow the user to select different models if needed
    prediction = model.predict(features_scaled)
    
    # Return prediction as JSON
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
