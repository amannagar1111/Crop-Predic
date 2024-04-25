from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)

df = pd.read_csv("Crop_recommendation.csv")

gb = GaussianNB()

scaler = MinMaxScaler()
X = df.drop('label', axis=1)
y = df['label']
X_scaled = scaler.fit_transform(X)
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        test_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        test_data_scaled = scaler.transform(test_data)

        gb.fit(X_train, y_train)
        predicted_crop = le.inverse_transform(gb.predict(test_data_scaled))

        if len(predicted_crop) > 0:
            return render_template('/result.html', predicted_crop=predicted_crop[0])
        else:
            return "Prediction not available."

