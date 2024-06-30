from flask import Flask, render_template, request, send_file
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('RandomForest.pkl', 'rb') as file:
    classifier = pickle.load(file)

# Load the dataset
df = pd.read_csv('Crop_recommendation.csv')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorus'])
        K = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        
        prediction = classifier.predict([[N, P, K, temperature, humidity, ph, rainfall]])
        
        crops = [
            'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 
            'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 
            'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 
            'rice', 'watermelon'
        ]
        
        result = crops[prediction[0]]
        
        return render_template('index.html', result=result)


@app.route('/download-dataset')
def download_dataset():
    return send_file('Crop_recommendation.csv', as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
