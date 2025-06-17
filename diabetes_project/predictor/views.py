# predictor/views.py
import joblib
import numpy as np
from django.shortcuts import render

# Load once at the top-level to avoid reloading every time
model = joblib.load('predictor/model/diabetes_model.pkl')

def predict_diabetes(request):
    result = None
    if request.method == 'POST':
        data = [
            float(request.POST['pregnancies']),
            float(request.POST['glucose']),
            float(request.POST['blood_pressure']),
            float(request.POST['skin_thickness']),
            float(request.POST['insulin']),
            float(request.POST['bmi']),
            float(request.POST['diabetes_pedigree']),
            float(request.POST['age']),
        ]
        prediction = model.predict([data])[0]
        result = 'Positive' if prediction == 1 else 'Negative'
    
    return render(request, 'predictor/form.html', {'result': result})
