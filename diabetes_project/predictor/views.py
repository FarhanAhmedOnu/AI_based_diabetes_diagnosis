import joblib
import numpy as np
import pandas as pd
from django.shortcuts import render
from tensorflow.keras.models import load_model

# Load resources
feature_order = joblib.load('predictor/model/feature_order.pkl')
preprocessor = joblib.load('predictor/model/preprocessor.pkl')

models = {
    'lr': joblib.load('predictor/model/logistic_reg.pkl'),
    'rf': joblib.load('predictor/model/random_forest.pkl'),
    'xgb': joblib.load('predictor/model/xgboost.pkl'),
    'lgb': joblib.load('predictor/model/lightgbm.pkl'),
    'nn': load_model('predictor/model/neural_net.h5')
}

def predict_diabetes(request):
    result = None
    if request.method == 'POST':
        # Extract and organize input
        input_data = {}
        for key in feature_order:
            input_data[key] = request.POST.get(key)
        
        df_input = pd.DataFrame([input_data])

        # Apply same preprocessing
        X_processed = preprocessor.transform(df_input)
        
        # Choose model
        model_key = request.POST.get('model_choice')
        model = models[model_key]

        # Predict
        if model_key == 'nn':
            prediction = model.predict(X_processed)[0][0]
        else:
            prediction = model.predict(X_processed)[0]
        
        result = 'Positive' if prediction >= 0.5 else 'Negative'

    return render(request, 'predictor/form.html', {'result': result})
