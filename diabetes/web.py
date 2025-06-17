import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 1. Load dataset
df = pd.read_csv('diabetes_data.csv')  # Replace with your actual file path

# 2. Clean column names
df.columns = df.columns.str.strip()

# 3. Define features
categorical_features = [
    "gender", "family_diabetes", "hypertensive",
    "family_hypertension", "cardiovascular_disease",
    "stroke", "age_group"
]
target_col = 'diabetes'

# 4. Handle age_group if necessary
if 'age_group' not in df.columns or df['age_group'].isnull().any():
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 30, 45, 60, 100],
        labels=['Young', 'Mid', 'Senior', 'Elderly']
    ).astype(str)

# 5. Split features and target
X = df.drop(columns=target_col)
y = df[target_col]

# 6. Preprocessing
numerical_features = [col for col in X.columns if col not in categorical_features]
scaler = StandardScaler()
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer([
    ('num', scaler, numerical_features),
    ('cat', encoder, categorical_features)
])

# Fit preprocessor
preprocessor.fit(X)
X_processed = preprocessor.transform(X)

# Save preprocessor and feature order
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(X.columns.tolist(), 'feature_order.pkl')

# 7. Apply SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_processed, y)

# 8. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Create model directory
os.makedirs('models', exist_ok=True)

# 9. Logistic Regression
lr = LogisticRegression(max_iter=500)
lr.fit(X_train, y_train)
joblib.dump(lr, 'models/lr.pkl')

# 10. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, 'models/rf.pkl')

# 11. XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
joblib.dump(xgb, 'models/xgb.pkl')

# 12. LightGBM
lgb = LGBMClassifier(random_state=42)
lgb.fit(X_train, y_train)
joblib.dump(lgb, 'models/lgb.pkl')

# 13. Neural Network
nn = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
nn.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
nn.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
nn.save('models/nn.h5')

print("✅ All models trained and saved successfully using SMOTE.")
