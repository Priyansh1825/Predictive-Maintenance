import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess data"""
    print("📥 LOADING AND PREPROCESSING DATA...")
    
    # Load data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
    data = pd.read_csv(url)
    
    # Clean data
    data_clean = data.drop(columns=['UDI', 'Product ID'], errors='ignore')
    
    # Feature engineering
    data_eng = data_clean.copy()
    
    # Create new features
    if 'Air temperature [K]' in data.columns and 'Process temperature [K]' in data.columns:
        data_eng['Temperature_difference'] = data_eng['Process temperature [K]'] - data_eng['Air temperature [K]']
        data_eng['Temperature_ratio'] = data_eng['Process temperature [K]'] / data_eng['Air temperature [K]']
    
    if 'Rotational speed [rpm]' in data.columns and 'Torque [Nm]' in data.columns:
        data_eng['Power'] = data_eng['Rotational speed [rpm]'] * data_eng['Torque [Nm]']
        data_eng['Efficiency'] = data_eng['Torque [Nm]'] / (data_eng['Rotational speed [rpm]'] + 1e-5)
    
    if 'Tool wear [min]' in data.columns:
        data_eng['Tool_wear_category'] = pd.cut(data_eng['Tool wear [min]'], 
                                               bins=[0, 50, 150, 200, 300],
                                               labels=['Low', 'Medium', 'High', 'Critical'])
    
    # Encode categorical features
    if 'Type' in data_eng.columns:
        le_type = LabelEncoder()
        data_eng['Type_encoded'] = le_type.fit_transform(data_eng['Type'])
    
    if 'Tool_wear_category' in data_eng.columns:
        le_wear = LabelEncoder()
        data_eng['Tool_wear_encoded'] = le_wear.fit_transform(data_eng['Tool_wear_category'])
    
    # Prepare features and target
    target_column = 'Machine failure'
    exclude_columns = [target_column, 'Type', 'Tool wear [min]', 'Tool_wear_category']
    feature_columns = [col for col in data_eng.columns if col not in exclude_columns]
    
    X = data_eng[feature_columns]
    y = data_eng[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection
    feature_selector = SelectKBest(score_func=f_classif, k=10)
    X_train_selected = feature_selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = feature_selector.transform(X_test_scaled)
    
    return X_train_selected, X_test_selected, y_train, y_test, scaler, feature_selector, feature_columns

def train_best_model():
    """Train and save the best model"""
    print("🤖 TRAINING BEST MODEL FOR DEPLOYMENT...")
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler, feature_selector, feature_columns = load_and_preprocess_data()
    
    # Train Random Forest (usually best for this type of problem)
    print("🎯 Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"✅ Model trained successfully!")
    print(f"📊 Accuracy: {accuracy:.4f}")
    print(f"📊 F1-Score: {f1:.4f}")
    
    return model, scaler, feature_selector, feature_columns

def save_models():
    """Save the trained model and preprocessing objects"""
    print("💾 SAVING MODELS AND PREPROCESSORS...")
    
    # Train model
    model, scaler, feature_selector, feature_columns = train_best_model()
    
    # Create models directory if it doesn't exist
    import os
    os.makedirs('../models', exist_ok=True)
    
    # Save objects
    joblib.dump(model, '../models/predictive_maintenance_model.pkl')
    joblib.dump(scaler, '../models/scaler.pkl')
    joblib.dump(feature_selector, '../models/feature_selector.pkl')
    
    # Save feature columns
    import json
    with open('../models/feature_columns.json', 'w') as f:
        json.dump(feature_columns, f)
    
    print("✅ Models saved successfully!")
    print("📁 Saved files:")
    print("   - predictive_maintenance_model.pkl (Trained model)")
    print("   - scaler.pkl (Feature scaler)")
    print("   - feature_selector.pkl (Feature selector)")
    print("   - feature_columns.json (Feature names)")
    
    return True

if __name__ == "__main__":
    save_models()