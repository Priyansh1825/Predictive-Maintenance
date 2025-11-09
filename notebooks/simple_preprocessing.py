import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data():
    """Load and clean the predictive maintenance dataset"""
    print("📥 LOADING AND CLEANING DATA...")
    
    # Load data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
    data = pd.read_csv(url)
    
    print(f"✅ Original data shape: {data.shape}")
    
    # Basic cleaning
    # Remove unnecessary columns
    columns_to_drop = ['UDI', 'Product ID']  # ID columns not useful for prediction
    data_clean = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
    
    # Check for duplicates
    duplicates = data_clean.duplicated().sum()
    if duplicates > 0:
        print(f"⚠️  Found {duplicates} duplicates. Removing...")
        data_clean = data_clean.drop_duplicates()
    
    print(f"✅ Cleaned data shape: {data_clean.shape}")
    return data_clean

def feature_engineering(data):
    """Create new features from existing ones"""
    print("\n🔧 PERFORMING FEATURE ENGINEERING...")
    
    data_eng = data.copy()
    
    # 1. Temperature difference
    if 'Air temperature [K]' in data.columns and 'Process temperature [K]' in data.columns:
        data_eng['Temperature_difference'] = data_eng['Process temperature [K]'] - data_eng['Air temperature [K]']
        print("✅ Created: Temperature_difference")
    
    # 2. Power feature (Rotational speed * Torque)
    if 'Rotational speed [rpm]' in data.columns and 'Torque [Nm]' in data.columns:
        data_eng['Power'] = data_eng['Rotational speed [rpm]'] * data_eng['Torque [Nm]']
        print("✅ Created: Power")
    
    # 3. Tool wear categories
    if 'Tool wear [min]' in data.columns:
        data_eng['Tool_wear_category'] = pd.cut(data_eng['Tool wear [min]'], 
                                               bins=[0, 50, 150, 200, 300],
                                               labels=['Low', 'Medium', 'High', 'Critical'])
        print("✅ Created: Tool_wear_category")
    
    # 4. Temperature ratio
    if 'Air temperature [K]' in data.columns and 'Process temperature [K]' in data.columns:
        data_eng['Temperature_ratio'] = data_eng['Process temperature [K]'] / data_eng['Air temperature [K]']
        print("✅ Created: Temperature_ratio")
    
    # 5. Machine efficiency (hypothetical)
    if 'Rotational speed [rpm]' in data.columns and 'Torque [Nm]' in data.columns:
        data_eng['Efficiency'] = data_eng['Torque [Nm]'] / (data_eng['Rotational speed [rpm]'] + 1e-5)
        print("✅ Created: Efficiency")
    
    print(f"✅ Engineered features. New shape: {data_eng.shape}")
    return data_eng

def encode_categorical_features(data):
    """Encode categorical variables"""
    print("\n🔠 ENCODING CATEGORICAL FEATURES...")
    
    data_encoded = data.copy()
    
    # Encode Type column if it exists
    if 'Type' in data.columns:
        le_type = LabelEncoder()
        data_encoded['Type_encoded'] = le_type.fit_transform(data_encoded['Type'])
        print(f"✅ Encoded 'Type': {dict(zip(le_type.classes_, range(len(le_type.classes_))))}")
    
    # Encode tool wear category if it exists
    if 'Tool_wear_category' in data.columns:
        le_wear = LabelEncoder()
        data_encoded['Tool_wear_encoded'] = le_wear.fit_transform(data_encoded['Tool_wear_category'])
        print(f"✅ Encoded tool wear categories: {list(le_wear.classes_)}")
    
    return data_encoded

def prepare_features_target(data):
    """Prepare features (X) and target (y)"""
    print("\n🎯 PREPARING FEATURES AND TARGET...")
    
    # Define target column
    target_column = 'Machine failure'
    
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Select features (exclude non-feature columns)
    exclude_columns = [target_column, 'Type', 'Tool wear [min]', 'Tool_wear_category']
    
    feature_columns = [col for col in data.columns if col not in exclude_columns]
    
    X = data[feature_columns]
    y = data[target_column]
    
    print(f"✅ Features: {len(feature_columns)} columns")
    print(f"✅ Target: {target_column} (Failure rate: {y.mean():.2%})")
    print(f"✅ Feature names: {feature_columns}")
    
    return X, y, feature_columns

def scale_features(X_train, X_test):
    """Scale features using StandardScaler"""
    print("\n⚖️ SCALING FEATURES...")
    
    scaler = StandardScaler()
    
    # Scale numerical features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Features scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    print(f"\n📊 SPLITTING DATA (test_size={test_size})...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✅ Training set: {X_train.shape[0]} samples")
    print(f"✅ Test set: {X_test.shape[0]} samples")
    print(f"✅ Training failure rate: {y_train.mean():.2%}")
    print(f"✅ Test failure rate: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test

def feature_selection(X_train, y_train, X_test, k=10):
    """Select top k features using ANOVA F-test"""
    print(f"\n🔍 PERFORMING FEATURE SELECTION (top {k} features)...")
    
    feature_selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = feature_selector.fit_transform(X_train, y_train)
    X_test_selected = feature_selector.transform(X_test)
    
    # Get selected feature names
    if hasattr(X_train, 'columns'):
        selected_features = X_train.columns[feature_selector.get_support()].tolist()
        feature_scores = feature_selector.scores_[feature_selector.get_support()]
        
        print("🎯 TOP SELECTED FEATURES:")
        for feature, score in zip(selected_features, feature_scores):
            print(f"   - {feature}: {score:.2f}")
    
    print(f"✅ Selected {X_train_selected.shape[1]} features")
    
    return X_train_selected, X_test_selected, feature_selector

def full_preprocessing_pipeline():
    """Run complete preprocessing pipeline"""
    print("🚀 STARTING COMPLETE PREPROCESSING PIPELINE")
    print("="*60)
    
    # Step 1: Load and clean
    data = load_and_clean_data()
    
    # Step 2: Feature engineering
    data_eng = feature_engineering(data)
    
    # Step 3: Encode categorical features
    data_encoded = encode_categorical_features(data_eng)
    
    # Step 4: Prepare features and target
    X, y, feature_columns = prepare_features_target(data_encoded)
    
    # Step 5: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 6: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Step 7: Feature selection
    X_train_selected, X_test_selected, feature_selector = feature_selection(X_train_scaled, y_train, X_test_scaled)
    
    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    # Display final results
    print("\n📊 FINAL DATA SHAPES:")
    print(f"X_train: {X_train_selected.shape}")
    print(f"X_test: {X_test_selected.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    
    print(f"\n🎯 TRAINING SET FAILURE RATE: {y_train.mean():.2%}")
    print(f"🎯 TEST SET FAILURE RATE: {y_test.mean():.2%}")
    
    print(f"\n🔧 PREPROCESSING OBJECTS CREATED:")
    print(f"   - Scaler: {type(scaler).__name__}")
    print(f"   - Feature Selector: {type(feature_selector).__name__}")
    
    return {
        'X_train': X_train_selected,
        'X_test': X_test_selected,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_columns,
        'scaler': scaler,
        'feature_selector': feature_selector,
        'original_data': data
    }

# Run the preprocessing pipeline
if __name__ == "__main__":
    processed_data = full_preprocessing_pipeline()