# notebooks/simple_model_training.py
"""
SIMPLE MODEL BUILDING & TRAINING - FIXED VERSION
No imports needed - everything in one file
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_and_preprocess_data():
    """Load and preprocess data (same as Step 3)"""
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
    
    print(f"✅ Data preprocessing completed")
    print(f"✅ Training set: {X_train_selected.shape}")
    print(f"✅ Test set: {X_test_selected.shape}")
    print(f"✅ Failure rate: {y_train.mean():.2%}")
    
    return X_train_selected, X_test_selected, y_train, y_test, feature_columns, scaler, feature_selector

def initialize_models():
    """Initialize multiple ML models"""
    print("\n🤖 INITIALIZING MACHINE LEARNING MODELS...")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True),
    }
    
    print("✅ Models initialized:")
    for name, model in models.items():
        print(f"   - {name}: {type(model).__name__}")
    
    return models

def train_and_evaluate_models(models, X_train, X_test, y_train, y_test):
    """Train and evaluate all models"""
    print("\n🎯 TRAINING AND EVALUATING MODELS...")
    print("="*60)
    
    results = {}
    
    for name, model in models.items():
        print(f"\n📊 TRAINING {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"   ✅ Accuracy: {accuracy:.4f}")
        print(f"   ✅ F1-Score: {f1:.4f}")
        print(f"   ✅ ROC-AUC: {auc:.4f}" if auc else "   ✅ ROC-AUC: Not available")
    
    return results

def perform_hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for best models"""
    print("\n⚙️ PERFORMING HYPERPARAMETER TUNING...")
    
    # Define parameter grids
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5]
        }
    }
    
    tuned_models = {}
    
    for name, param_grid in param_grids.items():
        print(f"\n🎛️  Tuning {name}...")
        
        if name == 'Random Forest':
            model = RandomForestClassifier(random_state=42)
        else:
            model = GradientBoostingClassifier(random_state=42)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        tuned_models[name] = {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
        
        print(f"   ✅ Best parameters: {grid_search.best_params_}")
        print(f"   ✅ Best CV F1-score: {grid_search.best_score_:.4f}")
    
    return tuned_models

def cross_validation_evaluation(models, X_train, y_train):
    """Perform cross-validation for robust evaluation"""
    print("\n📊 CROSS-VALIDATION EVALUATION...")
    
    cv_results = {}
    
    for name, model in models.items():
        print(f"\n🔄 {name} Cross-Validation...")
        
        # Perform 5-fold cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        
        cv_results[name] = {
            'mean_f1': cv_scores.mean(),
            'std_f1': cv_scores.std(),
            'all_scores': cv_scores
        }
        
        print(f"   ✅ Mean F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_results

def plot_model_comparison(results, cv_results, X_test, y_test):
    """Create comparison plots of model performance - FIXED VERSION"""
    print("\n📈 CREATING MODEL COMPARISON PLOTS...")
    
    # Extract metrics for plotting
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    f1_scores = [results[name]['f1_score'] for name in model_names]
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Accuracy and F1-Score comparison
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    axes[0, 0].bar(x_pos - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='skyblue')
    axes[0, 0].bar(x_pos + width/2, f1_scores, width, label='F1-Score', alpha=0.8, color='lightcoral')
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('Scores')
    axes[0, 0].set_title('Model Performance: Accuracy vs F1-Score')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(model_names, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Cross-validation results
    if cv_results:
        cv_means = [cv_results[name]['mean_f1'] for name in model_names if name in cv_results]
        cv_stds = [cv_results[name]['std_f1'] for name in model_names if name in cv_results]
        cv_names = [name for name in model_names if name in cv_results]
        
        if cv_means:  # Only plot if we have CV results
            axes[0, 1].bar(cv_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.8, color='lightgreen')
            axes[0, 1].set_xlabel('Models')
            axes[0, 1].set_ylabel('F1-Score')
            axes[0, 1].set_title('Cross-Validation Performance (5-fold)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Confusion Matrix for best model
    best_model_name = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
    best_model = results[best_model_name]['model']
    y_pred_best = results[best_model_name]['predictions']
    
    # FIX: Use the provided y_test parameter
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}')
    
    # Plot 4: Feature Importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = best_model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(feature_importance)[::-1]
        sorted_importance = feature_importance[indices]
        
        axes[1, 1].bar(range(len(sorted_importance)), sorted_importance, color='orange')
        axes[1, 1].set_xlabel('Feature Rank')
        axes[1, 1].set_ylabel('Importance')
        axes[1, 1].set_title(f'Feature Importance - {best_model_name}')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # If no feature importance, show ROC curve if available
        if results[best_model_name]['probabilities'] is not None:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test, results[best_model_name]['probabilities'])
            axes[1, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {results[best_model_name]["roc_auc"]:.2f})')
            axes[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[1, 1].set_xlabel('False Positive Rate')
            axes[1, 1].set_ylabel('True Positive Rate')
            axes[1, 1].set_title(f'ROC Curve - {best_model_name}')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../data/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_report(results, cv_results, y_test):
    """Print detailed performance report - FIXED VERSION"""
    print("\n" + "="*70)
    print("📊 DETAILED MODEL PERFORMANCE REPORT")
    print("="*70)
    
    # Find best model
    best_model_name = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
    best_predictions = results[best_model_name]['predictions']
    
    print(f"\n🏆 BEST PERFORMING MODEL: {best_model_name}")
    print(f"   F1-Score: {results[best_model_name]['f1_score']:.4f}")
    print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
    if results[best_model_name]['roc_auc']:
        print(f"   ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
    
    print("\n📈 ALL MODEL RESULTS:")
    print("-" * 50)
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  F1-Score: {result['f1_score']:.4f}")
        if result['roc_auc']:
            print(f"  ROC-AUC:  {result['roc_auc']:.4f}")
        
        if name in cv_results:
            print(f"  CV F1:    {cv_results[name]['mean_f1']:.4f} (+/- {cv_results[name]['std_f1'] * 2:.4f})")
    
    print("\n🔍 CLASSIFICATION REPORT FOR BEST MODEL:")
    print("-" * 50)
    print(classification_report(y_test, best_predictions, target_names=['No Failure', 'Failure']))

def main():
    """Main function to run complete model training pipeline"""
    print("🚀 STARTING MODEL BUILDING & TRAINING PIPELINE")
    print("="*70)
    
    # Step 1: Load and preprocess data
    X_train, X_test, y_train, y_test, feature_columns, scaler, feature_selector = load_and_preprocess_data()
    
    # Step 2: Initialize models
    models = initialize_models()
    
    # Step 3: Train and evaluate base models
    results = train_and_evaluate_models(models, X_train, X_test, y_train, y_test)
    
    # Step 4: Cross-validation
    cv_results = cross_validation_evaluation(models, X_train, y_train)
    
    # Step 5: Hyperparameter tuning for best models
    tuned_models = perform_hyperparameter_tuning(X_train, y_train)
    
    # Step 6: Visualizations - FIXED: Pass y_test as parameter
    plot_model_comparison(results, cv_results, X_test, y_test)
    
    # Step 7: Detailed report - FIXED: Pass y_test as parameter
    print_detailed_report(results, cv_results, y_test)
    
    print("\n" + "="*70)
    print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    return {
        'results': results,
        'cv_results': cv_results,
        'tuned_models': tuned_models,
        'X_test': X_test,
        'y_test': y_test,
        'feature_columns': feature_columns
    }

# Run the complete pipeline
if __name__ == "__main__":
    training_results = main()