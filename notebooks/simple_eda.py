import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load predictive maintenance dataset"""
    print("📥 LOADING DATASET...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
    
    try:
        data = pd.read_csv(url)
        print("✅ Dataset loaded successfully!")
        print(f"📊 Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def basic_analysis(data):
    """Perform basic data analysis"""
    print("\n" + "="*60)
    print("📋 BASIC DATA ANALYSIS")
    print("="*60)
    
    print(f"Dataset Shape: {data.shape}")
    print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
    
    print("\n📄 FIRST 5 ROWS:")
    print(data.head())
    
    print("\n🔍 COLUMN NAMES:")
    print(data.columns.tolist())
    
    print("\n📊 DATA TYPES:")
    print(data.dtypes)
    
    print("\n❓ MISSING VALUES:")
    missing = data.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values!")
    
    print("\n📈 BASIC STATISTICS:")
    print(data.describe())

def target_analysis(data):
    """Analyze the target variable"""
    print("\n" + "="*60)
    print("🎯 TARGET VARIABLE ANALYSIS")
    print("="*60)
    
    if 'Machine failure' in data.columns:
        target_counts = data['Machine failure'].value_counts()
        print("Target Distribution:")
        print(target_counts)
        
        failure_rate = data['Machine failure'].mean() * 100
        print(f"\n📊 Failure Rate: {failure_rate:.2f}%")
        
        # Plot target distribution
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.pie(target_counts.values, labels=['No Failure', 'Failure'], 
                autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        plt.title('Machine Failure Distribution')
        
        plt.subplot(1, 2, 2)
        sns.countplot(x='Machine failure', data=data, palette='viridis')
        plt.title('Failure Count Plot')
        plt.xlabel('Machine Failure (0=No, 1=Yes)')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('../data/target_distribution.png')
        plt.show()
        
    else:
        print("❌ 'Machine failure' column not found!")
        print("Available columns:", data.columns.tolist())

def correlation_analysis(data):
    """Analyze correlations between features"""
    print("\n" + "="*60)
    print("📊 CORRELATION ANALYSIS")
    print("="*60)
    
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    plt.figure(figsize=(12, 8))
    correlation_matrix = numeric_data.corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                center=0, fmt='.2f', square=True)
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig('../data/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print top correlations with target
    if 'Machine failure' in correlation_matrix.columns:
        target_corr = correlation_matrix['Machine failure'].sort_values(ascending=False)
        print("\n🔝 TOP CORRELATIONS WITH TARGET:")
        print(target_corr)

def feature_distributions(data):
    """Plot distributions of key features"""
    print("\n" + "="*60)
    print("📈 FEATURE DISTRIBUTIONS")
    print("="*60)
    
    # Key features to analyze
    features = ['Air temperature [K]', 'Process temperature [K]', 
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(features):
        if feature in data.columns:
            # Histogram
            data[feature].hist(bins=30, ax=axes[i], color='skyblue', 
                              edgecolor='black', alpha=0.7)
            axes[i].set_title(f'Distribution of {feature}', fontsize=12)
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
            
            # Add statistics
            mean_val = data[feature].mean()
            std_val = data[feature].std()
            axes[i].axvline(mean_val, color='red', linestyle='--', 
                           label=f'Mean: {mean_val:.2f}')
            axes[i].axvline(mean_val + std_val, color='orange', linestyle='--', 
                           alpha=0.7, label=f'±1 STD')
            axes[i].axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7)
            axes[i].legend()
    
    # Hide empty subplots
    for i in range(len(features), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('../data/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def failure_analysis(data):
    """Analyze features by failure status"""
    print("\n" + "="*60)
    print("🔧 FAILURE ANALYSIS BY FEATURES")
    print("="*60)
    
    if 'Machine failure' not in data.columns:
        print("❌ Cannot perform failure analysis - target column missing")
        return
    
    features = ['Air temperature [K]', 'Process temperature [K]', 
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(features):
        if feature in data.columns:
            # Box plot
            sns.boxplot(x='Machine failure', y=feature, data=data, ax=axes[i], palette='Set2')
            axes[i].set_title(f'{feature} vs Failure', fontsize=12)
            axes[i].set_xlabel('Machine Failure (0=No, 1=Yes)')
            axes[i].set_ylabel(feature)
    
    # Hide empty subplots
    for i in range(len(features), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Feature Distribution by Machine Failure Status', fontsize=16)
    plt.tight_layout()
    plt.savefig('../data/failure_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run all analysis"""
    print("🚀 STARTING PREDICTIVE MAINTENANCE EDA")
    print("="*60)
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Perform analyses
    basic_analysis(data)
    target_analysis(data)
    correlation_analysis(data)
    feature_distributions(data)
    failure_analysis(data)
    
    print("\n" + "="*60)
    print("✅ EDA COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("📁 Generated files in /data folder:")
    print("   - target_distribution.png")
    print("   - correlation_matrix.png") 
    print("   - feature_distributions.png")
    print("   - failure_analysis.png")

# Run the analysis
if __name__ == "__main__":
    main()