# src/data_loader.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    def __init__(self):
        self.data = None
        
    def load_data(self):
        """Load predictive maintenance dataset from URL"""
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
        try:
            self.data = pd.read_csv(url)
            print("✅ Dataset loaded successfully!")
            print(f"📊 Dataset shape: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def basic_info(self):
        """Display basic information about the dataset"""
        if self.data is not None:
            print("="*50)
            print("📋 DATASET BASIC INFORMATION")
            print("="*50)
            print(f"Shape: {self.data.shape}")
            print("\n📄 First 5 rows:")
            print(self.data.head())
            print("\n🔍 Dataset Info:")
            print(self.data.info())
            print("\n📈 Statistical Summary:")
            print(self.data.describe())
            print("\n❓ Missing Values:")
            print(self.data.isnull().sum())