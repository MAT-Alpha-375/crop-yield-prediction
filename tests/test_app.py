"""
Tests for the Crop Yield Prediction App
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import the app functions (you may need to adjust imports based on your app structure)
# from app import load_models, predict_yield, etc.

class TestDataLoading:
    """Test data loading functionality"""
    
    def test_dataset_loading(self):
        """Test that the dataset can be loaded"""
        try:
            df = pd.read_csv("FinalDataset.csv")
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert 'Crop Type' in df.columns
            assert 'Country' in df.columns
        except FileNotFoundError:
            pytest.skip("Dataset file not found")
    
    def test_crop_types_extraction(self):
        """Test that crop types can be extracted from dataset"""
        try:
            df = pd.read_csv("FinalDataset.csv")
            crop_types = df['Crop Type'].unique()
            assert len(crop_types) > 0
            assert all(isinstance(ct, str) for ct in crop_types)
        except FileNotFoundError:
            pytest.skip("Dataset file not found")
    
    def test_countries_extraction(self):
        """Test that countries can be extracted from dataset"""
        try:
            df = pd.read_csv("FinalDataset.csv")
            countries = df['Country'].unique()
            assert len(countries) > 0
            assert all(isinstance(c, str) for c in countries)
        except FileNotFoundError:
            pytest.skip("Dataset file not found")

class TestModelLoading:
    """Test model loading functionality"""
    
    def test_catboost_model_exists(self):
        """Test that CatBoost model file exists"""
        import os
        model_path = "Catboost Model/catboost_yield_model.cbm"
        assert os.path.exists(model_path), f"CatBoost model not found at {model_path}"
    
    def test_random_forest_model_exists(self):
        """Test that Random Forest model file exists"""
        import os
        model_path = "RF_Model/Yield_Prediction_RF_Model.pkl"
        assert os.path.exists(model_path), f"Random Forest model not found at {model_path}"
    
    def test_xgboost_model_exists(self):
        """Test that XGBoost model file exists"""
        import os
        model_path = "XGboost Model/xgb_model.pkl"
        assert os.path.exists(model_path), f"XGBoost model not found at {model_path}"

class TestRequirements:
    """Test that all required packages are available"""
    
    def test_required_packages(self):
        """Test that all required packages can be imported"""
        required_packages = [
            'streamlit',
            'pandas',
            'numpy',
            'joblib',
            'shap',
            'catboost',
            'xgboost',
            'matplotlib',
            'plotly'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError as e:
                pytest.fail(f"Failed to import {package}: {e}")

class TestAppStructure:
    """Test the overall app structure"""
    
    def test_app_file_exists(self):
        """Test that the main app file exists"""
        import os
        assert os.path.exists("app.py"), "Main app file not found"
    
    def test_requirements_file_exists(self):
        """Test that requirements file exists"""
        import os
        assert os.path.exists("requirements.txt"), "Requirements file not found"
    
    def test_readme_exists(self):
        """Test that README file exists"""
        import os
        assert os.path.exists("README.md"), "README file not found"

if __name__ == "__main__":
    pytest.main([__file__])
