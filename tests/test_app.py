"""
Tests for the Crop Yield Prediction App
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd
import pytest


def _exists_any(paths: Iterable[str]) -> bool:
    """Return True if any path in the iterable exists."""
    return any(os.path.exists(p) for p in paths)


class TestDataLoading:
    """Test data loading functionality"""

    def test_dataset_loading(self) -> None:
        """Test that the dataset can be loaded"""
        try:
            df: pd.DataFrame = pd.read_csv("FinalDataset.csv")
        except FileNotFoundError:
            pytest.skip("Dataset file not found")
            

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "Crop Type" in df.columns
        assert "Country" in df.columns

    def test_crop_types_extraction(self) -> None:
        """Test that crop types can be extracted from dataset"""
        try:
            df: pd.DataFrame = pd.read_csv("FinalDataset.csv")
        except FileNotFoundError:
            pytest.skip("Dataset file not found")
            

        crop_types = df["Crop Type"].unique()
        assert len(crop_types) > 0
        # handle numpy string dtype as well
        assert all(isinstance(ct, (str, np.str_)) for ct in crop_types)

    def test_countries_extraction(self) -> None:
        """Test that countries can be extracted from dataset"""
        try:
            df: pd.DataFrame = pd.read_csv("FinalDataset.csv")
        except FileNotFoundError:
            pytest.skip("Dataset file not found")
            

        countries = df["Country"].unique()
        assert len(countries) > 0
        assert all(isinstance(c, (str, np.str_)) for c in countries)


class TestModelLoading:
    """Test model loading functionality"""

    def test_catboost_model_exists(self) -> None:
        """Test that CatBoost model file exists"""
        paths = [
            "catboost_model/catboost_yield_model.cbm",   # new path
            "catboost_model/catboost_yield_model.cbm",   # old path (with space)
        ]
        assert _exists_any(paths), f"CatBoost model not found in any of: {paths}"

    def test_random_forest_model_exists(self) -> None:
        """Test that Random Forest model file exists"""
        paths = [
            "rf_model/Yield_Prediction_RF_Model.pkl",  # new path (snake_case)
            "rf_model/Yield_Prediction_rf_model.pkl",  # old path
        ]
        assert _exists_any(paths), f"RF model not found in any of: {paths}"

    def test_xgboost_model_exists(self) -> None:
        """Test that XGBoost model file exists"""
        paths = [
            "xgb_model/xgb_model.pkl",          # new path
            "xgb_model/xgb_model.pkl",      # old path (with space/case)
        ]
        assert _exists_any(paths), f"XGBoost model not found in any of: {paths}"


class TestRequirements:
    """Test that all required packages are available"""

    def test_required_packages(self) -> None:
        """Test that all required packages can be imported"""
        required_packages = [
            "streamlit",
            "pandas",
            "numpy",
            "joblib",
            "shap",
            "catboost",
            "xgboost",
            "matplotlib",
            "plotly",
        ]

        for package in required_packages:
            try:
                __import__(package)
            except ImportError as e:  # pragma: no cover
                pytest.fail(f"Failed to import {package}: {e}")


class TestAppStructure:
    """Test the overall app structure"""

    def test_app_file_exists(self) -> None:
        """Test that the main app file exists"""
        assert os.path.exists("app.py"), "Main app file not found"

    def test_requirements_file_exists(self) -> None:
        """Test that requirements file exists"""
        assert os.path.exists("requirements.txt"), "Requirements file not found"

    def test_readme_exists(self) -> None:
        """Test that README file exists"""
        assert os.path.exists("README.md"), "README file not found"
