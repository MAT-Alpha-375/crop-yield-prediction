# Crop Yield Prediction App 🌾

An AI-powered agricultural yield prediction system built with Streamlit and multiple machine learning models including CatBoost, Random Forest, and XGBoost.

## 🚀 Features

- **Multi-Model Prediction**: Uses CatBoost, Random Forest, and XGBoost models for accurate yield predictions
- **Interactive Web Interface**: Beautiful Streamlit-based UI with modern styling
- **Feature Importance Analysis**: SHAP-based explanations for model predictions
- **Multiple Crop Types**: Support for various agricultural crops
- **Global Coverage**: Predictions for multiple countries
- **Real-time Predictions**: Instant yield estimates based on input parameters

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Machine Learning**: 
  - CatBoost
  - Random Forest (scikit-learn)
  - XGBoost
- **Data Processing**: Pandas, NumPy
- **Model Explainability**: SHAP
- **Model Persistence**: Joblib

## 📋 Requirements

- Python 3.10+
- See `requirements.txt` for detailed package versions

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd crop-yield-prediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── FinalDataset.csv       # Training dataset
├── catboost_model/        # CatBoost model files
├── rf_model/              # Random Forest model files
├── xgb_model/             # XGBoost model files
└── README.md              # This file
```

## 🎯 Usage

1. Open the application in your web browser
2. Select the crop type from the dropdown
3. Choose the country
4. Input agricultural parameters (temperature, rainfall, etc.)
5. Click "Predict Yield" to get instant predictions
6. View feature importance explanations using SHAP

## 🤖 Models

The application uses three different machine learning models:

- **CatBoost**: Gradient boosting on decision trees with categorical features
- **Random Forest**: Ensemble of decision trees for robust predictions
- **XGBoost**: Extreme gradient boosting for high-performance predictions

## 📊 Dataset

The `FinalDataset.csv` contains agricultural data including:
- Crop types
- Environmental factors
- Geographic information
- Historical yield data

## 🔧 Configuration

The application automatically loads pre-trained models from their respective directories. Ensure all model files are present:
- `catboost_model/catboost_yield_model.cbm`
- `rf_model/Yield_Prediction_rf_model.pkl`
- `xgb_model/xgb_model.pkl`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Muhammad Ahsan Tariq - Initial work

## 🙏 Acknowledgments

- Streamlit team for the amazing web framework
- CatBoost, scikit-learn, and XGBoost communities
- Agricultural research community for datasets and insights

## 📞 Support

If you have any questions or need support, please open an issue on GitHub.

---

**Note**: This is a research project for agricultural yield prediction. Predictions should be used as guidance and not as the sole basis for agricultural decisions.
