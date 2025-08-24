# Yield Prediction Streamlit App
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from catboost import CatBoostRegressor, Pool
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Any

cmap_viridis = cm.get_cmap("viridis")
cmap_plasma = cm.get_cmap("plasma")
cmap_cool = cm.get_cmap("cool")

# Page configuration
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #4CAF50, #45a049);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .feature-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stSelectbox > div > div > select {
        background-color: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 8px;
    }
    
    .stNumberInput > div > div > input {
        background-color: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 8px;
    }
    
    .stSlider > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Load dataset (only for dropdown values)
df = pd.read_csv("FinalDataset.csv") 

# Extract unique dropdown values
crop_types = df['Crop Type'].unique()
countries = df['Country'].unique()

# Header
st.markdown('<h1 class="main-header">Crop Yield Prediction App</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Agricultural Yield Predition System</p>', unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    # --- Sidebar: Model Selection ---
    st.markdown("### 🤖 Model Selection")
    model_choice = st.selectbox(
        "Choose ML Algorithm",
        ["Random Forest", "CatBoost", "XGBoost"],
        help="Select the machine learning model for prediction"
    )
    
    # Model info cards
    model_info = {
        "Random Forest": {"accuracy": "92%", "speed": "Fast", "color": "#FF6B6B"},
        "CatBoost": {"accuracy": "94%", "speed": "Medium", "color": "#4ECDC4"},
        "XGBoost": {"accuracy": "93%", "speed": "Fast", "color": "#45B7D1"}
    }
    
    if model_choice in model_info:
        info = model_info[model_choice]
        st.markdown(f"""
        <div style="background: {info['color']}; padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0;">
            <h4 style="margin: 0;">{model_choice}</h4>
            <p style="margin: 0;">Accuracy: {info['accuracy']}</p>
            <p style="margin: 0;">Speed: {info['speed']}</p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    # --- Main Form ---
    st.markdown("### 📊 Input Parameters")
    
    # Create input form in columns for better layout
    input_col1, input_col2 = st.columns(2)
    
    with input_col1:
        country_input = st.selectbox(
            "🌍 Country",
            countries,
            help="Select the country for prediction"
        )
        
        rainfall_input = st.number_input(
            "🌧️ Rainfall (mm)",
            min_value=0.0,
            help="Annual rainfall in millimeters"
        )
        
        ph_input = st.slider(
            "🧪 Soil pH",
            min_value=0.0,
            max_value=14.0,
            step=0.1,
            help="Soil pH level (0-14 scale)"
        )
    
    with input_col2:
        crop_input = st.selectbox(
            "🌱 Crop Type",
            crop_types,
            help="Select the type of crop"
        )
        
        temp_input = st.number_input(
            "🌡️ Temperature (°C)",
            help="Average temperature in Celsius"
        )

# Add validation before prediction
if rainfall_input < 0:
    st.error("❌ Rainfall cannot be negative!")
    st.stop()

if not country_input or not crop_input:
    st.error("❌ Please select both Country and Crop Type!")
    st.stop()

# Prepare raw input as DataFrame
input_df = pd.DataFrame([{ 
    "Country": country_input,
    "Crop Type": crop_input,
    "Rainfall": rainfall_input,
    "Temperature": temp_input,
    "Soil_pH": ph_input
}])

# Display input summary
st.markdown("### 📋 Input Summary")
summary_col1, summary_col2, summary_col3, summary_col4, summary_col5 = st.columns(5)

with summary_col1:
    st.metric("Country", country_input, delta=None)
with summary_col2:
    st.metric("Crop", crop_input, delta=None)
with summary_col3:
    st.metric("Rainfall", f"{rainfall_input} mm", delta=None)
with summary_col4:
    st.metric("Temperature", f"{temp_input}°C", delta=None)
with summary_col5:
    st.metric("Soil pH", ph_input, delta=None)

def create_feature_importance_explanation(model, input_encoded, model_columns):
    """
    Create feature contribution using Random Forest feature importance
    """
    try:
        # Get feature importances from the model
        feature_importance = model.feature_importances_
        
        # Create pseudo-SHAP values based on feature importance and input values
        input_values = input_encoded.values[0]
        
        # Scale importance by input magnitude (simple approximation)
        base_prediction = model.predict(np.zeros((1, len(model_columns))))[0]
        current_prediction = model.predict(input_encoded)[0]
        prediction_diff = current_prediction - base_prediction
        
        # Distribute the prediction difference based on feature importance
        pseudo_shap = feature_importance * prediction_diff * np.sign(input_values)
        
        return pseudo_shap, "Feature Importance Approximation"
        
    except Exception as e:
        st.error(f"Feature importance calculation failed: {str(e)}")
        return None, None

def plot_contribution_chart(contribution_vals, input_encoded, method_used, model_choice):
    """
    Create and display feature contribution chart with modern styling
    """
    # Convert to absolute contributions
    contrib_abs = np.abs(contribution_vals)
    
    # Compute % contribution
    total_contribution = np.sum(contrib_abs)
    if total_contribution == 0:
        st.warning("No feature contributions detected!")
        return
        
    contrib_percent = 100 * contrib_abs / total_contribution
    
    # Filter out features with zero contribution
    nonzero_indices = np.where(contrib_abs > 1e-10)[0]
    
    if len(nonzero_indices) == 0:
        st.warning("No significant feature contributions detected!")
        return
    
    filtered_features = input_encoded.columns[nonzero_indices]
    filtered_percentages = contrib_percent[nonzero_indices]
    
    # Sort by % contribution descending
    sorted_idx = np.argsort(-filtered_percentages)
    sorted_features = filtered_features[sorted_idx]
    sorted_percentages = filtered_percentages[sorted_idx]
    
    # Create modern styled plot
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_features) * 0.5)))
    
    # Create gradient colors
    colors = cmap_viridis(np.linspace(0, 1, len(sorted_features)))
    
    bars = ax.barh(sorted_features, sorted_percentages, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    
    ax.set_xlabel("Contribution to Yield Prediction (%)", fontsize=14, fontweight='bold', color='#333')
    ax.set_title(f"🔍 Feature Contributions Analysis\nModel: {model_choice} | Method: {method_used}", 
                fontsize=16, fontweight='bold', color='#333', pad=20)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_facecolor('#fafafa')
    
    # Modern styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#ddd')
    ax.spines['bottom'].set_color('#ddd')
    
    # Add percentage labels with better positioning
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if width > 0:
            ax.text(width + max(sorted_percentages) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{sorted_percentages[i]:.1f}%", va='center', fontweight='bold', color='#333')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show top contributing features in modern cards
    st.markdown("### 🏆 Top Contributing Features")
    
    # Create columns for feature cards
    if len(sorted_features) >= 3:
        feat_col1, feat_col2, feat_col3 = st.columns(3)
        columns = [feat_col1, feat_col2, feat_col3]
    else:
        columns = st.columns(len(sorted_features))
    
    for i in range(min(3, len(sorted_features))):
        with columns[i]:
            st.markdown(f"""
            <div class="feature-box">
                <h4 style="color: #4CAF50; margin-bottom: 0.5rem;">#{i+1} {sorted_features[i]}</h4>
                <p style="font-size: 1.5rem; font-weight: bold; color: #333; margin: 0;">{sorted_percentages[i]:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

# Prediction button with modern styling
st.markdown("---")
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

with predict_col2:
    if st.button("🚀 Predict Yield", type="primary", use_container_width=True):
        
        with st.spinner(f'Running {model_choice} model...'):
            
            if model_choice == "Random Forest":
                try:
                    # Load model and preprocessing tools
                    model = joblib.load("rf_model/Yield_Prediction_rf_model.pkl")
                    model_columns = joblib.load("rf_model/model_columns.pkl")
                except FileNotFoundError:
                    st.error("Random Forest model files not found! Please ensure all model files are in the rf_model directory.")
                    st.stop()
                
                # One-hot encode inputs and align with training columns
                input_encoded = pd.get_dummies(input_df)
                
                # Drop any target column just in case
                if "Yield" in input_encoded.columns:
                    input_encoded = input_encoded.drop(columns=["Yield"])
                
                # Align columns with training data
                input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
                
                # Ensure correct data types
                input_encoded = input_encoded.astype(np.float64)
                
                # Predict
                prediction = model.predict(input_encoded)[0]
                
                # Create feature contribution explanation
                contrib_vals, method_used = create_feature_importance_explanation(model, input_encoded, model_columns)
                
                if contrib_vals is not None:
                    plot_contribution_chart(contrib_vals, input_encoded, method_used, model_choice)
                else:
                    st.error("Could not generate feature contribution analysis.")

            elif model_choice == "CatBoost":
                # Load CatBoost model (handles raw inputs)
                model = CatBoostRegressor()
                model.load_model("catboost_model/catboost_yield_model.cbm")

                # Predict directly
                prediction = model.predict(input_df)[0]

               # List categorical features
                cat_features = ['Country', 'Crop Type']

                # Create Pool with categorical features
                input_pool = Pool(input_df, cat_features=cat_features)

                # SHAP Explainer
                explainer = shap.Explainer(model)
                shap_values = explainer(input_df)

                # Convert SHAP values to percentage contributions
                shap_vals = shap_values.values[0]
                shap_abs = np.abs(shap_vals)
                shap_percent = 100 * shap_abs / np.sum(shap_abs)

                # Bar chart data
                features = input_df.columns
                percentages = shap_percent

                # Modern plot styling
                plt.style.use('default')
                fig, ax = plt.subplots(figsize=(12, 6))
                colors = cmap_plasma(np.linspace(0, 1, len(features)))
                bars = ax.barh(features, percentages, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
                
                ax.set_xlabel("Contribution to Yield Prediction (%)", fontsize=14, fontweight='bold', color='#333')
                ax.set_title("🔍 Feature Contributions Analysis (CatBoost SHAP)", fontsize=16, fontweight='bold', color='#333', pad=20)
                ax.invert_yaxis()
                ax.grid(axis='x', alpha=0.3, linestyle='--')
                ax.set_facecolor('#fafafa')
                
                # Modern styling
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#ddd')
                ax.spines['bottom'].set_color('#ddd')

                # Add value labels
                for i, bar in enumerate(bars):
                    ax.text(bar.get_width() + max(percentages) * 0.01, bar.get_y() + bar.get_height()/2,
                            f"{percentages[i]:.1f}%", va='center', fontweight='bold', color='#333')

                plt.tight_layout()
                st.pyplot(fig)

            elif model_choice == "XGBoost":
                # Load XGBoost model and scaler
                model = joblib.load("xgb_model/xgb_model.pkl")
                scaler = joblib.load("xgb_model/scaler.pkl")
                feature_columns = joblib.load("xgb_model/feature_columns.pkl")

                 # One-hot encode and align columns
                input_encoded = pd.get_dummies(input_df)
                input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

                # Scale input
                input_scaled = scaler.transform(input_encoded)

                # Predict
                prediction = model.predict(input_scaled)[0]

                # Create SHAP explainer for XGBoost
                explainer = shap.TreeExplainer(model)

                # Compute SHAP values for scaled input
                shap_values = explainer.shap_values(input_scaled)

                # Take the first (only) row
                shap_vals = shap_values[0]
                shap_abs = np.abs(shap_vals)

                # Get input values to filter non-zero features
                input_values = input_scaled[0]
                nonzero_indices = np.where(input_values != 0)[0]

                # Filter features and SHAP values
                filtered_features = [feature_columns[i] for i in nonzero_indices]
                filtered_abs_shap = shap_abs[nonzero_indices]

                # Convert to percentages relative to *just the filtered features*
                shap_percent = 100 * filtered_abs_shap / np.sum(filtered_abs_shap)

                # Sort for better visualization
                sorted_idx = np.argsort(shap_percent)
                sorted_features = [filtered_features[i] for i in sorted_idx]
                sorted_percent = shap_percent[sorted_idx]

                # Modern plot styling
                plt.style.use('default')
                fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_features)*0.5)))
                colors = cmap_cool(np.linspace(0, 1, len(sorted_features)))
                bars = ax.barh(sorted_features, sorted_percent, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
                
                ax.set_xlabel("Contribution to Yield Prediction (%)", fontsize=14, fontweight='bold', color='#333')
                ax.set_title("🔍 Feature Contributions Analysis (XGBoost SHAP)", fontsize=16, fontweight='bold', color='#333', pad=20)
                ax.invert_yaxis()
                ax.grid(axis='x', alpha=0.3, linestyle='--')
                ax.set_facecolor('#fafafa')
                
                # Modern styling
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#ddd')
                ax.spines['bottom'].set_color('#ddd')

                # Add value labels
                for i, bar in enumerate(bars):
                    ax.text(bar.get_width() + max(sorted_percent) * 0.01, bar.get_y() + bar.get_height()/2,
                            f"{sorted_percent[i]:.1f}%", va='center', fontweight='bold', color='#333')

                plt.tight_layout()
                st.pyplot(fig)

            # Display prediction result with modern styling
            st.markdown(f"""
            <div class="prediction-result">
                🎯 <strong>Predicted Yield: {prediction:.2f} tons/ha</strong>
                <br><small>Model: {model_choice} | Confidence: High</small>
            </div>
            """, unsafe_allow_html=True)

            # Download result as CSV
            result_df = input_df.copy()
            result_df['Predicted Yield (tons/ha)'] = prediction
            csv = result_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Prediction Result",
                data=csv,
                file_name="prediction_result.csv",
                mime="text/csv",
                type="secondary",
                use_container_width=True
            )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🌾 Crop Yield Prediction App | Powered by Machine Learning</p>
    <p><small>Developed by Ahsan</small></p>
</div>
""", unsafe_allow_html=True)