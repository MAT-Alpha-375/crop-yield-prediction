# Project Structure Documentation

## Overview
This document provides a detailed explanation of the Crop Yield Prediction App project structure, including the purpose of each file and directory.

## Root Directory Structure

```
crop-yield-prediction/
├── 📁 .github/                    # GitHub-specific configurations
├── 📁 .vscode/                    # VS Code editor settings
├── 📁 Catboost Model/             # CatBoost machine learning model files
├── 📁 RF_Model/                   # Random Forest model files
├── 📁 XGboost Model/              # XGBoost model files
├── 📁 docs/                       # Project documentation
├── 📁 myenv/                      # Python virtual environment (excluded from Git)
├── 📁 tests/                      # Test files and test configuration
├── 📁 venv/                       # Alternative virtual environment (excluded from Git)
├── 📄 .gitignore                  # Git ignore patterns
├── 📄 .pre-commit-config.yaml     # Pre-commit hooks configuration
├── 📄 CONTRIBUTING.md             # Contribution guidelines
├── 📄 Dockerfile                  # Docker container configuration
├── 📄 LICENSE                     # MIT License file
├── 📄 Makefile                    # Build and development commands
├── 📄 README.md                   # Main project documentation
├── 📄 app.py                      # Main Streamlit application
├── 📄 docker-compose.yml          # Docker Compose configuration
├── 📄 init_git.py                 # Git repository initialization script
├── 📄 pyproject.toml              # Modern Python project configuration
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                    # Traditional Python setup configuration
└── 📄 FinalDataset.csv            # Training dataset (17MB)
```

## Detailed File Descriptions

### Core Application Files

#### `app.py` (19KB, 502 lines)
- **Purpose**: Main Streamlit web application
- **Features**: 
  - Multi-model yield prediction (CatBoost, Random Forest, XGBoost)
  - Interactive web interface with modern styling
  - SHAP-based feature importance analysis
  - Real-time predictions
- **Key Components**:
  - Streamlit UI components
  - Model loading and prediction logic
  - Data visualization with Plotly and Matplotlib
  - Custom CSS styling

#### `FinalDataset.csv` (17MB)
- **Purpose**: Training dataset for the machine learning models
- **Content**: Agricultural data including crop types, environmental factors, geographic information, and historical yield data
- **Note**: Large file - consider using Git LFS for version control

### Machine Learning Models

#### `Catboost Model/` Directory
- **Purpose**: Contains CatBoost gradient boosting model
- **Files**:
  - `catboost_yield_model.cbm`: Trained CatBoost model file

#### `RF_Model/` Directory
- **Purpose**: Contains Random Forest model
- **Files**:
  - `Yield_Prediction_RF_Model.pkl`: Trained Random Forest model
  - `model_columns.pkl`: Feature column names for the model

#### `XGboost Model/` Directory
- **Purpose**: Contains XGBoost model
- **Files**:
  - `xgb_model.pkl`: Trained XGBoost model
  - `feature_columns.pkl`: Feature column names for the model
  - `scaler.pkl`: Data scaler for feature normalization

### Configuration Files

#### `requirements.txt`
- **Purpose**: Python package dependencies
- **Key Packages**:
  - `streamlit==1.47.1`: Web framework
  - `catboost==1.2.8`: CatBoost ML library
  - `scikit-learn==1.6.1`: Random Forest and utilities
  - `xgboost==3.0.2`: XGBoost ML library
  - `shap==0.48.0`: Model explainability
  - `pandas==2.2.2`: Data manipulation
  - `numpy==2.2.6`: Numerical computing

#### `pyproject.toml`
- **Purpose**: Modern Python project configuration
- **Features**:
  - Build system configuration
  - Project metadata
  - Development dependencies
  - Tool configurations (Black, Flake8, MyPy, pytest)

#### `setup.py`
- **Purpose**: Traditional Python package setup
- **Features**:
  - Package installation configuration
  - Entry points
  - Package data inclusion

### Development and Quality Assurance

#### `.github/` Directory
- **Purpose**: GitHub-specific configurations
- **Contents**:
  - `workflows/ci.yml`: GitHub Actions CI/CD pipeline
  - `ISSUE_TEMPLATE/`: Issue templates for bug reports and feature requests

#### `tests/` Directory
- **Purpose**: Test files and test configuration
- **Contents**:
  - `__init__.py`: Package initialization
  - `test_app.py`: Application tests including data loading, model loading, and package availability

#### `.pre-commit-config.yaml`
- **Purpose**: Pre-commit hooks for code quality
- **Tools**:
  - Black (code formatting)
  - Flake8 (linting)
  - MyPy (type checking)
  - isort (import sorting)
  - Bandit (security checks)

#### `Makefile`
- **Purpose**: Common development commands
- **Key Commands**:
  - `make install`: Install dependencies
  - `make run`: Run the application
  - `make test`: Run tests
  - `make lint`: Run linting checks
  - `make format`: Format code

### Containerization and Deployment

#### `Dockerfile`
- **Purpose**: Docker container configuration
- **Features**:
  - Python 3.10 slim base image
  - Multi-stage build for optimization
  - Non-root user for security
  - Health checks
  - Port 8501 exposure

#### `docker-compose.yml`
- **Purpose**: Multi-container deployment configuration
- **Services**:
  - Main application container
  - Optional Nginx reverse proxy
  - Network and volume configuration

### Documentation

#### `README.md`
- **Purpose**: Main project documentation
- **Sections**:
  - Project overview and features
  - Installation instructions
  - Usage guide
  - Technology stack
  - Contributing guidelines

#### `CONTRIBUTING.md`
- **Purpose**: Contribution guidelines
- **Content**:
  - Development workflow
  - Code style requirements
  - Issue reporting guidelines
  - Pull request process

#### `LICENSE`
- **Purpose**: MIT License for open source distribution
- **Terms**: Permissive license allowing commercial use, modification, and distribution

### Utility Scripts

#### `init_git.py`
- **Purpose**: Automated Git repository initialization
- **Features**:
  - Git installation check
  - Repository initialization
  - Initial commit creation
  - Remote origin setup assistance

## Virtual Environments

### `myenv/` and `venv/` Directories
- **Purpose**: Python virtual environments
- **Status**: Excluded from Git (in .gitignore)
- **Usage**: Isolate project dependencies from system Python

## Development Workflow

### 1. Initial Setup
```bash
# Clone repository
git clone <repository-url>
cd crop-yield-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Development Commands
```bash
# Run the application
make run
# or
streamlit run app.py

# Run tests
make test

# Format code
make format

# Run all quality checks
make check-all
```

### 3. Docker Development
```bash
# Build and run with Docker
docker-compose up --build

# Or build manually
docker build -t crop-yield-prediction .
docker run -p 8501:8501 crop-yield-prediction
```

## File Size Considerations

### Large Files
- `FinalDataset.csv` (17MB): Consider using Git LFS for version control
- Model files: May be large but essential for application functionality

### Excluded Files
- Virtual environments (`venv/`, `myenv/`)
- Python cache files (`__pycache__/`)
- IDE-specific files (`.vscode/`)
- Build artifacts and temporary files

## Security Considerations

- Non-root user in Docker containers
- Environment variables for configuration
- Input validation in the application
- Secure dependency management

## Performance Considerations

- Model files loaded on application startup
- Efficient data processing with Pandas
- Optimized Streamlit components
- Docker layer caching for faster builds

## Maintenance

### Regular Tasks
- Update dependencies (`pip install -r requirements.txt`)
- Run tests (`make test`)
- Check code quality (`make check-all`)
- Update documentation as needed

### Monitoring
- Application health checks
- Model performance metrics
- User feedback and issue tracking
- Dependency security updates
