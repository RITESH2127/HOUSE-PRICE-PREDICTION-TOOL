# Boston House Price Prediction Tool

> Machine Learning Internship Project — GAIL (India) Limited

An end-to-end regression project covering data exploration, preprocessing, model comparison, evaluation, model persistence, and Streamlit deployment.

## Live Demo

https://hc3gbqd52jpaghs25aj7vq.streamlit.app/

## Internship Context

This project was developed during my **Machine Learning Internship at GAIL (India) Limited**. The goal was to build a complete machine-learning workflow and deploy the resulting model as an interactive application.

## What the project does

- Loads the historical Boston Housing benchmark dataset.
- Performs exploratory data analysis.
- Splits data into training and testing sets using a fixed random seed.
- Compares Linear Regression, Ridge, Lasso, and Random Forest Regression.
- Evaluates models with R² and RMSE.
- Saves the selected model with Joblib.
- Provides real-time predictions through Streamlit.
- Includes reproducible dependencies and Docker support.

## Model Results

The original experiment stored MSE values under an RMSE column. The values below are corrected by taking the square root of those stored MSE values.

| Model | Baseline R² | Baseline RMSE | Transformed R² | Transformed RMSE |
|---|---:|---:|---:|---:|
| Linear Regression | 0.6688 | 4.9296 | 0.7315 | 4.4374 |
| Ridge | 0.6685 | 4.9308 | 0.7312 | 4.4394 |
| Lasso | 0.6501 | 5.0652 | 0.7207 | 4.5257 |
| Random Forest | 0.8921 | 2.8130 | 0.8909 | 2.8281 |

The documented experiment selected Random Forest based on the lowest RMSE.

## Run Locally

```bash
git clone https://github.com/RITESH2127/HOUSE-PRICE-PREDICTION-TOOL.git
cd HOUSE-PRICE-PREDICTION-TOOL
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Streamlit:

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501.

## Docker

```bash
docker build -t house-price-prediction .
docker run --rm -p 8501:8501 house-price-prediction
```

## Project Structure

```text
HOUSE-PRICE-PREDICTION-TOOL/
├── .streamlit/
│   └── config.toml
├── GAIL.svg.png
├── Figure_1.png
├── housing.csv
├── best_model.pkl
├── main.py
├── notebook_RITESH.ipynb
├── results.csv
├── streamlit_app.py
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

## Technology Stack

Python · Pandas · NumPy · Scikit-learn · Joblib · Streamlit · Matplotlib · Seaborn · Docker

## Streamlit Application

The application loads the persisted model using a repository-relative path, so it does not depend on the current working directory. It accepts all 13 model features and passes them to the model with the original feature names.

## Dataset

The repository contains 506 observations and 13 input features plus the MEDV target.

The Boston Housing dataset is a historical machine-learning benchmark. This application is intended for educational and portfolio demonstration purposes and should not be treated as a production real-estate valuation system.

## Internship Skills Demonstrated

- Data preprocessing
- Exploratory data analysis
- Feature transformation
- Regression modelling
- Model evaluation
- R² and RMSE analysis
- Model persistence
- Streamlit development
- Deployment configuration
- Reproducible Python environments

## Author

**Ritesh Kumar**  
B.Tech Computer Science Engineering  
Machine Learning Intern — GAIL (India) Limited
