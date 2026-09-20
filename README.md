<div align="center">

<img src="https://raw.githubusercontent.com/RITESH2127/HOUSE-PRICE-PREDICTION-TOOL/main/GAIL.svg.png" width="105" alt="GAIL India Limited">

# Boston House Price Prediction Tool

### End-to-End Machine Learning Application

**Machine Learning Internship Project — GAIL (India) Limited**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.64-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![CI](https://img.shields.io/github/actions/workflow/status/RITESH2127/HOUSE-PRICE-PREDICTION-TOOL/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/RITESH2127/HOUSE-PRICE-PREDICTION-TOOL/actions)

**[Live Demo](https://hc3gbqd52jpaghs25aj7vq.streamlit.app/)** · **[Source Code](https://github.com/RITESH2127/HOUSE-PRICE-PREDICTION-TOOL)**

</div>

---

## Executive Summary

This repository contains a complete **machine-learning regression application** developed during my **Machine Learning Internship at GAIL (India) Limited**.

The project was designed as an end-to-end ML workflow rather than a notebook-only experiment: data validation, exploratory analysis, preprocessing, model benchmarking, quantitative evaluation, model persistence, automated testing, CI, containerization, and an interactive Streamlit deployment.

### Project objective

Build a regression system that estimates the historical **median value of owner-occupied homes (MEDV)** from 13 housing and socioeconomic features, then expose the trained model through a browser-based application.

> **Internship:** Machine Learning Internship  
> **Organization:** GAIL (India) Limited  
> **Project:** Boston House Price Prediction Tool  
> **Application:** Streamlit interactive ML application

---

## Live Application

### Try the deployed model

**[Launch the Streamlit App →](https://hc3gbqd52jpaghs25aj7vq.streamlit.app/)**

The application accepts the 13 model inputs and returns an estimated MEDV value.

| Feature | Implementation |
|---|---|
| Interactive inference | Streamlit |
| Persisted model | Joblib |
| Input schema | 13 original features |
| Model loading | Repository-relative path |
| Deployment | Streamlit |
| Containerization | Docker |
| Validation | GitHub Actions + pytest |

---

## ML Lifecycle

```text
                    DATA
                     │
                     ▼
              Data Validation
                     │
                     ▼
              Exploratory Analysis
                     │
                     ▼
              Train / Test Split
                     │
                     ▼
              Preprocessing
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
      Baseline              Yeo-Johnson
      Pipeline             + StandardScaler
          │                     │
          └──────────┬──────────┘
                     ▼
              Model Benchmarking
                     │
                     ▼
               R² + RMSE
                     │
                     ▼
              Model Selection
                     │
                     ▼
              best_model.pkl
                     │
                     ▼
              Streamlit App
                     │
                     ▼
               Live Inference
```

---

## Model Benchmark

Four regression algorithms were evaluated:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor

The experiment evaluated both standard preprocessing and Yeo-Johnson transformed pipelines.

### Corrected evaluation results

The original experiment stored MSE values under an RMSE column. The table below reports the **correct RMSE**, calculated as the square root of those MSE values.

| Model | Baseline R² | Baseline RMSE | Transformed R² | Transformed RMSE |
|---|---:|---:|---:|---:|
| Linear Regression | 0.6688 | 4.9296 | 0.7315 | 4.4374 |
| Ridge | 0.6685 | 4.9308 | 0.7312 | 4.4394 |
| Lasso | 0.6501 | 5.0652 | 0.7207 | 4.5257 |
| **Random Forest** | **0.8921** | **2.8130** | 0.8909 | 2.8281 |

**Selection criterion:** lowest test-set RMSE.

The documented experiment therefore selected the **Random Forest baseline pipeline**.

---

## Why Pipelines?

The project uses Scikit-learn pipelines so preprocessing and the estimator remain together.

This provides a cleaner inference path:

```text
Raw User Input
      ↓
Same preprocessing used during training
      ↓
Trained estimator
      ↓
Prediction
```

This avoids manually reproducing preprocessing steps inside the Streamlit application and makes the saved artifact easier to reuse.

---

## Dataset

The repository contains the historical **Boston Housing benchmark dataset**:

- 506 observations
- 13 input features
- 1 target variable
- 80/20 train-test split
- `random_state=42`

The dataset is stored locally in `housing.csv`, so the training workflow does not depend on fetching the dataset from a remote URL at runtime.

### Features

| Feature | Meaning |
|---|---|
| CRIM | Per-capita crime rate |
| ZN | Residential land zoning |
| INDUS | Non-retail business acreage |
| CHAS | Charles River boundary indicator |
| NOX | Nitric oxide concentration |
| RM | Average rooms per dwelling |
| AGE | Older owner-occupied units |
| DIS | Distance to employment centres |
| RAD | Highway accessibility |
| TAX | Property-tax rate |
| PTRATIO | Pupil-teacher ratio |
| B | Historical benchmark demographic-derived feature |
| LSTAT | Lower-status population percentage |

**Target:** `MEDV` — median value of owner-occupied homes, expressed in thousands of dollars in the original benchmark.

> **Dataset note:** Boston Housing is a historical benchmark with known methodological and ethical limitations. This project is for educational, internship, and portfolio purposes and is **not a production real-estate valuation system**.

---

## Exploratory Data Analysis

The project includes an exploratory visualization of the numerical variables.

![Exploratory Data Analysis](https://raw.githubusercontent.com/RITESH2127/HOUSE-PRICE-PREDICTION-TOOL/main/Figure_1.png)

EDA was used to understand feature distributions and motivate comparison between baseline preprocessing and power-transformed pipelines.

---

## Technology Stack

### Machine Learning
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

### Modelling
- Linear Regression
- Ridge
- Lasso
- Random Forest
- StandardScaler
- Yeo-Johnson PowerTransformer
- Scikit-learn Pipeline

### Application
- Streamlit

### Engineering & Deployment
- Git
- GitHub
- GitHub Actions
- pytest
- Docker

---

## Repository Architecture

```text
HOUSE-PRICE-PREDICTION-TOOL/
│
├── .github/
│   └── workflows/
│       └── ci.yml              # Automated CI
│
├── .streamlit/
│   └── config.toml             # Streamlit configuration
│
├── tests/
│   └── test_model.py           # Model smoke test
│
├── GAIL.svg.png                # GAIL internship/project branding
├── Figure_1.png                # EDA visualization
├── housing.csv                 # Local dataset
├── best_model.pkl              # Persisted trained pipeline
├── main.py                     # Training + evaluation
├── notebook_RITESH.ipynb       # Exploratory notebook
├── results.csv                 # Experiment results
├── streamlit_app.py            # Web application
├── requirements.txt            # Dependencies
├── Dockerfile                  # Container definition
├── .gitignore
└── README.md
```

---

## Run Locally

### Requirements

- Python 3.12+
- pip
- Git

### 1. Clone

```bash
git clone https://github.com/RITESH2127/HOUSE-PRICE-PREDICTION-TOOL.git
cd HOUSE-PRICE-PREDICTION-TOOL
```

### 2. Create environment

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install

```bash
pip install -r requirements.txt
```

### 4. Launch

```bash
streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```

---

## Retrain From Scratch

The complete training workflow is available in `main.py`.

```bash
python main.py
```

The script:

1. Validates the dataset.
2. Creates the reproducible train/test split.
3. Trains all candidate pipelines.
4. Calculates R² and true RMSE.
5. Writes results to `results.csv`.
6. Selects the lowest-RMSE pipeline.
7. Serializes the selected pipeline to `best_model.pkl`.

---

## Docker

Build:

```bash
docker build -t house-price-prediction .
```

Run:

```bash
docker run --rm -p 8501:8501 house-price-prediction
```

Then open `http://localhost:8501`.

---

## Testing & CI

A model smoke test verifies that the persisted artifact can be loaded and can generate a prediction with the expected feature schema.

Run locally:

```bash
pytest -q
```

GitHub Actions runs the training workflow and test suite automatically on repository changes.

---

## Internship Learning Outcomes

This project represents practical experience across the full ML development lifecycle.

### Machine Learning
- Regression modelling
- Ensemble learning
- Feature transformation
- Standardization
- Model comparison
- R² and RMSE evaluation
- Model selection

### Software Engineering
- Modular training workflow
- Repository-relative paths
- Dependency management
- Model serialization
- Automated testing
- Continuous integration

### Deployment
- Streamlit application development
- Docker containerization
- Cloud deployment
- Interactive model inference

### Professional Outcome

The project demonstrates the transition from **experimental data science work to a reproducible, deployable machine-learning application**, reflecting the practical focus of my **Machine Learning Internship at GAIL (India) Limited**.

---

## Project Documentation

The repository contains the primary development artifacts:

- `notebook_RITESH.ipynb` — exploratory analysis
- `main.py` — training and model evaluation
- `results.csv` — experiment results
- `best_model.pkl` — persisted model pipeline
- `streamlit_app.py` — deployed application

---

## Responsible Use

This project should not be used for real-world property valuation.

The benchmark is historical, its feature space is limited, and its data does not represent present-day housing markets. Predictions should not be interpreted as financial, investment, lending, or real-estate advice.

---

## Internship Acknowledgement

This project was developed as part of my **Machine Learning Internship at GAIL (India) Limited**.

The internship provided hands-on exposure to applying machine-learning concepts across:

**Data → Analysis → Modelling → Evaluation → Deployment**

**Organization:** GAIL (India) Limited  
**Internship Domain:** Machine Learning / Data Science  
**Project:** Boston House Price Prediction Tool  
**Deployment:** Streamlit

---

## Author

<div align="center">

### Ritesh Kumar

**B.Tech — Computer Science Engineering**

Machine Learning · Artificial Intelligence · Data Science

[GitHub](https://github.com/RITESH2127)

</div>

---

<div align="center">

### Built as a Machine Learning Internship Project at GAIL (India) Limited

**[Launch Live Application →](https://hc3gbqd52jpaghs25aj7vq.streamlit.app/)**

</div>

---

### Disclaimer

This repository is intended for **educational, internship, demonstration, and portfolio purposes**. The dataset and model are not intended for production property valuation or financial decision-making.
