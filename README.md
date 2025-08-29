# Boston House Price Prediction: A Machine Learning Approach
![GAIL](https://github.com/RITESH2127/HOUSE-PRICE-PREDICTION-TOOL/blob/main/GAIL.svg.png?raw=true)
This repository contains the source code, analysis, and documentation for a machine learning project aimed at predicting Boston house prices. This project was undertaken as part of the summer training program with GAIL (India) Limited. It covers the entire machine learning lifecycle from data exploration and preprocessing to model training, evaluation, and deployment as an interactive web application.

## Table of Contents
1.  Project Overview
2.  Problem Statement
3.  Dataset
4.  Exploratory Data Analysis (EDA)
5.  Data Preprocessing & Feature Engineering
6.  Modeling and Evaluation
7.  Results
8.  Streamlit Web Application
9.  Project Structure
10. Setup and Instructions to Run

## Project Overview

The core objective of this project is to develop a robust regression model capable of accurately predicting the median value of owner-occupied homes in various Boston suburbs. The project employs a systematic approach, beginning with a thorough analysis of the dataset, followed by preprocessing to handle data irregularities. Several machine learning models are trained and rigorously evaluated to identify the best-performing algorithm. The final model is then deployed as a user-friendly web application using Streamlit, allowing for real-time price predictions based on user-provided house features.

## Problem Statement

To build a predictive model that can estimate the median value of homes (`MEDV`) in the Boston area based on 13 socio-economic and housing-related features. The success of the model will be measured by its predictive accuracy, specifically using R-squared (R²) and Root Mean Squared Error (RMSE) metrics.

## Dataset

The project utilizes the well-known Boston Housing dataset, which contains 506 data points and 14 attributes (13 features and 1 target variable). The data is loaded directly into the project via a URL in the `main.py` script.

**Features:**
*   **CRIM**: Per capita crime rate by town.
*   **ZN**: Proportion of residential land zoned for lots over 25,000 sq.ft.
*   **INDUS**: Proportion of non-retail business acres per town.
*   **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise).
*   **NOX**: Nitric oxides concentration (parts per 10 million).
*   **RM**: Average number of rooms per dwelling.
*   **AGE**: Proportion of owner-occupied units built prior to 1940.
*   **DIS**: Weighted distances to five Boston employment centres.
*   **RAD**: Index of accessibility to radial highways.
*   **TAX**: Full-value property-tax rate per $10,000.
*   **PTRATIO**: Pupil-teacher ratio by town.
*   **B**: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
*   **LSTAT**: % lower status of the population.

**Target Variable:**
*   **MEDV**: Median value of owner-occupied homes in $1000s.

## Exploratory Data Analysis (EDA)

A preliminary analysis was conducted to understand the dataset's characteristics. The `plot_all_histograms` function in `main.py` was used to visualize the distribution of each numerical feature. The analysis revealed that several features, such as `CRIM`, `ZN`, and `DIS`, were highly skewed. This skewness can negatively impact the performance of linear models, necessitating transformation.

![image alt](https://github.com/RITESH2127/HOUSE-PRICE-PREDICTION-TOOL/blob/main/Figure_1.png?raw=true)

## Data Preprocessing & Feature Engineering

To prepare the data for modeling, the following steps were taken, as implemented in `main.py`:
1.  **Train-Test Split**: The dataset was split into training (80%) and testing (20%) sets using `train_test_split` to ensure unbiased model evaluation.
2.  **Power Transformation**: To address the feature skewness observed during EDA, the `PowerTransformer` with the 'yeo-johnson' method was applied. This helps in making the feature distributions more Gaussian-like.
3.  **Standardization**: All features were scaled using `StandardScaler`. This step standardizes features by removing the mean and scaling to unit variance, which is crucial for the performance of linear models like Ridge and Lasso.
4.  **Pipeline Creation**: All preprocessing steps were encapsulated into `sklearn.pipeline.Pipeline` objects. This approach streamlines the workflow, prevents data leakage from the test set into the training process, and makes the model easily reproducible.

## Modeling and Evaluation

Four different regression models were trained and evaluated to find the best predictor for house prices. The `evaluate_models` function in `main.py` systematically trains and evaluates each model on two versions of the data: one with only scaling (baseline) and another with both power transformation and scaling.

**Models Trained:**
*   `LinearRegression`
*   `Ridge`
*   `Lasso`
*   `RandomForestRegressor`

**Evaluation Metrics:**
*   **R-squared (R²)**: Represents the proportion of the variance in the dependent variable that is predictable from the independent variable(s).
*   **Root Mean Squared Error (RMSE)**: Measures the square root of the average of the squared differences between predicted and actual values.

The best model was selected based on the lowest RMSE on the test set. The logic for this selection is present at the end of `main.py`, and the final model is saved to `best_model.pkl`.

## Results

The performance metrics for all models are documented in `results.csv`. The **RandomForest Regressor** demonstrated superior performance over the other models.

| Model            | Baseline R2 | Baseline RMSE | Transformed R2 | Transformed RMSE |
| ---------------- | ----------- | ------------- | -------------- | ---------------- |
| LinearRegression | 0.6687      | 24.2911       | 0.7314         | 19.6907          |
| Ridge            | 0.6684      | 24.3129       | 0.7312         | 19.7085          |
| Lasso            | 0.6501      | 25.6567       | 0.7207         | 20.4789          |
| **RandomForest** | **0.8921**  | **7.9127**    | **0.8909**     | **7.9983**       |

The results indicate that while power transformation improved the performance of linear models, the tree-based `RandomForestRegressor` performed exceptionally well even on the non-transformed (but scaled) data, achieving the lowest RMSE.

## Streamlit Web Application

The best-performing model (`RandomForestRegressor`) is deployed in an interactive web application using Streamlit. The code for the application is in `streamlit_app.py`.

**Features:**
*   Loads the saved `best_model.pkl` pipeline.
*   Provides a clean UI with input fields for all 13 features.
*   Predicts the house price in real-time when the user clicks the "Predict House Price" button.
*   Displays the predicted price in thousands of dollars.

## Project Structure
├── best_model.pkl 
# Serialized best performing model pipeline 
├── main.py 
# Main script for training, evaluation, and model saving 
├── notebook_RITESH.ipynb 
# Jupyter Notebook for initial exploration and analysis 
├── readme.md 
# This documentation file 
├── results.csv 
# CSV file containing model performance metrics 
└── streamlit_app.py 
# Script for the Streamlit web application

## Setup and Instructions to Run

To run this project on your local machine, please follow these steps:

**1. Prerequisites**
*   Python 3.8+
*   `pip` package manager

**2. Clone the Repository**
```sh
git clone https://github.com/RITESH2127/HOUSE-PRICE-PREDICTION-TOOL.git
cd HOUSE-PRICE-PREDICTION-TOOL
```

**3. Install Dependencies**
Install all the necessary Python libraries.
```sh
pip install pandas numpy scikit-learn streamlit joblib matplotlib seaborn
```

**4. Run the (main.py) Training Script to perform data preprocessing, model training, and evaluation.**
This will generate the results.csv file and save the best model as best_model.pkl.

```sh
python main.py
```

**5. Launch the Streamlit Application Run the following command in your terminal to start the web application:**

Your web browser should automatically open with the application running. If not, navigate to the local URL provided in the terminal (usually http://localhost:8501).
