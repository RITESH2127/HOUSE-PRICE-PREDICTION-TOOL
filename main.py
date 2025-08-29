import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)
import warnings
from sklearn.exceptions import FitFailedWarning

warnings.filterwarnings("ignore", category=FitFailedWarning)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import math

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

df = pd.read_csv('https://raw.githubusercontent.com/RITESH2127/HOUSE-PRICE-PREDICTION-TOOL/refs/heads/main/housing.csv', header=None, delimiter=r"\s+", names=column_names)
df.head()

df.info()

def plot_all_histograms(df, title_prefix=""):
    num_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = 3
    n_rows = math.ceil(len(num_cols) / n_cols)

    plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    for i, col in enumerate(num_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"{title_prefix}{col}")
        plt.xlabel("")
        plt.ylabel("")

    plt.tight_layout()
    plt.show()

plot_all_histograms(df, title_prefix="Original - ")

from scipy.stats import skew

skews = df.apply(skew).sort_values(ascending=False)
print(skews)

X = df.drop("MEDV", axis=1)
y = df["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pt = PowerTransformer(method="yeo-johnson")
X_train_transformed = pt.fit_transform(X_train)
X_test_transformed = pt.transform(X_test)

X_train_transformed = pd.DataFrame(
    X_train_transformed, columns=X_train.columns, index=X_train.index
)
X_test_transformed = pd.DataFrame(
    X_test_transformed, columns=X_test.columns, index=X_test.index
)

baseline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])

transformed = Pipeline([
    ("power", PowerTransformer(method="yeo-johnson")),
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])

baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)
r2_base = r2_score(y_test, y_pred_base)
rmse_base = mean_squared_error(y_test, y_pred_base)

transformed.fit(X_train, y_train)
y_pred_trans = transformed.predict(X_test)
r2_trans = r2_score(y_test, y_pred_trans)
rmse_trans = mean_squared_error(y_test, y_pred_trans)

print({"baseline_R2": r2_base, "baseline_RMSE": rmse_base,
       "transformed_R2": r2_trans, "transformed_RMSE": rmse_trans})

def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = []
    for name, model in models.items():
        pipe_base = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])
        pipe_trans = Pipeline([
            ("power", PowerTransformer(method="yeo-johnson")),
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipe_base.fit(X_train, y_train)
        y_pred_base = pipe_base.predict(X_test)
        r2_base = r2_score(y_test, y_pred_base)
        rmse_base = mean_squared_error(y_test, y_pred_base)

        pipe_trans.fit(X_train, y_train)
        y_pred_trans = pipe_trans.predict(X_test)
        r2_trans = r2_score(y_test, y_pred_trans)
        rmse_trans = mean_squared_error(y_test, y_pred_trans)

        results.append({
            "Model": name,
            "Baseline R2": r2_base,
            "Baseline RMSE": rmse_base,
            "Transformed R2": r2_trans,
            "Transformed RMSE": rmse_trans
        })
    return pd.DataFrame(results)

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "RandomForest": RandomForestRegressor(random_state=42)
}

results_df = evaluate_models(models, X_train, X_test, y_train, y_test)
results_df
results_df.to_csv("results.csv")

# Find the minimum error across both RMSE columns
results_df["Best_RMSE"] = results_df[["Baseline RMSE", "Transformed RMSE"]].min(axis=1)

# Select the row with the minimum RMSE
best_model_row = results_df.loc[results_df["Best_RMSE"].idxmin()]

# Store model name in a variable
best_model = best_model_row["Model"]

print("Best Model:", best_model)

import joblib

joblib.dump(models[best_model], "best_model.pkl")
print("Saved best model to best_model.pkl")

