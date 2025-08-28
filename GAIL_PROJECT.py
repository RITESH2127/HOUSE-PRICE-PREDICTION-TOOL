import pandas as pd
import numpy as np

# Load the Boston housing dataset directly from CMU’s server
url = "http://lib.stat.cmu.edu/datasets/boston"
raw = pd.read_csv(url, sep="\s+", skiprows=22, header=None)

# Reshape the dataset into features and target
X_data = np.hstack([raw.values[::2, :], raw.values[1::2, :2]])
y_data = raw.values[1::2, 2]

# Feature names from the dataset documentation
cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
        'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

X = pd.DataFrame(X_data, columns=cols)
y = pd.Series(y_data, name="MEDV")

# Train / Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR


model_dict = {
    "LinearReg": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "ElasticNet": ElasticNet(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoost": GradientBoostingRegressor(random_state=42),
    "SVR": SVR()
}

pipelines = {
    name: Pipeline([("scale", StandardScaler()), ("estimator", model)])
    for name, model in model_dict.items()
}


trained_models = {}
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    trained_models[name] = pipe
    print(f"{name} trained.")


from sklearn.metrics import mean_squared_error, r2_score

results = {}
for name, pipe in trained_models.items():
    preds = pipe.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results[name] = {"MSE": mse, "R2": r2}
    print(f"{name}: MSE={mse:.2f}, R2={r2:.3f}")

# Pick the best one (based on MSE here)
best_model = min(results, key=lambda k: results[k]["MSE"])
print("\nBest model:", best_model)
print("Metrics:", results[best_model])


import joblib

joblib.dump(trained_models[best_model], "best_model.pkl")
print("Saved best model to best_model.pkl")
