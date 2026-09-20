from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "housing.csv"
MODEL_PATH = ROOT / "best_model.pkl"
RESULTS_PATH = ROOT / "results.csv"
COLUMNS = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

def load_data():
    df = pd.read_csv(DATA_PATH, header=None, sep=r"\s+", names=COLUMNS)
    if df.shape != (506, 14):
        raise ValueError(f"Unexpected dataset shape: {df.shape}")
    return df

def evaluate(name, model, transformed, X_train, X_test, y_train, y_test):
    steps = [("power", PowerTransformer(method="yeo-johnson"))] if transformed else []
    steps.extend([("scaler", StandardScaler()), ("model", model)])
    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    return {"Model": name, "Variant": "Transformed" if transformed else "Baseline",
            "R2": r2_score(y_test, pred), "RMSE": mean_squared_error(y_test, pred) ** 0.5,
            "Pipeline": pipe}

def main():
    df = load_data()
    X, y = df.drop(columns="MEDV"), df["MEDV"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }
    evaluations = [evaluate(name, model, transformed, X_train, X_test, y_train, y_test)
                    for name, model in models.items() for transformed in (False, True)]
    pd.DataFrame([{k: v for k, v in row.items() if k != "Pipeline"} for row in evaluations]).to_csv(RESULTS_PATH, index=False)
    best = min(evaluations, key=lambda row: row["RMSE"])
    joblib.dump(best["Pipeline"], MODEL_PATH)
    print(f"Best model: {best['Model']} ({best['Variant']})")
    print(f"R2: {best['R2']:.4f} | RMSE: {best['RMSE']:.4f}")

if __name__ == "__main__":
    main()
