from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "best_model.pkl"

def test_model_loads_and_predicts():
    model = joblib.load(MODEL)
    X = pd.DataFrame([[0.1, 0.0, 7.0, 0, 0.5, 6.0, 60.0, 4.0, 1.0, 300.0, 18.0, 390.0, 12.0]],
                     columns=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"])
    prediction = model.predict(X)
    assert len(prediction) == 1
