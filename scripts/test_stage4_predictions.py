import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# ---------- PATHS ----------
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_PATH, "datasets")
MODEL_PATH = os.path.join(BASE_PATH, "models")

print(" Starting Testing for All Models...\n")

# ==========================================================
#  TEST BREAST CANCER STAGE MODEL
# ==========================================================
def test_breast_stage_model():
    print("\n Testing BREAST Cancer Stage Model...")
    df = pd.read_csv(os.path.join(DATA_PATH, "breast_clinical.csv")).dropna()
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

    #  exclude survival_5yr and cancer_type
    feature_cols = [c for c in df.columns if c not in ["stage", "cancer_type", "survival_5yr"]]
    X = df[feature_cols].select_dtypes(include=['number'])
    y_true = df["stage"]

    model = joblib.load(os.path.join(MODEL_PATH, "breast_stage_model.pkl"))
    y_pred = model.predict(X)

    print(" Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))


# ==========================================================
#  TEST BREAST CANCER SURVIVAL MODEL
# ==========================================================
def test_breast_survival_model():
    print("\n Testing BREAST Cancer Survival Model...")
    df = pd.read_csv(os.path.join(DATA_PATH, "breast_clinical.csv")).dropna()
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

    feature_cols = ["age", "tumor_size", "lymph_nodes", "metastasis"]
    X = df[feature_cols]
    y_true = df["survival_5yr"]

    model = joblib.load(os.path.join(MODEL_PATH, "breast_survival_model.pkl"))
    y_pred = model.predict(X)

    print(" Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))


# ==========================================================
#  TEST LUNG CANCER STAGE MODEL
# ==========================================================
def test_lung_stage_model():
    print("\n Testing LUNG Cancer Stage Model...")
    df = pd.read_csv(os.path.join(DATA_PATH, "lung_clinical.csv")).dropna()
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

    #  exclude survival_5yr and cancer_type
    feature_cols = [c for c in df.columns if c not in ["stage", "cancer_type", "survival_5yr"]]
    X = df[feature_cols].select_dtypes(include=['number'])
    y_true = df["stage"]

    model = joblib.load(os.path.join(MODEL_PATH, "lung_stage_model.pkl"))
    y_pred = model.predict(X)

    print(" Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))


# ==========================================================
#  TEST LUNG CANCER SURVIVAL MODEL
# ==========================================================
def test_lung_survival_model():
    print("\n Testing LUNG Cancer Survival Model...")
    df = pd.read_csv(os.path.join(DATA_PATH, "lung_clinical.csv")).dropna()
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

    feature_cols = ["age", "tumor_size", "lymph_nodes", "metastasis"]
    X = df[feature_cols]
    y_true = df["survival_5yr"]

    model = joblib.load(os.path.join(MODEL_PATH, "lung_survival_model.pkl"))
    y_pred = model.predict(X)

    print(" Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))


# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    test_breast_stage_model()
    test_breast_survival_model()
    test_lung_stage_model()
    test_lung_survival_model()
    print("\n All models tested successfully!")
