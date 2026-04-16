import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------- PATHS ----------
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_PATH, "datasets")
MODEL_PATH = os.path.join(BASE_PATH, "models")
os.makedirs(MODEL_PATH, exist_ok=True)

# ==========================================================
# BREAST CANCER STAGE MODEL
# ==========================================================
def train_breast_stage_model():
    print("\n🏥 Training BREAST Cancer Stage Model...")
    df = pd.read_csv(os.path.join(DATA_PATH, "breast_clinical.csv")).dropna()
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

    # If stage column is missing, auto-create based on tumor size
    if "stage" not in df.columns:
        df["stage"] = df["tumor_size"].apply(
            lambda x: "Stage I" if x < df["tumor_size"].median() else "Stage II"
        )

    # Exclude survival_5yr from stage model training
    feature_cols = [c for c in df.columns if c not in ["stage", "cancer_type", "survival_5yr"]]
    X = df[feature_cols].select_dtypes(include=['number'])
    y = df["stage"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))

    # Save model
    joblib.dump(model, os.path.join(MODEL_PATH, "breast_stage_model.pkl"))
    print("Breast stage model trained and saved.")

# ==========================================================
# BREAST CANCER SURVIVAL MODEL
# ==========================================================
def train_breast_survival_model():
    print("\nTraining BREAST Cancer Survival Model (Yes/No)...")
    df = pd.read_csv(os.path.join(DATA_PATH, "breast_clinical.csv")).dropna()
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

    # Ensure survival_5yr column exists
    if "survival_5yr" not in df.columns:
        df["survival_5yr"] = 1  # default dummy

    feature_cols = ["age", "tumor_size", "lymph_nodes", "metastasis"]
    X = df[feature_cols]
    y = df["survival_5yr"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(" Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))

    joblib.dump(model, os.path.join(MODEL_PATH, "breast_survival_model.pkl"))
    print(" Breast survival model trained and saved.")

# ==========================================================
#  LUNG CANCER STAGE MODEL
# ==========================================================
def train_lung_stage_model():
    print("\n Training LUNG Cancer Stage Model...")
    df = pd.read_csv(os.path.join(DATA_PATH, "lung_clinical.csv")).dropna()
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

    # If stage not present, create it from tumor size
    if "stage" not in df.columns:
        df["stage"] = df["tumor_size"].apply(
            lambda x: "Stage I" if x < df["tumor_size"].median() else "Stage II"
        )

    #  Exclude survival_5yr from stage model training
    feature_cols = [c for c in df.columns if c not in ["stage", "cancer_type", "survival_5yr"]]
    X = df[feature_cols].select_dtypes(include=['number'])
    y = df["stage"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(" Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))

    joblib.dump(model, os.path.join(MODEL_PATH, "lung_stage_model.pkl"))
    print("Lung stage model trained and saved.")

# ==========================================================
#  LUNG CANCER SURVIVAL MODEL
# ==========================================================
def train_lung_survival_model():
    print("\n Training LUNG Cancer Survival Model (Yes/No)...")
    df = pd.read_csv(os.path.join(DATA_PATH, "lung_clinical.csv")).dropna()
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

    if "survival_5yr" not in df.columns:
        df["survival_5yr"] = 1

    feature_cols = ["age", "tumor_size", "lymph_nodes", "metastasis"]
    X = df[feature_cols]
    y = df["survival_5yr"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(" Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))

    joblib.dump(model, os.path.join(MODEL_PATH, "lung_survival_model.pkl"))
    print(" Lung survival model trained and saved.")

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    print(" Starting Training for All Models...\n")
    train_breast_stage_model()
    train_breast_survival_model()
    train_lung_stage_model()
    train_lung_survival_model()
    print("\n All models trained successfully!")
