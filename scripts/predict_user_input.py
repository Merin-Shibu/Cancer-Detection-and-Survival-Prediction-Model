
# test_cancer_predictor.py

import os
import joblib
import pandas as pd

# ---------- PATHS ----------
BASE_PATH = r"C:\Users\merin\OneDrive\Documents\cancer_detect_project"
MODEL_PATH = os.path.join(BASE_PATH, "models")

# ---------- LOAD MODELS ----------
def load_model(model_name):
    path = os.path.join(MODEL_PATH, model_name)
    print(f"✅ Loading model: {model_name}")
    return joblib.load(path)

try:
    breast_stage_model = load_model("breast_stage_model.pkl")
    breast_survival_model = load_model("breast_survival_model.pkl")
    lung_stage_model = load_model("lung_stage_model.pkl")
    lung_survival_model = load_model("lung_survival_model.pkl")
    print("✅ All models loaded successfully!\n")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

# ---------- PREDICTION FUNCTIONS ----------
def predict_breast_stage_and_survival(user_input: dict):
    df = pd.DataFrame([user_input])
    stage = breast_stage_model.predict(df)[0]
    survival_pred = breast_survival_model.predict_median(df)
    survival_days = getattr(survival_pred, 'values', [survival_pred])[0]
    return stage, int(survival_days)

def predict_lung_stage_and_survival(user_input: dict):
    df = pd.DataFrame([user_input])
    stage = lung_stage_model.predict(df.drop(columns=["time", "status"], errors="ignore"))[0]
    survival_pred = lung_survival_model.predict_median(df)
    survival_days = getattr(survival_pred, 'values', [survival_pred])[0]
    return stage, int(survival_days)

# ---------- MAIN CLI LOOP ----------
def main():
    print("🏥 Cancer Stage & Survival Predictor (CLI)\n")
    cancer_type = input("Select Cancer Type (Breast/Lung): ").strip().lower()

    if cancer_type == "breast":
        print("\nEnter Breast Cancer Patient Details:")
        age = float(input("Age: "))
        protein1 = float(input("Protein1: "))
        protein2 = float(input("Protein2: "))
        protein3 = float(input("Protein3: "))
        protein4 = float(input("Protein4: "))

        stage, survival = predict_breast_stage_and_survival({
            "Age": age,
            "Protein1": protein1,
            "Protein2": protein2,
            "Protein3": protein3,
            "Protein4": protein4
        })

        print(f"\n🩷 Predicted Breast Cancer Stage: {stage}")
        print(f"🩷 Predicted Median Survival (days): {survival}")

    elif cancer_type == "lung":
        print("\nEnter Lung Cancer Patient Details:")
        age = float(input("Age: "))
        sex = int(input("Sex (1=Male, 2=Female): "))
        ph_ecog = float(input("PH ECOG: "))
        ph_karno = float(input("PH Karno: "))
        pat_karno = float(input("PAT Karno: "))
        meal_cal = float(input("Meal Calories: "))
        wt_loss = float(input("Weight Loss: "))

        stage, survival = predict_lung_stage_and_survival({
            "age": age,
            "sex": sex,
            "ph.ecog": ph_ecog,
            "ph.karno": ph_karno,
            "pat.karno": pat_karno,
            "meal.cal": meal_cal,
            "wt.loss": wt_loss,
            "time": 1,    # dummy for CoxPHFitter
            "status": 1   # dummy for CoxPHFitter
        })

        print(f"\n🫁 Predicted Lung Cancer Stage: {stage}")
        print(f"🫁 Predicted Median Survival (days): {survival}")

    else:
        print("❌ Invalid cancer type! Please enter 'Breast' or 'Lung'.")

if __name__ == "__main__":
    main()
