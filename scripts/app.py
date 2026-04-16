import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import joblib

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
BASE_PATH = r"C:\Users\merin\OneDrive\Documents\cancer_detect_project"
MODEL_PATH = os.path.join(BASE_PATH, "models")

# Image-based models (Stages 1–3)
IMAGE_MODELS = {
    "stage1": os.path.join(MODEL_PATH, "normal_abnormal_model.h5"),
    "stage2": os.path.join(MODEL_PATH, "cancer_type_model.h5"),
    "breast_stage3": os.path.join(MODEL_PATH, "breast_stage_model.h5"),
}

# Clinical models (Stage 4)
STAGE4_MODELS = {
    "breast_stage": os.path.join(MODEL_PATH, "breast_stage_model.pkl"),
    "breast_survival": os.path.join(MODEL_PATH, "breast_survival_model.pkl"),
    "lung_stage": os.path.join(MODEL_PATH, "lung_stage_model.pkl"),
    "lung_survival": os.path.join(MODEL_PATH, "lung_survival_model.pkl"),
}

# -------------------------------------------------------
# LOAD MODELS
# -------------------------------------------------------
@st.cache_resource
def load_models():
    models = {}
    for key, path in IMAGE_MODELS.items():
        if os.path.exists(path):
            models[key] = tf.keras.models.load_model(path)
            models[key].compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    for key, path in STAGE4_MODELS.items():
        if os.path.exists(path):
            models[key] = joblib.load(path)
    return models

models = load_models()
st.sidebar.success("✅ Models loaded successfully!")

# -------------------------------------------------------
# IMAGE CLASSIFICATION (Stage 1–3)
# -------------------------------------------------------
def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Stage 1: Normal vs Abnormal
    pred1 = models["stage1"].predict(img_array)[0][0]
    if pred1 >= 0.5:
        return "Normal", "🟢 Normal (score={:.4f})".format(pred1)

    # Stage 2: Cancer type
    pred2 = models["stage2"].predict(img_array)[0][0]
    if pred2 < 0.5:
        # Breast Cancer
        pred3 = models["breast_stage3"].predict(img_array)[0][0]
        if pred3 < 0.5:
            return "Benign Breast Cancer", "🩵 Benign Breast Cancer (score={:.4f})".format(pred3)
        else:
            return "Malignant Breast Cancer", "❤️ Malignant Breast Cancer (score={:.4f})".format(pred3)
    else:
        # Lung Cancer (only malignant available)
        return "Malignant Lung Cancer", "🫁 Malignant Lung Cancer (auto-assumed, only malignant data)"

# -------------------------------------------------------
# APP LAYOUT
# -------------------------------------------------------
st.title("🏥 Multi-Stage Cancer Detection & Prediction System")
st.write("This app integrates image classification (Stage 1–3) with stage & survival prediction (Stage 4).")

uploaded_img = st.file_uploader("📸 Upload a histopathology image", type=["jpg", "jpeg", "png"])

if uploaded_img:
    st.image(uploaded_img, caption="Uploaded Image", use_container_width=True)
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_img.getbuffer())

    label, desc = classify_image("temp_image.jpg")
    st.subheader("🔍 Image Classification Result")
    st.info(desc)

    if "Breast" in label:
        st.success("🩷 Proceeding to Breast Cancer Stage & Survival Prediction")
    elif "Lung" in label:
        st.success("🫁 Proceeding to Lung Cancer Stage & Survival Prediction")

# -------------------------------------------------------
# STAGE 4: CLINICAL PREDICTION
# -------------------------------------------------------
st.divider()
st.header("📊 Stage 4 – Predict Cancer Stage & Survival (Clinical Data)")

cancer_type = st.selectbox("Select Cancer Type", ["Breast", "Lung"])

if cancer_type == "Breast":
    st.subheader("🩷 Breast Cancer Inputs")
    age = st.number_input("Age", min_value=10, max_value=100, value=40)
    p1 = st.number_input("Protein1", value=0.0)
    p2 = st.number_input("Protein2", value=0.0)
    p3 = st.number_input("Protein3", value=0.0)
    p4 = st.number_input("Protein4", value=0.0)

    if st.button("Predict Breast Stage & Survival"):
        df = pd.DataFrame([{
            "Age": age, "Protein1": p1, "Protein2": p2, "Protein3": p3, "Protein4": p4
        }])
        stage = models["breast_stage"].predict(df)[0]
        survival_pred = models["breast_survival"].predict_median(df)
        survival = float(survival_pred.values[0]) if hasattr(survival_pred, "values") else float(survival_pred)
        st.success(f"🩷 Predicted Stage: {stage}")
        st.info(f"📆 Estimated Median Survival: {int(survival)} days")

elif cancer_type == "Lung":
    st.subheader("🫁 Lung Cancer Inputs")
    age = st.number_input("Age", min_value=10, max_value=100, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    ph_ecog = st.number_input("PH ECOG", value=0.0)
    ph_karno = st.number_input("PH Karno", value=0.0)
    pat_karno = st.number_input("PAT Karno", value=0.0)
    meal_cal = st.number_input("Meal Calories", value=0.0)
    wt_loss = st.number_input("Weight Loss", value=0.0)

    if st.button("Predict Lung Stage & Survival"):
        df = pd.DataFrame([{
            "age": age,
            "sex": 1 if sex == "Male" else 2,
            "ph.ecog": ph_ecog,
            "ph.karno": ph_karno,
            "pat.karno": pat_karno,
            "meal.cal": meal_cal,
            "wt.loss": wt_loss,
            "time": 1,
            "status": 1
        }])
        stage = models["lung_stage"].predict(df.drop(columns=["time", "status"], errors="ignore"))[0]
        survival_pred = models["lung_survival"].predict_median(df)
        survival = float(survival_pred.values[0]) if hasattr(survival_pred, "values") else float(survival_pred)
        st.success(f"🫁 Predicted Stage: {stage}")
        st.info(f"📆 Estimated Median Survival: {int(survival)} days")


