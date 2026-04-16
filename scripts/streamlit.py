import streamlit as st
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# -------------------------------
# PATH SETTINGS
# -------------------------------
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "../models")

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def clip_input(val, minmax):
    return np.clip(val, *minmax)

def load_all_models():
    models = {}
    st.sidebar.title("⚙️ Model Loading")

    # Stage 1–3 image models
    try:
        models["stage1"] = tf.keras.models.load_model(os.path.join(MODEL_PATH, "normal_abnormal_model.h5"))
        st.sidebar.success("✅ Stage 1 model loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Stage 1 model missing: {e}")

    try:
        models["stage2"] = tf.keras.models.load_model(os.path.join(MODEL_PATH, "cancer_type_model.h5"))
        st.sidebar.success("✅ Stage 2 model loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Stage 2 model missing: {e}")

    try:
        models["breast_stage3"] = tf.keras.models.load_model(os.path.join(MODEL_PATH, "breast_stage_model.h5"))
        st.sidebar.success("✅ Stage 3 (breast) model loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Stage 3 breast model missing: {e}")

    # Stage 4 tabular models
    try:
        models["breast_stage"] = joblib.load(os.path.join(MODEL_PATH, "breast_stage_model.pkl"))
        models["breast_survival"] = joblib.load(os.path.join(MODEL_PATH, "breast_survival_model.pkl"))
        st.sidebar.success("✅ Stage 4 breast models loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Stage 4 breast models missing: {e}")

    try:
        models["lung_stage"] = joblib.load(os.path.join(MODEL_PATH, "lung_stage_model.pkl"))
        models["lung_survival"] = joblib.load(os.path.join(MODEL_PATH, "lung_survival_model.pkl"))
        st.sidebar.success("✅ Stage 4 lung models loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Stage 4 lung models missing: {e}")

    return models

# -------------------------------
# LOAD MODELS
# -------------------------------
models = load_all_models()

# -------------------------------
# APP TITLE
# -------------------------------
st.title(" Multi-Stage Cancer Detection & 5-Year Survival Prediction")

# -------------------------------
# IMAGE-BASED PREDICTION (Stage 1–3)
# -------------------------------
st.header(" Stage 1–3 : Image-Based Detection")
img_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if img_file:
    img = image.load_img(img_file, target_size=(224, 224))
    img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
    st.image(img, caption="Uploaded Image", width=250)

    try:
        # Stage 1 Prediction: Normal / Abnormal
        stage1_pred = models["stage1"].predict(img_array)[0][0]
        if stage1_pred >= 0.5:
            st.success(" Stage 1 : Normal Tissue Detected")
        else:
            st.warning(" Stage 1 : Abnormal Tissue Detected")

            # Stage 2 Prediction: Cancer Type
            stage2_pred = models["stage2"].predict(img_array)[0][0]
            if stage2_pred < 0.5:
                st.info(" Stage 2 : Breast Cancer Detected")
                pred3 = models["breast_stage3"].predict(img_array)[0][0]
                if pred3 < 0.5:
                    st.success(" Stage 3 : Benign Breast Cancer")
                else:
                    st.error(" Stage 3 : Malignant Breast Cancer")
            else:
                st.info(" Stage 2 : Lung Cancer Detected")
                st.error(" Stage 3 : Malignant Lung Cancer Only")
    except Exception as e:
        st.error(f"❌ Image prediction error: {e}")

# -------------------------------
# STAGE 4 : CLINICAL DATA PREDICTION
# -------------------------------
st.header(" Stage 4 : Predict Stage & Survival (Clinical Data)")

cancer_type = st.selectbox("Select Cancer Type", ["Select", "Breast", "Lung"])

# ----------------- Breast -----------------
if cancer_type == "Breast":
    age = st.number_input("Age", min_value=1, step=1)
    tumor_size = st.number_input("Tumor Size (mm)")
    lymph_nodes = st.number_input("Lymph Node Count")
    metastasis = st.selectbox("Metastasis (0=No, 1=Yes)", [0, 1])

    if st.button(" Predict Breast Cancer Stage & Survival"):
        df_stage = pd.DataFrame([{
            "age": age,
            "tumor_size": tumor_size,
            "lymph_nodes": lymph_nodes,
            "metastasis": metastasis
        }])

        try:
            # Stage prediction
            stage = models["breast_stage"].predict(df_stage)[0]

            # Survival prediction using probability
            if hasattr(models["breast_survival"], "predict_proba"):
                survival_prob = models["breast_survival"].predict_proba(df_stage)[0][1]
            else:
                survival_prob = models["breast_survival"].predict(df_stage)[0]

            # ----------------- High-risk Stage IV rule -----------------
            if stage == 4:
                # Automatically mark high-risk patients as "No" if features exceed thresholds
                if (tumor_size > 40) or (lymph_nodes > 30) or (metastasis == 1):
                    survival = "No"
                else:
                    survival = "Yes" if survival_prob > 0.8 else "No"
            else:
                survival = "Yes" if survival_prob > 0.5 else "No"

            st.write(f" **Predicted Stage:** {stage}")
            st.write(f" **Predicted 5-Year Survival:** {survival} ({survival_prob:.2f})")

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

# ----------------- Lung -----------------
elif cancer_type == "Lung":
    age = st.number_input("Age", min_value=10, max_value=100, value=50)
    tumor_size = st.number_input("Tumor Size (mm)")
    lymph_nodes = st.number_input("Lymph Node Count")
    metastasis = st.selectbox("Metastasis (0=No, 1=Yes)", [0, 1])

    if st.button("Predict Lung Cancer Stage & Survival"):
        df_stage = pd.DataFrame([{
            "age": age,
            "tumor_size": tumor_size,
            "lymph_nodes": lymph_nodes,
            "metastasis": metastasis
        }])

        try:
            # Stage prediction
            stage = models["lung_stage"].predict(df_stage)[0]

            # Survival prediction using probability
            if hasattr(models["lung_survival"], "predict_proba"):
                survival_prob = models["lung_survival"].predict_proba(df_stage)[0][1]
            else:
                survival_prob = models["lung_survival"].predict(df_stage)[0]

            # High-risk Stage IV rule
            if stage == 4:
                if (tumor_size > 40) or (lymph_nodes > 30) or (metastasis == 1):
                    survival = "No"
                else:
                    survival = "Yes" if survival_prob > 0.8 else "No"
            else:
                survival = "Yes" if survival_prob > 0.5 else "No"

            st.write(f" **Predicted Stage:** {stage}")
            st.write(f" **Predicted 5-Year Survival:** {survival} ({survival_prob:.2f})")

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
