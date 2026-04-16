import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# ---------------------------------
# Load all trained models
# ---------------------------------
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../models")

stage1_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "normal_abnormal_model.h5"))
stage2_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "cancer_type_model.h5"))
stage3_breast_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "breast_stage_model.h5"))



# Compile (optional for predictions)
stage1_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
stage2_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
stage3_breast_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ---------------------------------
# Image classification function
# ---------------------------------
def classify_image(img_path):
    # Preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ---- Stage 1: Normal vs Abnormal ----
    pred1 = stage1_model.predict(img_array)[0][0]
    if pred1 >= 0.5:
        print(f"🟢 Stage 1 → NORMAL (score={pred1:.4f})")
        return "Normal"
    else:
        print(f"🔴 Stage 1 → ABNORMAL (score={pred1:.4f})")

        # ---- Stage 2: Breast vs Lung ----
        pred2 = stage2_model.predict(img_array)[0][0]
        if pred2 < 0.5:
            print(f"🩷 Stage 2 → BREAST CANCER (score={pred2:.4f})")

            # ---- Stage 3: Benign vs Malignant ----
            pred3 = stage3_breast_model.predict(img_array)[0][0]
            if pred3 < 0.5:
                print(f"🩵 Stage 3 → BENIGN BREAST CANCER (score={pred3:.4f})")
                return "Benign Breast Cancer"
            else:
                print(f"❤️ Stage 3 → MALIGNANT BREAST CANCER (score={pred3:.4f})")
                return "Malignant Breast Cancer"
        else:
            print(f"🫁 Stage 2 → LUNG CANCER (score={pred2:.4f})")
            print(f"🔴 Stage 3 → MALIGNANT LUNG CANCER")
            return "Malignant Lung Cancer"

# ---------------------------------
# Example usage:
# ---------------------------------
# Replace this path with your test image path
classify_image(r"C:\Users\merin\OneDrive\Documents\cancer_detect_project\data\normal_abnormal_v2\NORMAL\lungn30.jpeg")
