
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# --- Stage 1 Model (Normal vs Abnormal) ---
stage1_model = tf.keras.models.load_model("../models/normal_abnormal_model.h5")

stage1_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- Stage 2 Model (Cancer Type: Breast vs Lung) ---
stage2_model = tf.keras.models.load_model("../models/cancer_type_model.h5")

stage2_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# --- Function to classify image ---
def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ----- Stage 1: Normal vs Abnormal -----
    pred1 = stage1_model.predict(img_array)[0][0]

    if pred1 < 0.5:
        print(f" Stage 1 → ABNORMAL (score={pred1:.4f})")

        # ----- Stage 2: Breast vs Lung -----
        pred2 = stage2_model.predict(img_array)[0][0]

        if pred2 < 0.5:
            print(f" Stage 2 → BREAST CANCER (score={pred2:.4f})")
            return "Breast Cancer"
        else:
            print(f" Stage 2 → LUNG CANCER (score={pred2:.4f})")
            return "Lung Cancer"

    else:
        print(f" Stage 1 → NORMAL (score={pred1:.4f})")
        return "Normal"


# --- Run classification ---
if __name__ == "__main__":
    img_path = r"C:\Users\merin\OneDrive\Documents\cancer_detect_project\data\lung_cancer\malignant\lungaca368.jpeg"
    print(f" Testing image: {img_path}")
    result = classify_image(img_path)
    print(f"\n Final Result: {result}")
