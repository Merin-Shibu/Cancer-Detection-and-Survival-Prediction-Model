import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load your model
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH, "models", "normal_abnormal_model.h5")

model = tf.keras.models.load_model(MODEL_PATH)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Path to any image you want to test
img_path = r"C:\Users\merin\OneDrive\Documents\cancer_detect_project\data\lung_cancer\malignant\lungaca29.jpeg"
# Preprocess
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)[0][0]

# Interpret result
if prediction < 0.5:
    print(f"🔴 The image is ABNORMAL (score={prediction:.4f})")
else:
    print(f"🟢 The image is NORMAL (score={prediction:.4f})")

