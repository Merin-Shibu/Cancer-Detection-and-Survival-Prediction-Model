import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# -------------------------------
# Paths
# -------------------------------
import os

base_dir = os.path.dirname(os.path.abspath(__file__))  # script folder
train_data_dir = os.path.join(base_dir, "..", "data", "breast_cancer")



# -------------------------------
# Image Preprocessing
# -------------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 80% training, 20% validation
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# -------------------------------
# Model Building (Transfer Learning)
# -------------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model layers
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # binary classification
])

# -------------------------------
# Compile Model
# -------------------------------
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -------------------------------
# Train Model
# -------------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# -------------------------------
# Save Model
# -------------------------------
model_save_path = os.path.join(base_dir, "..", "models", "breast_stage_model.h5")


print("✅ Stage 3 Model (Benign vs Malignant) saved successfully!")


