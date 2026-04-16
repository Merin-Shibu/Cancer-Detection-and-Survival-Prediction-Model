import os, shutil

# === 1. Base path ===
base_path = r"C:\Users\merin\OneDrive\Documents\cancer_detect_project\data"

# === 2. Destination folders ===
normal_dir = os.path.join(base_path, "normal_abnormal_v2", "NORMAL")
abnormal_dir = os.path.join(base_path, "normal_abnormal_v2", "ABNORMAL")

os.makedirs(normal_dir, exist_ok=True)
os.makedirs(abnormal_dir, exist_ok=True)

# === 3. Copy benign → NORMAL  ===
for cancer_type in ["breast_cancer", "lung_cancer"]:
    src = os.path.join(base_path, cancer_type, "benign")
    for img in os.listdir(src):
        shutil.copy(os.path.join(src, img), normal_dir)

# === 4. Copy malignant → ABNORMAL ===
for cancer_type in ["breast_cancer", "lung_cancer"]:
    src = os.path.join(base_path, cancer_type, "malignant")
    for img in os.listdir(src):
        shutil.copy(os.path.join(src, img), abnormal_dir)

print("✅ normal_abnormal_v2 dataset created successfully!")
print(f"NORMAL images → {normal_dir}")
print(f"ABNORMAL images → {abnormal_dir}")
