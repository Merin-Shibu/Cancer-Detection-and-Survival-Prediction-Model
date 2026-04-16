import pandas as pd

file_path = r"C:\Users\merin\OneDrive\Documents\cancer_detect_project\datasets\NCCTG_Lung_Cancer_Data_535_29.csv"

df = pd.read_csv(file_path)
print("🫁 Lung dataset shape:", df.shape)
print("\n🔹 Columns:\n", df.columns.tolist())
print("\n📊 Sample data:\n", df.head())

