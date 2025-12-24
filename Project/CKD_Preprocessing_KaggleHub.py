import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import kagglehub
import os

# ==========================================
# 2.1 IMPORTING DATASET
# ==========================================

print("--- 2.1.1 Downloading Data ---")
path = kagglehub.dataset_download("mansoordaku/ckdisease")
print("Path to dataset files:", path)

# FIND the CSV file
csv_file = None
for file in os.listdir(path):
    if file.endswith(".csv"):
        csv_file = os.path.join(path, file)
        break

if csv_file:
    print(f"Found CSV file: {csv_file}")
    # Load dataset (Handling '?' and '\t?' as NaN)
    df = pd.read_csv(csv_file, na_values=['?', '\t?'])
else:
    raise FileNotFoundError("No CSV file found!")

print("\n--- 2.1.2 Dataset Sample ---")
print(df.head())
print("\nDataset Shape:", df.shape)

# ==========================================
# 2.2 DATA WRANGLING
# ==========================================

print("\n--- 2.2.1 Data Cleaning ---")
# Drop ID
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# DEFINE which columns are numbers and which are text
# This prevents destroying the text data
cols_to_force_numeric = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'sg', 'al', 'su']

# Clean only the numeric columns
for col in cols_to_force_numeric:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("Data Types Checked.")
print(df.dtypes)

# 2.2.3 Handling Missing Values
print("\n--- 2.2.3 Missing Values Before Handling ---")
print(df.isnull().sum())

# Impute Missing Values
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(exclude=np.number).columns

# Fill numbers with MEAN
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

# Fill text with MODE (Most Frequent)
for col in cat_cols:
    if not df[col].mode().empty:
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna("Unknown") # Fallback

print("\n--- Missing Values After Handling ---")
# This should be 0 now!
print("Total Missing:", df.isnull().sum().sum())

# 2.2.4 Data Normalization
scaler = MinMaxScaler()
cols_to_scale = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']

# Ensure columns exist before scaling
existing_cols_to_scale = [c for c in cols_to_scale if c in df.columns]
df_scaled = df.copy()
df_scaled[existing_cols_to_scale] = scaler.fit_transform(df_scaled[existing_cols_to_scale])

print("\n--- 2.2.4 Normalized Sample ---")
print(df_scaled[['age', 'bp']].head())

# 2.2.5 Data Binning
bins = [0, 40, 60, 100]
labels = ['Young Adult', 'Middle Aged', 'Senior']
df_scaled['Age_Group'] = pd.cut(df['age'], bins=bins, labels=labels)

print("\n--- 2.2.5 Data Binning Sample ---")
print(df_scaled[['age', 'Age_Group']].head())

# 2.2.6 Indicator Variables
# Clean Target Variable
if 'classification' in df_scaled.columns:
    df_scaled['classification'] = df_scaled['classification'].astype(str).str.strip()
    df_scaled['classification'] = df_scaled['classification'].map({'ckd': 1, 'notckd': 0, 'ckd\t': 1})
    # Fill any mapping errors with 0
    df_scaled['classification'] = df_scaled['classification'].fillna(0)

# Get Dummies
df_final = pd.get_dummies(df_scaled, columns=['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'Age_Group'], drop_first=True)

print("\n--- 2.2.6 Indicator Variables Sample ---")
print(df_final.head())

# 2.2.7 Export
output_filename = 'preprocessed_ckd_data.csv'
df_final.to_csv(output_filename, index=False)
print(f"\n--- SUCCESS: Data Exported to {output_filename} ---")