import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- 3.0 EXPLORATORY DATA ANALYSIS (EDA) ---

# 3.1 Load Preprocessed Data
print("--- Loading Data ---")
try:
    df = pd.read_csv('preprocessed_ckd_data.csv')
    print("Data loaded successfully.")
    print("Shape:", df.shape)
    
    # CRITICAL FIX: Clean column names (remove hidden tabs/spaces)
    df.columns = df.columns.str.strip()
    print("Cleaned Column Names:", df.columns.tolist())
    
except FileNotFoundError:
    print("Error: 'preprocessed_ckd_data.csv' not found. Please run the Progress 2 script first.")
    exit()

# 3.2 Descriptive Statistics
print("\n--- 3.2 Descriptive Statistics ---")
# Select columns that exist in the dataframe
target_cols = ['age', 'bp', 'bgr', 'bu', 'sc', 'hemo', 'pcv', 'wc', 'rc', 'classification']
available_cols = [c for c in target_cols if c in df.columns]

if available_cols:
    print(df[available_cols].describe())
    print("\n--- Variance ---")
    print(df[available_cols].var())
    print("\n--- Mode ---")
    print(df[available_cols].mode().iloc[0])
else:
    print("Warning: None of the target columns were found!")

# 3.3 Grouping
print("\n--- 3.3 Grouping Analysis ---")
if 'classification' in df.columns:
    print("Mean values grouped by Classification (0=NotCKD, 1=CKD):")
    print(df.groupby('classification')[available_cols].mean())
    
    # Visualize Grouping (Boxplot for Hemoglobin)
    if 'hemo' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='classification', y='hemo', data=df)
        plt.title('Hemoglobin Levels by CKD Status')
        plt.xlabel('Classification (0=NotCKD, 1=CKD)')
        plt.ylabel('Hemoglobin (Normalized)')
        plt.show()
    else:
        print("Skipping Boxplot: 'hemo' column not found.")
else:
    print("Skipping Grouping: 'classification' column not found.")

# 3.4 ANOVA (Analysis of Variance)
print("\n--- 3.4 ANOVA Test ---")
if 'classification' in df.columns and 'hemo' in df.columns:
    ckd_hemo = df[df['classification'] == 1]['hemo']
    notckd_hemo = df[df['classification'] == 0]['hemo']
    
    f_val, p_val = stats.f_oneway(ckd_hemo, notckd_hemo)
    print(f"ANOVA for Hemoglobin: F-value={f_val:.4f}, p-value={p_val:.4e}")
    
    if p_val < 0.05:
        print("Result: Significant difference in Hemoglobin between groups (Reject H0).")
    else:
        print("Result: No significant difference (Fail to reject H0).")
else:
    print(f"Skipping ANOVA. Missing columns. Have: classification={ 'classification' in df.columns}, hemo={ 'hemo' in df.columns}")

# 3.5 Correlation Matrix
print("\n--- 3.5 Correlation Matrix ---")
if 'classification' in df.columns:
    # Correlation of all numerical features
    corr_matrix = df.corr()

    # Filter for top correlations with 'classification'
    target_corr = corr_matrix['classification'].sort_values(ascending=False)
    print("Top 5 Features correlated with Classification:")
    print(target_corr.head(6)) 
    print("\nBottom 5 Features correlated with Classification:")
    print(target_corr.tail(5))

    # Plot Heatmap
    plt.figure(figsize=(12, 10))
    # Select top 15 most correlated columns for readability
    top_cols = target_corr.abs().nlargest(15).index
    sns.heatmap(df[top_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap (Top 15 Features)')
    plt.show()
else:
    print("Skipping Correlation: 'classification' column not found.")

print("\n--- EDA Completed ---")