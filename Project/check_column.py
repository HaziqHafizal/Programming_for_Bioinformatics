import pandas as pd

try:
    df = pd.read_csv('preprocessed_ckd_data.csv')
    print("--- Column Names ---")
    print(df.columns.tolist())
except FileNotFoundError:
    print("Error: File not found.")