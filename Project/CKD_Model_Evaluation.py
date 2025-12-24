import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

# ==========================================
# 3.3 MODEL EVALUATION & REFINEMENT
# ==========================================

print("--- Loading Data ---")
try:
    df = pd.read_csv('preprocessed_ckd_data.csv')
    df.columns = df.columns.str.strip()
    print("Data loaded. Shape:", df.shape)
except FileNotFoundError:
    print("Error: 'preprocessed_ckd_data.csv' not found.")
    exit()

# ---------------------------------------------------------
# PART A: CLASSIFICATION EVALUATION (Grid Search & Overfitting)
# Focus: Random Forest (Best model from Progress 4)
# ---------------------------------------------------------
print("\n=== PART A: MODEL REFINEMENT (Random Forest) ===")

if 'classification' in df.columns:
    X = df.drop('classification', axis=1)
    y = df['classification']

    # Split Data (70% Train, 30% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 1. GRID SEARCH (Hyperparameter Tuning)
    # We test different "settings" for the Random Forest to find the best combo
    print("--- Performing Grid Search ---")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    print(f"Best Parameters found: {grid_search.best_params_}")

    # 2. EVALUATE BEST MODEL
    y_pred = best_rf.predict(X_test)
    print(f"\nRefined Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report (Refined):")
    print(classification_report(y_test, y_pred))

    # 3. CHECK FOR OVERFITTING (Cross-Validation)
    # If CV Score is significantly lower than Train Score, it's overfitting.
    cv_scores = cross_val_score(best_rf, X, y, cv=10)
    print(f"\n--- Overfitting Analysis ---")
    print(f"10-Fold Cross-Validation Mean Score: {cv_scores.mean():.4f}")
    print(f"Training Score (on full train set): {best_rf.score(X_train, y_train):.4f}")
    
    if best_rf.score(X_train, y_train) - cv_scores.mean() > 0.1:
        print("Result: POTENTIAL OVERFITTING DETECTED (High Train score, Lower CV score)")
    else:
        print("Result: Model is Generalized (No significant overfitting)")

else:
    print("Skipping Classification: 'classification' column missing.")

# ---------------------------------------------------------
# PART B: RIDGE REGRESSION (Syllabus Requirement)
# Focus: Predicting Hemoglobin (hemo) using PCV
# ---------------------------------------------------------
print("\n=== PART B: RIDGE REGRESSION ANALYSIS ===")

if 'hemo' in df.columns and 'pcv' in df.columns:
    X_reg = df[['pcv']]
    y_reg = df['hemo']
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    # 1. Standard Linear Regression (for comparison)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_reg, y_train_reg)
    y_pred_lin = lin_reg.predict(X_test_reg)
    mse_lin = mean_squared_error(y_test_reg, y_pred_lin)

    # 2. Ridge Regression
    # Ridge adds a penalty (alpha) to prevent overfitting in regression
    ridge_reg = Ridge(alpha=1.0) 
    ridge_reg.fit(X_train_reg, y_train_reg)
    y_pred_ridge = ridge_reg.predict(X_test_reg)
    mse_ridge = mean_squared_error(y_test_reg, y_pred_ridge)

    print(f"Linear Regression MSE: {mse_lin:.6f}")
    print(f"Ridge Regression MSE:  {mse_ridge:.6f}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test_reg, y_test_reg, color='gray', label='Actual Data')
    plt.plot(X_test_reg, y_pred_lin, color='blue', linestyle='--', label='Linear Regression')
    plt.plot(X_test_reg, y_pred_ridge, color='red', label='Ridge Regression')
    plt.title('Linear vs Ridge Regression (Hemoglobin Prediction)')
    plt.xlabel('Packed Cell Volume (PCV)')
    plt.ylabel('Hemoglobin')
    plt.legend()
    plt.show()

else:
    print("Skipping Ridge Regression: 'hemo' or 'pcv' columns missing.")

print("\n--- Model Evaluation Completed ---")