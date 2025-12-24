import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# ==========================================
# 3.2 MODEL DEVELOPMENT
# ==========================================

print("--- Loading Data for Modeling ---")
try:
    df = pd.read_csv('preprocessed_ckd_data.csv')
    # Clean column names just in case
    df.columns = df.columns.str.strip()
    print("Data loaded. Shape:", df.shape)
except FileNotFoundError:
    print("Error: 'preprocessed_ckd_data.csv' not found.")
    exit()

# ---------------------------------------------------------
# PART A: REGRESSION ANALYSIS (Satisfying MSE/R2 requirement)
# Goal: Predict 'hemo' (Hemoglobin) based on 'pcv' (Packed Cell Volume)
# This demonstrates simple/polynomial regression.
# ---------------------------------------------------------
print("\n=== PART A: REGRESSION ANALYSIS (Predicting Hemoglobin) ===")

if 'hemo' in df.columns and 'pcv' in df.columns:
    X_reg = df[['pcv']] # Feature
    y_reg = df['hemo']  # Target

    # Split Data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    # 1. Simple Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_reg, y_train_reg)
    y_pred_lin = lin_reg.predict(X_test_reg)

    print("\n--- Simple Linear Regression Results ---")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_test_reg, y_pred_lin):.4f}")
    print(f"R-squared (R2): {r2_score(y_test_reg, y_pred_lin):.4f}")

    # 2. Polynomial Regression (Degree 2) using Pipeline
    poly_reg = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly_reg.fit(X_train_reg, y_train_reg)
    y_pred_poly = poly_reg.predict(X_test_reg)

    print("\n--- Polynomial Regression (Degree 2) Results ---")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_test_reg, y_pred_poly):.4f}")
    print(f"R-squared (R2): {r2_score(y_test_reg, y_pred_poly):.4f}")

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test_reg, y_test_reg, color='black', label='Actual Data')
    plt.plot(X_test_reg, y_pred_lin, color='blue', linewidth=2, label='Linear Regression')
    # Sort for smooth polynomial line
    sorted_zip = sorted(zip(X_test_reg.values, y_pred_poly))
    X_poly_plot, y_poly_plot = zip(*sorted_zip)
    plt.plot(X_poly_plot, y_poly_plot, color='red', linewidth=2, label='Polynomial Regression')
    plt.title('Regression Analysis: PCV vs Hemoglobin')
    plt.xlabel('Packed Cell Volume (PCV)')
    plt.ylabel('Hemoglobin')
    plt.legend()
    plt.show()
else:
    print("Skipping Regression: 'hemo' or 'pcv' columns missing.")

# ---------------------------------------------------------
# PART B: CLASSIFICATION (Main Project Goal)
# Goal: Predict 'classification' (CKD vs NotCKD)
# ---------------------------------------------------------
print("\n=== PART B: CLASSIFICATION (Predicting CKD) ===")

if 'classification' in df.columns:
    X = df.drop('classification', axis=1)
    y = df['classification']

    # Split Data (70% Train, 30% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Gaussian Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    plt.figure(figsize=(10, 8))

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] # Probability for ROC

        # Evaluation Metrics
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion Matrix Visualization
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

        # ROC Curve Calculation (Plotting all in one chart below)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(1) # Switch back to main ROC figure
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    # Finalize ROC Plot
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

else:
    print("Skipping Classification: 'classification' target column missing.")

print("\n--- Model Development Completed ---")