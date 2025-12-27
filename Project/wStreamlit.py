import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(layout="wide")
st.title("Modelling")

st.write("Please wait while the model loads and trains...")

# ===============================
# Load dataset
# ===============================
df = pd.read_csv("../preprocessed_ckd_data.csv")

TARGET_COLUMN = "classification"

# ===============================
# Tabs
# ===============================
tab1, tab2 = st.tabs([
    "Random Forest vs Extra Trees",
    "Linear vs Ridge Regression"
])

# ======================================================
# TAB 1: RANDOM FOREST vs EXTRA TREES (CLASSIFICATION)
# ======================================================
with tab1:

    st.header("Training vs Validation Accuracy")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    n_estimators_list = [10, 30, 50, 100, 150]

    rf_train_acc, rf_val_acc = [], []
    et_train_acc, et_val_acc = [], []

    for n in n_estimators_list:

        rf = RandomForestClassifier(
            n_estimators=n,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        rf_train_acc.append(accuracy_score(y_train, rf.predict(X_train)))
        rf_val_acc.append(accuracy_score(y_val, rf.predict(X_val)))

        et = ExtraTreesClassifier(
            n_estimators=n,
            random_state=42,
            n_jobs=-1
        )
        et.fit(X_train, y_train)

        et_train_acc.append(accuracy_score(y_train, et.predict(X_train)))
        et_val_acc.append(accuracy_score(y_val, et.predict(X_val)))

    # -------------------------
    # Plot (OVERLAP FIXED)
    # -------------------------
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        n_estimators_list, rf_train_acc,
        label="RF Train",
        color="tab:blue",
        linewidth=3,
        marker="o",
        alpha=0.85
    )

    ax.plot(
        n_estimators_list, rf_val_acc,
        label="RF Validation",
        color="tab:blue",
        linestyle="--",
        linewidth=3,
        marker="s"
    )

    ax.plot(
        n_estimators_list, et_train_acc,
        label="ET Train",
        color="tab:green",
        linewidth=3,
        marker="^",
        alpha=0.85
    )

    ax.plot(
        n_estimators_list, et_val_acc,
        label="ET Validation",
        color="tab:red",
        linestyle="--",
        linewidth=3,
        marker="x"
    )

    ax.set_xlabel("Number of Trees (n_estimators)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training vs Validation Accuracy")
    ax.set_ylim(0.9, 1.01)
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    # -------------------------
    # Accuracy Table
    # -------------------------
    results = pd.DataFrame({
        "n_estimators": n_estimators_list,
        "RF Train Accuracy (%)": np.array(rf_train_acc) * 100,
        "RF Validation Accuracy (%)": np.array(rf_val_acc) * 100,
        "ET Train Accuracy (%)": np.array(et_train_acc) * 100,
        "ET Validation Accuracy (%)": np.array(et_val_acc) * 100
    })

    st.subheader("Accuracy Summary (%)")
    st.dataframe(results.round(2), use_container_width=True)

# ======================================================
# TAB 2: LINEAR vs RIDGE REGRESSION (ONE PLOT ONLY)
# ======================================================
with tab2:

    st.header("Hemoglobin Prediction")

    X_reg = df[["pcv"]]
    y_reg = df["hemo"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg,
        test_size=0.2,
        random_state=42
    )

    lin = LinearRegression()
    lin.fit(X_train, y_train)
    y_pred_lin = lin.predict(X_test)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)

    # -------------------------
    # Regression Metrics Table
    # -------------------------
    reg_results = pd.DataFrame({
        "Model": ["Linear Regression", "Ridge Regression"],
        "R² Score": [
            r2_score(y_test, y_pred_lin),
            r2_score(y_test, y_pred_ridge)
        ],
        "MSE": [
            mean_squared_error(y_test, y_pred_lin),
            mean_squared_error(y_test, y_pred_ridge)
        ]
    })

    st.subheader("Regression Performance Summary")
    st.dataframe(reg_results.round(4), use_container_width=True)

    # -------------------------
    # SINGLE Regression Plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        X_test, y_test,
        label="Actual",
        alpha=0.6
    )

    ax.plot(
        X_test, y_pred_lin,
        label="Linear Regression",
        color="tab:blue",
        linewidth=2
    )

    ax.plot(
        X_test, y_pred_ridge,
        label="Ridge Regression",
        color="tab:red",
        linestyle="--",
        linewidth=2
    )

    ax.set_xlabel("PCV")
    ax.set_ylabel("Hemoglobin")
    ax.set_title("Linear vs Ridge Regression")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
