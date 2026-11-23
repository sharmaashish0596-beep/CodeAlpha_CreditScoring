import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Credit Scoring Model", page_icon="💳", layout="wide")


# ----------------------------------------------------------------------
@st.cache_data
def make_credit_data(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    income = rng.normal(55000, 20000, n).clip(12000, 200000)
    age = rng.integers(21, 70, n)
    debt = rng.normal(15000, 9000, n).clip(0, 80000)
    late_payments = rng.poisson(1.2, n)
    credit_utilization = rng.beta(2, 3, n)
    loan_amount = rng.normal(12000, 6000, n).clip(500, 50000)
    employment_years = rng.integers(0, 35, n)

    df = pd.DataFrame(
        {
            "income": income,
            "age": age,
            "debt": debt,
            "late_payments": late_payments,
            "credit_utilization": credit_utilization,
            "loan_amount": loan_amount,
            "employment_years": employment_years,
        }
    )
    # Feature engineering
    df["debt_to_income"] = df["debt"] / df["income"]
    df["loan_to_income"] = df["loan_amount"] / df["income"]

    score = (
        2.0
        - 4.0 * df["debt_to_income"]
        - 0.7 * df["late_payments"]
        - 2.0 * df["credit_utilization"]
        - 2.0 * df["loan_to_income"]
        + 0.04 * df["employment_years"]
        + rng.normal(0, 0.8, n)
    )
    prob = 1 / (1 + np.exp(-3 * score))
    df["creditworthy"] = (rng.random(n) < prob).astype(int)
    return df


@st.cache_resource
def train_credit(model_name):
    df = make_credit_data()
    X = df.drop(columns="creditworthy")
    y = df["creditworthy"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    if model_name == "Logistic Regression":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_te, pred),
        "Precision": precision_score(y_te, pred, zero_division=0),
        "Recall": recall_score(y_te, pred, zero_division=0),
        "F1-Score": f1_score(y_te, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_te, proba),
    }
    cm = confusion_matrix(y_te, pred)
    return model, metrics, cm, list(X.columns)


def page_credit():
    st.header("💳 Credit Scoring Model")
    st.write("Past financial data se predict karta hai ki person creditworthy hai ya nahi.")
    model_name = st.selectbox(
        "Model chuno", ["Random Forest", "Logistic Regression", "Decision Tree"]
    )
    model, metrics, cm, cols = train_credit(model_name)

    st.subheader("Model Performance")
    c = st.columns(5)
    for col, (k, v) in zip(c, metrics.items()):
        col.metric(k, f"{v:.3f}")
    st.write("Confusion Matrix (rows = actual, columns = predicted)")
    st.dataframe(
        pd.DataFrame(
            cm,
            index=["Actual Bad", "Actual Good"],
            columns=["Pred Bad", "Pred Good"],
        )
    )
    if hasattr(model, "feature_importances_"):
        st.write("Feature Importance")
        st.bar_chart(pd.Series(model.feature_importances_, index=cols))

    st.subheader("Apni details daalo")
    a, b = st.columns(2)
    income = a.number_input("Yearly income", 12000, 200000, 55000, step=1000)
    age = a.number_input("Age", 21, 70, 30)
    debt = a.number_input("Total debt", 0, 80000, 12000, step=500)
    late = a.number_input("Late payments (count)", 0, 15, 1)
    util = b.slider("Credit utilization", 0.0, 1.0, 0.3)
    loan = b.number_input("Loan amount", 500, 50000, 10000, step=500)
    emp = b.number_input("Employment years", 0, 35, 5)

    if st.button("Predict Credit Score"):
        row = pd.DataFrame(
            [
                {
                    "income": income,
                    "age": age,
                    "debt": debt,
                    "late_payments": late,
                    "credit_utilization": util,
                    "loan_amount": loan,
                    "employment_years": emp,
                    "debt_to_income": debt / income,
                    "loan_to_income": loan / income,
                }
            ]
        )[cols]
        p = float(model.predict_proba(row)[0, 1])
        if p >= 0.5:
            st.success(f"Creditworthy ✅  (probability: {p:.1%})")
        else:
            st.error(f"Not creditworthy ❌  (probability of good credit: {p:.1%})")


page_credit()
