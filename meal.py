import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

from mealpy.swarm_based.FFA import BaseFFA
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Feature Selection Comparison", layout="wide")

st.title("🧪 Comparative Analysis: Correlation vs Firefly Feature Selection")
st.markdown("### Logistic Regression Classification on Diabetes Dataset")

uploaded_file = st.file_uploader("Upload Diabetes CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Dataset:")
    st.dataframe(df.head())

    target_col = st.selectbox("Select the Target Column", options=df.columns)
    y = df[target_col]
    X = df.drop(columns=[target_col])

    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    ##################
    # CORRELATION FS #
    ##################
    def correlation_selection(X, y, threshold=0.1):
        corr = X.corrwith(pd.Series(y)).abs()
        return corr[corr > threshold].index.tolist()

    st.subheader("🔹 Correlation-Based Feature Selection")
    start_corr = time.time()
    corr_features = correlation_selection(X, y)
    end_corr = time.time()
    X_corr = X[corr_features]

    ################
    # FIREFLY FS   #
    ################
    st.subheader("🔸 Firefly Feature Selection")

    def firefly_fitness(solution):
        selected = [i for i, bit in enumerate(solution) if bit > 0.5]
        if len(selected) == 0:
            return 1
        X_sel = X.iloc[:, selected]
        X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.3, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        return 1 - accuracy_score(y_test, pred)

    problem = {
        "fit_func": firefly_fitness,
        "lb": [0] * X.shape[1],
        "ub": [1] * X.shape[1],
        "minmax": "min",
    }

    start_firefly = time.time()
    model_fa = BasedFFA(epoch=30, pop_size=20)
    best_pos, _ = model_fa.solve(problem)
    end_firefly = time.time()
    firefly_indices = [i for i, bit in enumerate(best_pos) if bit > 0.5]
    X_firefly = X.iloc[:, firefly_indices]

    ####################
    # TRAIN AND COMPARE
    ####################

    def evaluate_model(X_sel, y, title):
        X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.3, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
        }

        cm = confusion_matrix(y_test, pred)

        return metrics, cm, classification_report(y_test, pred, output_dict=True)

    corr_metrics, corr_cm, corr_report = evaluate_model(X_corr, y, "Correlation")
    firefly_metrics, firefly_cm, firefly_report = evaluate_model(X_firefly, y, "Firefly")

    ####################
    # DISPLAY RESULTS
    ####################

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✅ Correlation-Based Logistic Regression")
        st.markdown(f"⏱️ Time: **{end_corr - start_corr:.2f}s**")
        st.write(corr_metrics)

        fig1, ax1 = plt.subplots()
        sns.heatmap(corr_cm, annot=True, fmt='d', cmap="Blues", ax=ax1)
        ax1.set_title("Correlation Confusion Matrix")
        st.pyplot(fig1)

    with col2:
        st.markdown("### ✅ Firefly-Based Logistic Regression")
        st.markdown(f"⏱️ Time: **{end_firefly - start_firefly:.2f}s**")
        st.write(firefly_metrics)

        fig2, ax2 = plt.subplots()
        sns.heatmap(firefly_cm, annot=True, fmt='d', cmap="Oranges", ax=ax2)
        ax2.set_title("Firefly Confusion Matrix")
        st.pyplot(fig2)

    ########################
    # METRICS COMPARISON
    ########################

    st.markdown("### 📊 Comparative Bar Chart of Metrics")
    metrics_df = pd.DataFrame({
        "Metric": list(corr_metrics.keys()),
        "Correlation-Based": list(corr_metrics.values()),
        "Firefly-Based": list(firefly_metrics.values())
    })

    metrics_df.set_index("Metric", inplace=True)
    st.bar_chart(metrics_df)

    #############################
    # SHOW SELECTED FEATURES
    #############################

    st.markdown("### 🧬 Selected Features")
    st.markdown(f"**Correlation ({len(corr_features)} features)**: {corr_features}")
    st.markdown(f"**Firefly ({len(firefly_indices)} features)**: {[X.columns[i] for i in firefly_indices]}")

