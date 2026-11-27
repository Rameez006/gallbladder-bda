"""
Module 7: Interactive Streamlit Dashboard
-----------------------------------------
Title: Gallbladder Cancer Detection and Risk Prediction Using Big Data Analytics
Run with: streamlit run app.py
"""

import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import FEATURE_DESCRIPTIONS, load_dataset
from src.eda_visualization import (
    plot_age_distribution,
    plot_correlation_heatmap,
    plot_gallstones_and_gender,
    plot_wall_thickness_box,
)
from src.ml_models import train_and_evaluate_models
from src.predictor import EDUCATIONAL_DISCLAIMER, predict_patient_risk
from src.preprocessing import clean_medical_dataframe, prepare_train_test_data
from src.spark_analytics import run_pyspark_analytics

st.set_page_config(
    page_title="Gallbladder Cancer Big Data Analytics",
    layout="wide",
)

st.title("Gallbladder Cancer Detection & Risk Prediction Using Big Data Analytics")
st.caption(
    "College Big Data & Machine Learning Mini Project · Educational Demonstration Prototype"
)

# Sidebar controls
st.sidebar.header("Dataset & Pipeline Settings")
uploaded_file = st.sidebar.file_uploader("Upload Medical CSV (Optional)", type=["csv"])
sample_size = st.sidebar.slider("Synthetic Dataset Size", 1000, 10000, 2400, step=200)
test_split = st.sidebar.slider("Test Split Ratio", 0.15, 0.35, 0.20, step=0.05)

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    data_source_label = f"Uploaded CSV ({len(raw_df)} rows)"
    temp_csv = os.path.join("data", "uploaded_temp.csv")
    os.makedirs("data", exist_ok=True)
    raw_df.to_csv(temp_csv, index=False)
    active_csv_path = temp_csv
else:
    raw_df, data_source_label = load_dataset(n_samples=sample_size)
    active_csv_path = os.path.join("data", "sample_gallbladder_dataset.csv")

cleaned_df, clean_summary = clean_medical_dataframe(raw_df)

section = st.sidebar.radio(
    "Navigate Modules",
    [
        "1. Project Overview & Dataset",
        "2. Big Data Analytics (PySpark)",
        "3. Exploratory Data Analysis",
        "4. ML Model Comparison",
        "5. Risk Prediction Interface",
        "6. Conclusion & Viva Summary",
    ],
)

if section == "1. Project Overview & Dataset":
    st.subheader("Project Overview & Data Preprocessing")
    st.info(
        f"Active Source: **{data_source_label}**. Built-in data is synthetic and "
        "created strictly for educational Big Data & ML coursework."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw Records", clean_summary["raw_rows"])
    c2.metric("Duplicates Removed", clean_summary["duplicates_removed"])
    c3.metric("Missing Cells Detected", clean_summary["missing_values_found"])
    c4.metric("Clean Records Ready", clean_summary["cleaned_rows"])

    st.markdown("### Clinical Attribute Dictionary")
    dict_df = pd.DataFrame(
        [{"Attribute": k, "Description": v} for k, v in FEATURE_DESCRIPTIONS.items()]
    )
    st.dataframe(dict_df, use_container_width=True, hide_index=True)

    st.markdown("### Sample Cleaned Dataset Preview")
    st.dataframe(cleaned_df.head(15), use_container_width=True)

elif section == "2. Big Data Analytics (PySpark)":
    st.subheader("Distributed Processing with Apache Spark (PySpark)")
    partitions = st.select_slider("Spark Worker Partitions", options=[2, 4, 8, 16], value=4)
    if st.button("Run PySpark Distributed Aggregation", type="primary"):
        with st.spinner("Running PySpark DataFrame transformations..."):
            spark_res = run_pyspark_analytics(active_csv_path, partitions=partitions)

        m1, m2, m3 = st.columns(3)
        m1.metric("Distributed Records Processed", spark_res["total_records"])
        m2.metric("Spark Partitions", spark_res["num_partitions"])
        m3.metric("High-Risk Filtered Cohort", spark_res["high_risk_filter_count"])

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Aggregation by Gallstone Status")
            st.dataframe(spark_res["gallstone_summary"], use_container_width=True, hide_index=True)
        with col_b:
            st.markdown("#### Aggregation by Age Bracket")
            st.dataframe(spark_res["age_bracket_summary"], use_container_width=True, hide_index=True)

elif section == "3. Exploratory Data Analysis":
    st.subheader("Exploratory Data Analysis (EDA)")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_age_distribution(cleaned_df), use_container_width=True)
    with col2:
        st.plotly_chart(plot_gallstones_and_gender(cleaned_df), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(plot_wall_thickness_box(cleaned_df), use_container_width=True)
    with col4:
        st.plotly_chart(plot_correlation_heatmap(cleaned_df), use_container_width=True)

elif section == "4. ML Model Comparison":
    st.subheader("Machine Learning Model Training & Comparison")
    X_train, X_test, y_train, y_test, preprocessor, report = prepare_train_test_data(
        raw_df, test_size=test_split
    )
    st.write(
        f"Train Rows: **{report.train_rows}** | Test Rows: **{report.test_rows}** "
        "(Stratified split; scalers fit strictly on training data to avoid leakage.)"
    )

    evals = train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor)

    metrics_rows = [
        {
            "Model": m.name,
            "Accuracy": round(m.accuracy, 4),
            "Precision": round(m.precision, 4),
            "Recall": round(m.recall, 4),
            "F1-Score": round(m.f1, 4),
            "ROC-AUC": round(m.roc_auc, 4),
        }
        for m in evals.values()
    ]
    st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)

    roc_fig = go.Figure()
    for m in evals.values():
        roc_fig.add_trace(
            go.Scatter(x=m.fpr, y=m.tpr, mode="lines", name=f"{m.name} (AUC={m.roc_auc:.3f})")
        )
    roc_fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline", line=dict(dash="dash"))
    )
    roc_fig.update_layout(
        title="ROC-AUC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
    )
    st.plotly_chart(roc_fig, use_container_width=True)

    selected_name = st.selectbox("Inspect Model Confusion Matrix & Feature Weights", list(evals.keys()))
    selected_eval = evals[selected_name]
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("#### Confusion Matrix (Test Set)")
        cm_df = pd.DataFrame(
            selected_eval.confusion_matrix,
            index=["Actual Low (0)", "Actual Elevated (1)"],
            columns=["Pred Low (0)", "Pred Elevated (1)"],
        )
        st.dataframe(cm_df, use_container_width=True)
    with c_right:
        st.markdown("#### Top Contributing Features")
        st.dataframe(selected_eval.feature_importances.head(8), use_container_width=True, hide_index=True)

elif section == "5. Risk Prediction Interface":
    st.subheader("Educational Risk Prediction Interface")
    st.warning(EDUCATIONAL_DISCLAIMER)

    X_train, X_test, y_train, y_test, preprocessor, _ = prepare_train_test_data(
        raw_df, test_size=test_split
    )
    evals = train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor)

    model_choice = st.selectbox("Select Trained Model", list(evals.keys()), index=2)

    with st.form("patient_risk_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age (years)", 20, 90, 58)
        gender = c2.selectbox("Gender", ["Female", "Male"])
        bmi = c3.number_input("BMI (kg/m²)", 15.0, 50.0, 28.4)

        gallstones = c1.selectbox("Gallstones Present", [0, 1], index=1)
        wall_thickness = c2.number_input("Wall Thickness (mm)", 1.0, 15.0, 4.2, step=0.1)
        polyp_size = c3.number_input("Polyp Size (mm)", 0.0, 25.0, 0.0, step=0.5)

        chronic_chol = c1.selectbox("Chronic Cholecystitis", [0, 1], index=1)
        porcelain = c2.selectbox("Porcelain Gallbladder", [0, 1], index=0)
        family_hist = c3.selectbox("Family History", [0, 1], index=0)

        bilirubin = c1.number_input("Bilirubin (mg/dL)", 0.2, 10.0, 1.4, step=0.1)
        alk_phos = c2.number_input("Alkaline Phosphatase (U/L)", 30, 500, 135)
        jaundice = c3.selectbox("Jaundice", [0, 1], index=0)
        abd_pain = c1.selectbox("Abdominal Pain", [0, 1], index=1)

        submitted = st.form_submit_button("Estimate Risk Tier", type="primary")

    if submitted:
        sample = {
            "Age": age,
            "Gender": gender,
            "Gallstones": gallstones,
            "Wall_Thickness_mm": wall_thickness,
            "Polyp_Size_mm": polyp_size,
            "Chronic_Cholecystitis": chronic_chol,
            "Porcelain_Gallbladder": porcelain,
            "Family_History": family_hist,
            "BMI": bmi,
            "Bilirubin_mg_dL": bilirubin,
            "Alkaline_Phosphatase_UL": alk_phos,
            "Jaundice": jaundice,
            "Abdominal_Pain": abd_pain,
        }
        result = predict_patient_risk(evals[model_choice], sample)
        st.success(f"Classification: **{result['risk_tier']}** | Model Probability: **{result['probability_pct']}%**")
        st.dataframe(result["top_features"], use_container_width=True, hide_index=True)

else:
    st.subheader("Conclusion & Future Enhancements")
    st.markdown(
        """
        - **Summary**: This project demonstrates an end-to-end Big Data and Machine Learning workflow combining PySpark distributed aggregations, leak-free Scikit-learn pipelines, and interactive Plotly/Streamlit analytics.
        - **Key Educational Finding**: In synthetic demonstration data aligned with clinical literature, gallstones, increased gallbladder wall thickness, chronic inflammation, and age emerge as strong statistical risk signals.
        - **Future Scope**: Integration with real hospital EHR registries (under IRB approval), imaging CNNs on ultrasound scans, SHAP explainability, and distributed Spark MLlib pipelines.
        """
    )
