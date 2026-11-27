export interface PythonSourceFile {
  path: string;
  moduleLabel: string;
  description: string;
  code: string;
}

export interface VivaItem {
  question: string;
  answer: string;
}

export const ATTRIBUTE_DICTIONARY = [
  {
    name: 'Age',
    type: 'Numerical (Years)',
    range: '21 – 86',
    role: 'Demographic risk factor; incidence increases significantly after age 50.',
  },
  {
    name: 'Gender',
    type: 'Categorical (Female / Male)',
    range: 'Binary encoded',
    role: 'Epidemiological studies report higher gallbladder carcinoma prevalence in females.',
  },
  {
    name: 'Gallstones',
    type: 'Binary (0 / 1)',
    range: '0 = Absent, 1 = Present',
    role: 'Chronic gallstone irritation is one of the strongest documented associated factors.',
  },
  {
    name: 'Wall_Thickness_mm',
    type: 'Numerical (mm)',
    range: '1.0 – 12.5',
    role: 'Normal wall thickness is < 3.0 mm; focal or diffuse thickening warrants investigation.',
  },
  {
    name: 'Polyp_Size_mm',
    type: 'Numerical (mm)',
    range: '0.0 – 22.0',
    role: 'Gallbladder polyps larger than 10 mm carry elevated clinical concern.',
  },
  {
    name: 'Chronic_Cholecystitis',
    type: 'Binary (0 / 1)',
    range: '0 = No, 1 = Yes',
    role: 'Long-standing mucosal inflammation promotes dysplastic changes.',
  },
  {
    name: 'Porcelain_Gallbladder',
    type: 'Binary (0 / 1)',
    range: '0 = No, 1 = Yes',
    role: 'Calcification of the gallbladder wall historically associated with higher malignancy risk.',
  },
  {
    name: 'Family_History',
    type: 'Binary (0 / 1)',
    range: '0 = No, 1 = Yes',
    role: 'Inherited biliary predisposition indicator.',
  },
  {
    name: 'BMI',
    type: 'Numerical (kg/m²)',
    range: '16.5 – 45.0',
    role: 'Elevated BMI correlates with cholelithiasis and metabolic risk.',
  },
  {
    name: 'Bilirubin_mg_dL',
    type: 'Numerical (mg/dL)',
    range: '0.2 – 8.5',
    role: 'Elevated serum bilirubin can indicate biliary tract obstruction.',
  },
  {
    name: 'Alkaline_Phosphatase_UL',
    type: 'Numerical (U/L)',
    range: '35 – 480',
    role: 'Cholestatic liver enzyme often elevated in biliary pathology.',
  },
  {
    name: 'Jaundice',
    type: 'Binary (0 / 1)',
    range: '0 = No, 1 = Yes',
    role: 'Clinical sign of biliary obstruction.',
  },
  {
    name: 'Abdominal_Pain',
    type: 'Binary (0 / 1)',
    range: '0 = No, 1 = Yes',
    role: 'Right upper quadrant pain commonly present in symptomatic gallstone disease.',
  },
  {
    name: 'Cancer_Risk',
    type: 'Target Binary Label',
    range: '0 = Low, 1 = Elevated',
    role: 'Supervised classification target for educational pattern modeling.',
  },
];

export const VIVA_QUESTIONS: VivaItem[] = [
  {
    question: '1. Why is Big Data Analytics (Apache Spark / PySpark) useful in healthcare datasets?',
    answer:
      'Hospital Electronic Health Records (EHRs), imaging metadata, and regional cancer registries grow to millions of rows across multi-center studies. PySpark distributes data across partitions and runs in-memory MapReduce aggregations, filtering, and schema validation much faster than single-threaded Pandas scripts.',
  },
  {
    question: '2. Are you using real hospital patient records or synthetic data?',
    answer:
      'We clearly distinguish the two: the built-in default dataset is a synthetic educational dataset generated using clinically documented risk-factor associations (such as gallstones, wall thickness, age, and chronic cholecystitis) for safe academic learning. The system also accepts external CSV uploads from public research registries such as SEER or Kaggle.',
  },
  {
    question: '3. How did you prevent Data Leakage in Module 2 and Module 5?',
    answer:
      'We performed the stratified train-test split BEFORE computing imputation medians, categorical modes, and StandardScaler Z-score parameters. All transformers are fit strictly on X_train inside Scikit-learn Pipelines and then applied unchanged to X_test.',
  },
  {
    question: '4. Why is Recall (Sensitivity) especially important in cancer risk prediction?',
    answer:
      'In medical screening prototypes, a False Negative (missing a high-risk patient) is far more costly than a False Positive (flagging a patient for follow-up ultrasound). Therefore, Recall and ROC-AUC are evaluated alongside Accuracy, especially when classes are imbalanced.',
  },
  {
    question: '5. How do Random Forest and Decision Tree differ?',
    answer:
      'A single Decision Tree learns hierarchical threshold splits on the full training set, making it interpretable but prone to overfitting. Random Forest trains an ensemble of trees on bootstrap samples with random feature subsets at each split, averaging predictions to reduce variance.',
  },
  {
    question: '6. Why do Logistic Regression and SVM require feature scaling while Decision Trees do not?',
    answer:
      'Logistic Regression and SVM optimize weight vectors and margin distances across features. Without StandardScaler, a large-scale feature like Alkaline Phosphatase (35–480 U/L) would dominate a small-scale feature like Bilirubin (0.2–8.5 mg/dL). Tree models split on rank thresholds and are scale-invariant.',
  },
  {
    question: '7. What does ROC-AUC measure?',
    answer:
      'ROC-AUC plots True Positive Rate (Recall) against False Positive Rate across every classification threshold from 0 to 1. An AUC of 0.50 equals random guessing, while 1.00 indicates perfect ranking separation between risk classes.',
  },
  {
    question: '8. How does class_weight="balanced" help with imbalanced medical datasets?',
    answer:
      'Because elevated-risk cases are a minority of total records, standard classifiers can achieve high accuracy by predicting the majority class. Balanced class weighting penalizes minority-class errors proportionally higher during training.',
  },
  {
    question: '9. Explain the Map and Reduce phases in your PySpark module.',
    answer:
      'In the Map phase, Spark splits the CSV DataFrame into worker partitions and computes partial counts, sums, and cohort filters locally per partition. In the Shuffle/Reduce phase, keys such as Gallstones=1 or Age_Group are grouped across partitions to produce global averages and prevalence rates.',
  },
  {
    question: '10. Why must the Prediction Interface include a medical disclaimer?',
    answer:
      'An educational prototype trained on synthetic or limited tabular data lacks prospective clinical validation, imaging context, and pathology confirmation. Presenting it as a confirmed diagnosis would be unsafe and ethically irresponsible.',
  },
];

export const PYTHON_PROJECT_FILES: PythonSourceFile[] = [
  {
    path: 'requirements.txt',
    moduleLabel: 'Dependencies',
    description: 'Core Python packages for PySpark, Scikit-learn, Pandas, Plotly, and Streamlit.',
    code: `pandas>=2.1.0
numpy>=1.26.0
scikit-learn>=1.4.0
pyspark>=3.5.0
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.18.0
streamlit>=1.32.0`,
  },
  {
    path: 'src/data_loader.py',
    moduleLabel: 'Module 1 · Dataset Collection',
    description: 'Generates the clearly labeled synthetic educational dataset or loads an external CSV.',
    code: `"""
Module 1: Dataset Collection & Synthetic Educational Data Generator
"""
import os
import numpy as np
import pandas as pd

FEATURE_DESCRIPTIONS = {
    "Patient_ID": "Unique anonymized educational record identifier",
    "Age": "Patient age in years (21 to 86)",
    "Gender": "Biological sex (Female / Male)",
    "Gallstones": "Presence of gallstones (1 = Yes, 0 = No)",
    "Wall_Thickness_mm": "Gallbladder wall thickness in mm (normal < 3.0 mm)",
    "Polyp_Size_mm": "Largest polyp diameter in mm (0 if absent)",
    "Chronic_Cholecystitis": "Long-term gallbladder inflammation (1 = Yes, 0 = No)",
    "Porcelain_Gallbladder": "Calcification of gallbladder wall (1 = Yes, 0 = No)",
    "Family_History": "Family history of biliary malignancy (1 = Yes, 0 = No)",
    "BMI": "Body Mass Index (kg/m^2)",
    "Bilirubin_mg_dL": "Serum total bilirubin (mg/dL)",
    "Alkaline_Phosphatase_UL": "Serum alkaline phosphatase (U/L)",
    "Jaundice": "Clinical jaundice (1 = Yes, 0 = No)",
    "Abdominal_Pain": "Right upper quadrant pain (1 = Yes, 0 = No)",
    "Cancer_Risk": "Target binary label (1 = Elevated Risk, 0 = Low Risk)",
}

def generate_synthetic_dataset(n_samples: int = 2400, seed: int = 42, inject_imperfections: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(54, 13, n_samples), 21, 86).round(0)
    gender = rng.choice(["Female", "Male"], size=n_samples, p=[0.64, 0.36])
    gallstones_prob = np.where(gender == "Female", 0.42, 0.28) + (age - 45) * 0.004
    gallstones = rng.binomial(1, np.clip(gallstones_prob, 0.1, 0.85))
    chronic_cholecystitis = rng.binomial(1, np.clip(0.12 + 0.35 * gallstones + (age - 50) * 0.003, 0.05, 0.75))
    porcelain_gb = rng.binomial(1, np.clip(0.03 + 0.08 * chronic_cholecystitis, 0.01, 0.25))
    family_history = rng.binomial(1, 0.14, size=n_samples)
    bmi = np.clip(rng.normal(26.8 + 2.2 * gallstones, 4.4, n_samples), 16.5, 45.0).round(1)
    wall_thickness = np.clip(
        rng.normal(2.2 + 1.4 * gallstones + 1.8 * chronic_cholecystitis + 1.5 * porcelain_gb, 0.9, n_samples),
        1.0, 12.5
    ).round(2)
    has_polyp = rng.binomial(1, 0.18, size=n_samples)
    polyp_size = np.where(has_polyp == 1, np.clip(rng.exponential(5.5, n_samples) + 2.0, 2.0, 22.0), 0.0).round(1)

    logit = (
        -5.4 + 0.042 * (age - 50) + 0.45 * (gender == "Female").astype(float)
        + 1.55 * gallstones + 0.58 * (wall_thickness - 2.5) + 0.16 * polyp_size
        + 1.05 * chronic_cholecystitis + 1.45 * porcelain_gb + 0.95 * family_history
        + 0.05 * (bmi - 25.0) + rng.normal(0, 0.65, n_samples)
    )
    cancer_risk = ((1.0 / (1.0 + np.exp(-logit))) >= 0.5).astype(int)
    bilirubin = np.clip(rng.normal(0.8 + 1.6 * cancer_risk + 0.5 * gallstones, 0.6, n_samples), 0.2, 8.5).round(2)
    alk_phos = np.clip(rng.normal(88 + 95 * cancer_risk + 28 * gallstones, 34, n_samples), 35, 480).round(0)
    jaundice = rng.binomial(1, np.clip(0.06 + 0.48 * cancer_risk + 0.12 * gallstones, 0.02, 0.9))
    abdominal_pain = rng.binomial(1, np.clip(0.22 + 0.42 * gallstones + 0.25 * cancer_risk, 0.1, 0.95))

    df = pd.DataFrame({
        "Patient_ID": [f"EDU-{10001 + i}" for i in range(n_samples)],
        "Age": age.astype(float), "Gender": gender, "Gallstones": gallstones.astype(float),
        "Wall_Thickness_mm": wall_thickness, "Polyp_Size_mm": polyp_size,
        "Chronic_Cholecystitis": chronic_cholecystitis, "Porcelain_Gallbladder": porcelain_gb,
        "Family_History": family_history, "BMI": bmi, "Bilirubin_mg_dL": bilirubin,
        "Alkaline_Phosphatase_UL": alk_phos, "Jaundice": jaundice,
        "Abdominal_Pain": abdominal_pain, "Cancer_Risk": cancer_risk,
    })

    if inject_imperfections:
        for col in ["Age", "Wall_Thickness_mm", "BMI", "Bilirubin_mg_dL", "Gender"]:
            missing_idx = rng.choice(df.index, size=max(1, int(0.015 * n_samples)), replace=False)
            df.loc[missing_idx, col] = np.nan
        duplicates = df.sample(n=max(5, int(0.015 * n_samples)), random_state=seed)
        df = pd.concat([df, duplicates], ignore_index=True)
    return df`,
  },
  {
    path: 'src/preprocessing.py',
    moduleLabel: 'Module 2 · Data Preprocessing',
    description: 'Removes duplicates, splits train/test before imputation/scaling, and builds a leak-free ColumnTransformer.',
    code: `"""
Module 2: Data Preprocessing Pipeline (Leak-Free)
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd

def clean_medical_dataframe(df: pd.DataFrame):
    raw_rows = len(df)
    missing_before = int(df.isna().sum().sum())
    cleaned = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = raw_rows - len(cleaned)
    if "Cancer_Risk" in cleaned.columns:
        cleaned = cleaned.dropna(subset=["Cancer_Risk"]).reset_index(drop=True)
        cleaned["Cancer_Risk"] = cleaned["Cancer_Risk"].astype(int)
    return cleaned, {
        "raw_rows": raw_rows,
        "duplicates_removed": duplicates_removed,
        "missing_values_found": missing_before,
        "cleaned_rows": len(cleaned),
    }

def prepare_train_test_data(df: pd.DataFrame, target_col="Cancer_Risk", test_size=0.2, random_state=42):
    cleaned, summary = clean_medical_dataframe(df)
    drop_cols = [c for c in ["Patient_ID", target_col] if c in cleaned.columns]
    X = cleaned.drop(columns=drop_cols)
    y = cleaned[target_col]

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(drop="if_binary", handle_unknown="ignore"))]), cat_cols),
    ])
    return X_train, X_test, y_train, y_test, preprocessor, summary`,
  },
  {
    path: 'src/spark_analytics.py',
    moduleLabel: 'Module 3 · Big Data (PySpark)',
    description: 'Uses Apache Spark DataFrames for partitioned loading, filtering, and groupBy aggregations.',
    code: `"""
Module 3: Big Data Analytics with Apache Spark (PySpark)
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def run_pyspark_analytics(csv_path: str, partitions: int = 4):
    spark = (
        SparkSession.builder
        .appName("GallbladderCancerBigDataAnalytics")
        .master(f"local[{partitions}]")
        .getOrCreate()
    )
    try:
        sdf = spark.read.option("header", "true").option("inferSchema", "true").csv(csv_path).repartition(partitions)
        sdf_clean = sdf.dropDuplicates().na.drop(subset=["Cancer_Risk"])

        gallstone_summary = (
            sdf_clean.groupBy("Gallstones")
            .agg(
                F.count("*").alias("patient_count"),
                F.round(F.avg("Wall_Thickness_mm"), 2).alias("avg_wall_thickness_mm"),
                F.round(F.mean("Cancer_Risk") * 100, 2).alias("elevated_risk_pct"),
            )
            .toPandas()
        )

        high_risk_cohort = sdf_clean.filter(
            (F.col("Age") >= 50) & (F.col("Gallstones") == 1) & (F.col("Wall_Thickness_mm") >= 3.5)
        )

        return {
            "total_records": sdf_clean.count(),
            "num_partitions": sdf_clean.rdd.getNumPartitions(),
            "high_risk_filter_count": high_risk_cohort.count(),
            "gallstone_summary": gallstone_summary,
        }
    finally:
        spark.stop()`,
  },
  {
    path: 'src/ml_models.py',
    moduleLabel: 'Module 5 · ML Comparison',
    description: 'Trains Logistic Regression, Decision Tree, Random Forest, and SVM inside leak-free pipelines.',
    code: `"""
Module 5: Machine Learning Training & Evaluation
"""
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor, random_state=42):
    specs = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced", random_state=random_state),
        "Support Vector Machine": SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=random_state),
    }

    results = {}
    for name, clf in specs.items():
        pipe = Pipeline([("preprocessor", clone(preprocessor)), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        results[name] = {
            "pipeline": pipe,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "fpr": fpr,
            "tpr": tpr,
        }
    return results`,
  },
  {
    path: 'app.py',
    moduleLabel: 'Module 7 · Streamlit App',
    description: 'Complete interactive Streamlit web application integrating all 7 modules.',
    code: `"""
Module 7: Interactive Streamlit Dashboard
Run with: streamlit run app.py
"""
import streamlit as st
from src.data_loader import load_dataset
from src.preprocessing import clean_medical_dataframe, prepare_train_test_data
from src.spark_analytics import run_pyspark_analytics
from src.ml_models import train_and_evaluate_models

st.set_page_config(page_title="Gallbladder Cancer Big Data Analytics", layout="wide")
st.title("Gallbladder Cancer Detection & Risk Prediction Using Big Data Analytics")
st.caption("Educational Big Data & Machine Learning Mini Project")

uploaded = st.sidebar.file_uploader("Upload Medical CSV", type=["csv"])
raw_df, label = load_dataset() if uploaded is None else (pd.read_csv(uploaded), "Uploaded CSV")
cleaned_df, summary = clean_medical_dataframe(raw_df)
st.info(f"Source: {label} · Cleaned Rows: {summary['cleaned_rows']}")
# Full multi-page tabs available in /python_project/app.py`,
  },
];
